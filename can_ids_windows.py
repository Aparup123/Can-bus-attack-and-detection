"""
can_ids_windows.py
Self-contained CAN bus simulator + feature extractor + RandomForest IDS trainer + live detector.
Run on Windows with:
    python -m venv venv
    venv\\Scripts\\activate
    pip install numpy pandas scikit-learn scipy joblib
    python can_ids_windows.py
"""

import threading, time, random, math
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump, load

# ---------- Simulation settings ----------
SIM_DURATION = 40.0        # total seconds to simulate (including attack periods)
NORMAL_RATE = 0.05         # average seconds between normal messages per producer (lower -> faster)
WINDOW_S = 2.0             # window size in seconds for features
WINDOW_STEP = 1.0          # step between windows (overlap if step < WINDOW_S)

# IDs and their normal payload generators
NORMAL_IDS = [0x100, 0x200, 0x300]
def normal_payload_gen(id_):
    # simple deterministic but varying payload
    base = int(time.time() * 1000) % 256
    return [(base + id_//0x100 + i) % 256 for i in range(8)]

# Attack schedule: each item = (start_s, end_s, type)
# We place attacks in the simulation timeline; windows that overlap attack periods are labeled attack.
ATTACK_SCHEDULE = [
    (8.0, 12.0, "spoof"),   # spoofing period
    (18.0, 22.0, "fuzz"),   # fuzzing period
    (28.0, 32.0, "dos"),    # DoS / flooding period
]

# ---------- In-process CAN message structure ----------
class CANMessage:
    def __init__(self, timestamp, arbitration_id, data, is_attack=False, attack_type=None):
        self.timestamp = timestamp
        self.arbitration_id = arbitration_id
        self.data = data  # list of bytes length up to 8
        self.is_attack = is_attack
        self.attack_type = attack_type

# Shared bus (thread-safe queue)
BUS_QUEUE = deque()
BUS_LOCK = threading.Lock()
SIM_START_TIME = None

# Producer threads (normal traffic generators)
def producer_thread(id_, rate):
    """Continuously push normal messages for id_ at ~rate seconds between msgs."""
    while True:
        now = time.time() - SIM_START_TIME
        if now >= SIM_DURATION:
            break
        payload = normal_payload_gen(id_)
        msg = CANMessage(timestamp=time.time(), arbitration_id=id_, data=payload, is_attack=False)
        with BUS_LOCK:
            BUS_QUEUE.append(msg)
        # jittered sleep
        time.sleep(max(0.001, random.gauss(rate, rate*0.2)))

# Attack threads (will push attack messages in their scheduled windows)
def attack_injector():
    """Inject attacks according to ATTACK_SCHEDULE timeline."""
    while True:
        now = time.time() - SIM_START_TIME
        if now >= SIM_DURATION:
            break
        for (s,e,atype) in ATTACK_SCHEDULE:
            if s <= now < e:
                # inside an attack window, push attack messages depending on type
                if atype == "spoof":
                    # pick existing id and send anomalous payloads faster
                    for _ in range(3):
                        data = [255]*8  # obviously abnormal
                        msg = CANMessage(timestamp=time.time(), arbitration_id=random.choice(NORMAL_IDS), data=data, is_attack=True, attack_type='spoof')
                        with BUS_LOCK:
                            BUS_QUEUE.append(msg)
                elif atype == "fuzz":
                    # random ids and random payload
                    for _ in range(4):
                        rid = random.randint(0x100, 0x3FF)
                        data = [random.randint(0,255) for _ in range(8)]
                        msg = CANMessage(timestamp=time.time(), arbitration_id=rid, data=data, is_attack=True, attack_type='fuzz')
                        with BUS_LOCK:
                            BUS_QUEUE.append(msg)
                elif atype == "dos":
                    # flood one id at high rate
                    for _ in range(12):
                        data = [random.randint(0,255) for _ in range(8)]
                        msg = CANMessage(timestamp=time.time(), arbitration_id=0x700, data=data, is_attack=True, attack_type='dos')
                        with BUS_LOCK:
                            BUS_QUEUE.append(msg)
                # small pause so CPU isn't pegged
                time.sleep(0.01)
        time.sleep(0.01)

# ---------- Collector: read messages from BUS_QUEUE into a list for feature extraction ----------
COLLECTED_MESSAGES = []  # list of CANMessage

def bus_collector():
    """Drain the BUS_QUEUE periodically and append to COLLECTED_MESSAGES."""
    while True:
        now = time.time() - SIM_START_TIME
        if now >= SIM_DURATION:
            # drain remaining
            with BUS_LOCK:
                while BUS_QUEUE:
                    COLLECTED_MESSAGES.append(BUS_QUEUE.popleft())
            break
        with BUS_LOCK:
            while BUS_QUEUE:
                COLLECTED_MESSAGES.append(BUS_QUEUE.popleft())
        time.sleep(0.005)

# ---------- Feature extraction ----------
def payload_entropy(data):
    if not data:
        return 0.0
    counts = np.bincount(data, minlength=256)
    probs = counts[counts>0] / counts.sum()
    return float(entropy(probs, base=2))

def extract_window_features(msgs):
    """Given a list of CANMessage objects inside one window, return a dict of features."""
    if not msgs:
        # return zeros baseline
        return {
            'num_msgs': 0, 'num_ids': 0,
            'iat_mean': 0.0, 'iat_std': 0.0,
            'payload_entropy_mean': 0.0, 'payload_entropy_std': 0.0
        }
    ts = [m.timestamp for m in msgs]
    ids = [m.arbitration_id for m in msgs]
    ents = [payload_entropy(m.data) for m in msgs]
    iat = np.diff(ts) if len(ts) > 1 else np.array([0.0])
    feat = {
        'num_msgs': len(msgs),
        'num_ids': len(set(ids)),
        'iat_mean': float(np.mean(iat)),
        'iat_std': float(np.std(iat)),
        'payload_entropy_mean': float(np.mean(ents)),
        'payload_entropy_std': float(np.std(ents)),
    }
    # optionally include frequency of top IDs as features (up to 4)
    top_ids = sorted(list(set(ids)))[:4]
    for i in range(4):
        fid = top_ids[i] if i < len(top_ids) else None
        feat[f'freq_id_{i}'] = sum(1 for m in msgs if m.arbitration_id == fid) if fid is not None else 0
    return feat

# Build feature windows with labels (attack if any message in window has is_attack True)
def build_features_and_labels(all_msgs, start_time, end_time, window_s=WINDOW_S, step=WINDOW_STEP):
    features = []
    labels = []
    t = start_time
    while t + window_s <= end_time + 1e-9:
        window_msgs = [m for m in all_msgs if (m.timestamp - SIM_START_TIME) >= t and (m.timestamp - SIM_START_TIME) < t + window_s]
        feat = extract_window_features(window_msgs)
        # label attack if any message in window has is_attack True
        label = int(any(m.is_attack for m in window_msgs))
        feat['window_start'] = t
        feat['label'] = label
        features.append(feat)
        labels.append(label)
        t += step
    df = pd.DataFrame(features)
    return df

# ---------- Main flow ----------
def main():
    global SIM_START_TIME
    print("Starting CAN bus simulation (in-process) ...")
    SIM_START_TIME = time.time()

    # start producer threads
    producers = []
    for id_ in NORMAL_IDS:
        t = threading.Thread(target=producer_thread, args=(id_, NORMAL_RATE), daemon=True)
        t.start()
        producers.append(t)

    # start attack injector
    atk_thread = threading.Thread(target=attack_injector, daemon=True)
    atk_thread.start()

    # start collector
    col_thread = threading.Thread(target=bus_collector, daemon=True)
    col_thread.start()

    # Wait for simulation end
    while time.time() - SIM_START_TIME < SIM_DURATION:
        elapsed = time.time() - SIM_START_TIME
        if int(elapsed) % 5 == 0:
            # print occasionally
            print(f"Simulating... {elapsed:.1f}s / {SIM_DURATION}s")
        time.sleep(0.5)

    # give threads a moment to finish draining
    time.sleep(0.5)
    print("Simulation finished. Collected messages:", len(COLLECTED_MESSAGES))

    # Build features/labels
    df = build_features_and_labels(COLLECTED_MESSAGES, start_time=0.0, end_time=SIM_DURATION, window_s=WINDOW_S, step=WINDOW_STEP)
    print("Built feature windows:", len(df))
    df.to_csv("features_sim.csv", index=False)
    print("Saved features_sim.csv")

    # Prepare data for ML
    X = df.drop(columns=['window_start', 'label'])
    y = df['label']
    if y.nunique() <= 1:
        print("Warning: no attack windows found in the simulation. Increase SIM_DURATION or check ATTACK_SCHEDULE.")
        return

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model
    dump(clf, "rf_can_ids.joblib")
    print("Saved RF model to rf_can_ids.joblib")

    # ---------- Live detection demo ----------
    print("\nStarting live detection demo (processing windows from simulation data)...")
    # We'll simulate 'real-time' by scanning windows produced earlier and using clf to predict
    for idx, row in df.iterrows():
        feat = row.drop(labels=['window_start','label']).values.reshape(1, -1)
        pred = clf.predict(feat)[0]
        start = row['window_start']
        label = row['label']
        if pred == 1:
            # alert
            when = SIM_START_TIME + start
            print(f"[ALERT] Attack detected in window starting at t={start:.1f}s  (true_label={label})")
        # small pause to simulate streaming
        time.sleep(0.05)

    print("Live demo finished. You can inspect features_sim.csv and rf_can_ids.joblib.")

if __name__ == "__main__":
    main()
