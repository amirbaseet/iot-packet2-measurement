#!/usr/bin/env python3
"""
POSITIVE fingerprint test, SECOND dataset (CICIoT2023) -- replicates fingerprint_positive.py so the
device-fingerprinting channel is shown on two datasets, not one. Predict the source DEVICE from BENIGN packet-2
features alone; if it scores high, devices are identifiable from ordinary traffic and the paper's central claim
("the model learned the device") rests on a positive measurement.

Design (identical hygiene): NOT device-disjoint -- device is the LABEL, so holding it out is incoherent; split by
flow, stratified by device. Benign only. Report macro one-vs-rest AUC + per-device AUC spread + MAJORITY baseline
+ top features. Label = source eth.src device MAC; the two dominant gateway/router MACs are excluded as
infrastructure. Packet-2 behavioural features only -- identity builds the label, never a feature.

CICIoT2023 CSV schema (packet-level): eth.src=col1, eth.dst=col2; flow key (p4,p5,p10,p11,p6). 14-vector reuses
feature_ablation.py exactly (payload_len at p17, TCP window at p15 -- differs from the CIC IIoT frame-len vector).
XGBoost 3.3.0 hist, seed 42, n_jobs 1.
"""
import zipfile, io, os, json, collections
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

ZIP = os.environ.get("CICIOT_ZIP", "./data/CICIOT23/archive.zip")
HERE = os.path.dirname(os.path.abspath(__file__))
BENIGN = ["BenignTraffic.csv", "BenignTraffic1.csv", "BenignTraffic2.csv", "BenignTraffic3.csv"]
FEATNAMES = ["ip_p","sport","dport","iplen1","iplen2","ttl1","ttl2","win1","win2","flags1","flags2",
             "pay1","pay2","iat"]
# dominant benign endpoints (census B_dst_concentration): 3c:.. 46.9%, 40:.. 22.4% -> gateway/router, excluded
INFRA = {"3c:18:a0:41:c3:a0", "40:5d:82:35:14:c8"}
PERDEV_CAP = 3000     # bound memory: at most this many benign flows per device

def f(x):
    try:
        v = float(x); return v if v == v else 0.0
    except: return 0.0

def feats(p1, p2):
    t1, t2 = f(p1[0]), f(p2[0])
    return [f(p1[6]),f(p1[10]),f(p1[11]),f(p1[7]),f(p2[7]),f(p1[8]),f(p2[8]),f(p1[15]),f(p2[15]),
            f(p1[14]),f(p2[14]),f(p1[17]),f(p2[17]),(t2-t1) if t2>=t1 else 0.0]

def is_device(mac):
    m = mac.strip().lower()
    return bool(m) and m not in INFRA and not m.startswith("ff:") and not m.startswith("01:00:5e") and m != ""

# assemble benign packet-2 flows, label by source device (eth.src of the first packet)
first = {}; seen = set(); zf = zipfile.ZipFile(ZIP)
per_dev = collections.defaultdict(list)     # mac -> list of feature rows (capped)
for member in BENIGN:
    try: fh = zf.open(member)
    except KeyError: continue
    text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline=""); text.readline()
    for line in text:
        p = line.rstrip("\n").split(",")
        if len(p) <= 17: continue
        key = (p[4], p[5], p[10], p[11], p[6])
        if key in seen: continue
        if key not in first: first[key] = p; continue
        p1 = first[key]; seen.add(key); del first[key]
        src = p1[1].strip().lower()
        if not is_device(src): continue
        if len(per_dev[src]) < PERDEV_CAP: per_dev[src].append(feats(p1, p))

# keep devices with >=20 flows
X = []; y = []
for mac, rows in per_dev.items():
    if len(rows) >= 20:
        X.extend(rows); y.extend([mac] * len(rows))
X = np.array(X, float); y = np.array(y)
support = collections.Counter(y)
classes = sorted(set(y)); idx = {d: i for i, d in enumerate(classes)}
yi = np.array([idx[d] for d in y])
maj = max(support.values()) / len(y)
print(f"benign flows labeled: {len(y)}  devices (support>=20): {len(classes)}  majority baseline: {maj:.3f}", flush=True)

Xtr, Xte, ytr, yte = train_test_split(X, yi, test_size=0.3, random_state=42, stratify=yi)
clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="mlogloss",
                    random_state=42, n_jobs=1, tree_method="hist", num_class=len(classes))
clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte); pred = proba.argmax(1)
acc = accuracy_score(yte, pred)
macro_auc = roc_auc_score(yte, proba, multi_class="ovr", average="macro", labels=list(range(len(classes))))
per_dev_auc = []
for i in range(len(classes)):
    yb = (yte == i).astype(int)
    if yb.sum() and (yb == 0).any(): per_dev_auc.append(roc_auc_score(yb, proba[:, i]))
per_dev_auc.sort()
auc_lo, auc_med, auc_hi = per_dev_auc[0], per_dev_auc[len(per_dev_auc)//2], per_dev_auc[-1]
imp = sorted(zip(FEATNAMES, clf.feature_importances_), key=lambda x: -x[1])
top_feats = [(ff, round(float(w), 4)) for ff, w in imp[:5]]
print(f"\nDEVICE-ID (CICIoT2023) from benign packet-2 features: accuracy {acc:.3f} (majority {maj:.3f})  macro OVR-AUC {macro_auc:.3f}", flush=True)
print(f"per-device OVR-AUC: min {auc_lo:.3f} / median {auc_med:.3f} / max {auc_hi:.3f}  over {len(classes)} devices", flush=True)
print(f"top features: {top_feats}", flush=True)
out = {"note": "POSITIVE fingerprint test on CICIoT2023: predict source DEVICE from benign packet-2 features. "
               "NOT device-disjoint (device is the label); flow split stratified by device; identity builds the "
               "label, never a feature. Two dominant gateway/router MACs excluded as infrastructure.",
       "dataset": "CICIoT2023", "FEATS": FEATNAMES, "n_devices": len(classes), "n_benign_flows": int(len(y)),
       "majority_baseline_acc": round(maj, 4), "accuracy": round(acc, 4), "macro_ovr_auc": round(float(macro_auc), 4),
       "per_device_ovr_auc": {"min": round(auc_lo, 4), "median": round(auc_med, 4), "max": round(auc_hi, 4)},
       "top_features": top_feats, "gateway_excluded": sorted(INFRA), "perdev_cap": PERDEV_CAP}
json.dump(out, open(os.path.join(HERE, "fingerprint_positive_ciciot2023.json"), "w"), indent=2)
print("Wrote fingerprint_positive_ciciot2023.json")
print("READ: macro-AUC well above 0.5 and accuracy well above majority on a SECOND dataset => the fingerprinting "
      "channel is not a one-dataset artifact.")
