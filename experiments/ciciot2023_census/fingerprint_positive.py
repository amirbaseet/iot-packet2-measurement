#!/usr/bin/env python3
"""
POSITIVE fingerprint test (reviewer round-2, "worth adding"): the §IV-G ablation is a NULL, and a null cannot
establish redundancy over absence. The origin control (predict source DATASET from benign) already shows the
positive-instrument pattern. Here we run its analogue on DEVICE IDENTITY: train a classifier to predict the
VICTIM DEVICE from packet-2 features on BENIGN traffic ONLY. If it scores high, the device-fingerprinting channel
is demonstrated POSITIVELY, and the paper's central claim stops being an interpretation.

Design (per the hygiene note the paper spends 7 pages on): this is DELIBERATELY NOT device-disjoint -- device is
the LABEL, so holding it out is incoherent. We split by flow, stratified by device. Benign only. Report
macro-averaged one-vs-rest AUC + per-device support + the MAJORITY-class baseline (support is uneven, so 1/n is
the wrong baseline). Device label = the IoT-device endpoint MAC (the udp-frag victim set; src-preferred). Packet-2
behavioural features only -- identity builds the label, never a feature. XGBoost 3.3.0 hist, seed 42, n_jobs 1.
"""
import subprocess, os, json, collections
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

EX = "./data/iiot/extract"
BENIGN = EX + "/benign/benign_whole-network3.pcap"
HERE = os.path.dirname(os.path.abspath(__file__))
UDP_CACHE = "./cache/iiot_udpfrag_ablation_cache.json"
FIELDS = ["frame.time_epoch","eth.src","eth.dst","ip.src","ip.dst","ip.proto","ip.len","ip.ttl",
          "tcp.srcport","tcp.dstport","udp.srcport","udp.dstport","tcp.flags","tcp.window_size","frame.len"]
FEATNAMES = ["ip_p","sport","dport","iplen1","iplen2","ttl1","ttl2","win1","win2","flags1","flags2",
             "framelen1","framelen2","iat"]
# 28:87:ba:bd:c6:6c carries 67% of labeled benign flows across every family -- the classic gateway/aggregator
# signature. Excluded so this measures IoT-DEVICE fingerprinting, not gateway detection.
INFRA = {"28:87:ba:bd:c6:6c"}

def tshark(pcap):
    p = subprocess.run(["tshark","-r",pcap,"-T","fields"]+sum([["-e",f] for f in FIELDS],[]), capture_output=True, text=True)
    for line in p.stdout.splitlines(): yield line.split("\t")

def fnum(x):
    try:
        v = float(x); return v if v == v else 0.0
    except: return 0.0

def feats(c1, c, ts):
    t1, t2 = fnum(c1[0]), fnum(ts)
    return [fnum(c[5]),fnum(c[8] or c[10]),fnum(c[9] or c[11]),fnum(c1[6]),fnum(c[6]),fnum(c1[7]),fnum(c[7]),
            fnum(c1[13]),fnum(c[13]),fnum(c1[12]),fnum(c[12]),fnum(c1[14]),fnum(c[14]),(t2-t1) if t2>=t1 else 0.0]

# device label universe = udp-frag victims UNION MITM sensor MACs (broadens the roster), minus infrastructure
devset = {str(v).strip().lower() for v in json.load(open(UDP_CACHE))["vic_mac"].values() if v}
_bs = os.path.join(HERE, "mitm_benign_sparsity.json")
if os.path.exists(_bs):
    devset |= {str(d["mac"]).strip().lower() for d in json.load(open(_bs))["per_device"].values() if d.get("mac")}
devset -= INFRA
print(f"device label set (IoT devices, gateway excluded): {len(devset)}", flush=True)

# assemble benign packet-2 flows, label by the device endpoint (src-preferred)
first = {}; done = set(); X = []; y = []
for c in tshark(BENIGN):
    if len(c) < 15: continue
    ips, ipd, proto = c[3], c[4], c[5]
    sport = c[8] or c[10]; dport = c[9] or c[11]
    key = (ips, ipd, sport, dport, proto)
    if key in done: continue
    if key not in first: first[key] = c; continue
    c1 = first[key]; done.add(key); del first[key]
    src, dst = c1[1].strip().lower(), c1[2].strip().lower()
    label = src if src in devset else (dst if dst in devset else None)
    if label is None: continue
    X.append(feats(c1, c, c[0])); y.append(label)

X = np.array(X, float); y = np.array(y)
support = collections.Counter(y)
# keep devices with enough support for a stratified split
keep = {d for d, n in support.items() if n >= 20}
mask = np.isin(y, list(keep)); X, y = X[mask], y[mask]
classes = sorted(set(y)); idx = {d: i for i, d in enumerate(classes)}
yi = np.array([idx[d] for d in y])
maj = max(collections.Counter(y).values()) / len(y)
print(f"benign flows labeled: {len(y)}  devices (support>=20): {len(classes)}  majority-baseline acc: {maj:.3f}", flush=True)
print("per-device support:", {d: int(support[d]) for d in classes}, flush=True)

Xtr, Xte, ytr, yte = train_test_split(X, yi, test_size=0.3, random_state=42, stratify=yi)
clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="mlogloss",
                    random_state=42, n_jobs=1, tree_method="hist", num_class=len(classes))
clf.fit(Xtr, ytr)
proba = clf.predict_proba(Xte); pred = proba.argmax(1)
acc = accuracy_score(yte, pred)
macro_auc = roc_auc_score(yte, proba, multi_class="ovr", average="macro", labels=list(range(len(classes))))
# per-device one-vs-rest AUC distribution (each device weighted equally -> the dominant class can't inflate it)
per_dev_auc = []
for i in range(len(classes)):
    yb = (yte == i).astype(int)
    if yb.sum() and (yb == 0).any(): per_dev_auc.append(roc_auc_score(yb, proba[:, i]))
per_dev_auc.sort()
auc_lo, auc_med, auc_hi = per_dev_auc[0], per_dev_auc[len(per_dev_auc)//2], per_dev_auc[-1]
# which features carry the device fingerprint
imp = sorted(zip(FEATNAMES, clf.feature_importances_), key=lambda x: -x[1])
top_feats = [(f, round(float(w), 4)) for f, w in imp[:5]]
print(f"\nDEVICE-ID from benign packet-2 features:  accuracy {acc:.3f} (majority {maj:.3f})  macro OVR-AUC {macro_auc:.3f}", flush=True)
print(f"per-device OVR-AUC: min {auc_lo:.3f} / median {auc_med:.3f} / max {auc_hi:.3f}", flush=True)
print(f"top features: {top_feats}", flush=True)
out = {"note": "POSITIVE fingerprint test: predict victim DEVICE from benign packet-2 features. NOT device-disjoint "
               "(device is the label); flow split stratified by device. Identity builds the label, never a feature. "
               "Gateway MAC 28:87:ba:bd:c6:6c excluded as infrastructure.",
       "n_devices": len(classes), "n_benign_flows": int(len(y)), "majority_baseline_acc": round(maj, 4),
       "accuracy": round(acc, 4), "macro_ovr_auc": round(float(macro_auc), 4),
       "per_device_ovr_auc": {"min": round(auc_lo, 4), "median": round(auc_med, 4), "max": round(auc_hi, 4)},
       "top_features": top_feats, "gateway_excluded": sorted(INFRA),
       "per_device_support": {d: int(support[d]) for d in classes}}
json.dump(out, open(os.path.join(HERE, "fingerprint_positive.json"), "w"), indent=2)
print("Wrote fingerprint_positive.json")
print("READ: macro-AUC well above 0.5 and accuracy well above majority => devices are fingerprintable from "
      "benign packet-2 features ALONE -> the channel is demonstrated positively, not inferred from a null ablation.")
