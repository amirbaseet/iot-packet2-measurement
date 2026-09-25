#!/usr/bin/env python3
"""
Feature ablation (reviewer §3.1): isolate the device-identifying channel. Retrain each family
device-disjoint with feature subsets and watch AUC/FPR move. Fields that our origin control and the
IoT-fingerprinting literature flag as device/stack identity: source port (allocation policy), TTL
(OS/hop-count), TCP window (stack). Prediction: for benign-opening families (MITM/DNS/Mirai) there is
no attack signal beyond identity, so removing these barely changes the already-failed disjoint AUC;
for a volumetric flood the attack signal is in packet size/rate, so ranking survives.

CICIoT2023 arm (this file): MITM-ArpSpoofing + DNS_Spoofing (leave-one-victim-out over 3 census
victims) + Mirai floods (2-victim holdout, C(5,2)=10). Reuses the exact extraction/eval of
mitm_dns_disjoint_test.py and mirai_disjoint_test.py; only a feature MASK is looped.
Feature order: ip_p sport dport iplen1 iplen2 ttl1 ttl2 win1 win2 flags1 flags2 pay1 pay2 iat.
XGBoost 3.3.0 hist, seed 42, n_jobs 1.
"""
import zipfile, io, itertools, json, os
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix

ZIP = os.environ.get("CICIOT_ZIP", "./data/CICIOT23/archive.zip")
HERE = os.path.dirname(os.path.abspath(__file__))
FEATS = ["ip_p","sport","dport","iplen1","iplen2","ttl1","ttl2","win1","win2","flags1","flags2","pay1","pay2","iat"]
IDX = {f: i for i, f in enumerate(FEATS)}
# ablation masks: which feature NAMES to DROP
MASKS = {
    "all14":        [],
    "no_stack":     ["ttl1", "ttl2", "win1", "win2"],       # OS/stack fingerprint
    "no_sport":     ["sport"],                                # allocation-policy fingerprint
    "no_identity":  ["ttl1", "ttl2", "win1", "win2", "sport"],
}
BENIGN = ["BenignTraffic.csv", "BenignTraffic1.csv", "BenignTraffic2.csv", "BenignTraffic3.csv"]
LOVO = {  # census 5%-victims (both-direction attribution)
    "MITM-ArpSpoofing.csv": {"08:7c:39:ce:6e:2a", "56:4f:8a:e1:f3:2d", "94:39:e5:5d:27:a6"},
    "DNS_Spoofing.csv":     {"dc:a6:32:dc:27:d5", "56:4f:8a:e1:f3:2d", "24:05:88:30:6f:89"},
}
MIRAI = {"Mirai-greeth_flood.csv", "Mirai-greip_flood.csv", "Mirai-udpplain.csv"}
MIRAI_VIC = {"08:7c:39:ce:6e:2a", "1c:12:b0:9b:0c:ec", "1c:fe:2b:98:16:dd", "9c:8e:cd:1d:ab:9f", "cc:f4:11:9c:d0:00"}

def f(x):
    try: return float(x)
    except: return np.nan

def feats(p1, p2):
    t1, t2 = f(p1[0]), f(p2[0])
    return [f(p1[6]), f(p1[10]), f(p1[11]), f(p1[7]), f(p2[7]), f(p1[8]), f(p2[8]),
            f(p1[15]), f(p2[15]), f(p1[14]), f(p2[14]), f(p1[17]), f(p2[17]),
            (t2 - t1) if (np.isfinite(t1) and np.isfinite(t2)) else np.nan]

def collect(member, is_attack, victims, single_victim_only):
    rows = []; first = {}; seen2 = set(); zf = zipfile.ZipFile(ZIP)
    with zf.open(member) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline=""); text.readline()
        for line in text:
            p = line.rstrip("\n").split(",")
            if len(p) <= 17: continue
            es, ed = p[1], p[2]
            if not is_attack and es not in victims and ed not in victims: continue
            key = (p[4], p[5], p[10], p[11], p[6])
            if key in seen2: continue
            if key not in first: first[key] = p; continue
            row = feats(first[key], p); seen2.add(key); del first[key]
            touched = {es, ed} & victims
            if single_victim_only:
                vt = touched if len(touched) == 1 else set()
            else:
                vt = touched
            rows.append((row, 1 if is_attack else 0, frozenset(vt)))
    return rows

def model(): return XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, eval_metric="logloss",
                                  random_state=42, n_jobs=1, tree_method="hist")
def fit_eval(Xtr, ytr, Xte, yte, cols):
    Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    Xf, Xc, yf, yc = train_test_split(Xtr, ytr, test_size=0.2, random_state=7, stratify=ytr)
    clf = model(); clf.fit(Xf, yf)
    sc = clf.predict_proba(Xc)[:, 1]; st = clf.predict_proba(Xte)[:, 1]
    thr = float(np.quantile(sc[yc == 0], 0.99)) if (yc == 0).any() else 0.5
    pd_ = (st >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(yte, pd_, labels=[0, 1]).ravel()
    tnc, fpc, _, _ = confusion_matrix(yc, (sc >= thr).astype(int), labels=[0, 1]).ravel()
    return (roc_auc_score(yte, st) if len(set(yte)) > 1 else float("nan"),
            fp / (fp + tn) if (fp + tn) else float("nan"),
            fpc / (fpc + tnc) if (fpc + tnc) else None)

def build(members, victims, single_victim_only):
    data = []
    for m in members: data += collect(m, True, victims, single_victim_only)
    for b in BENIGN: data += collect(b, False, victims, single_victim_only)
    X = np.array([d[0] for d in data], float); y = np.array([d[1] for d in data]); vic = [d[2] for d in data]
    return X, y, vic

def cols_for(mask): return [i for i in range(len(FEATS)) if FEATS[i] not in MASKS[mask]]

def random_eval(X, y, cols):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    a, fp, _ = fit_eval(Xtr, ytr, Xte, yte, cols)
    return round(float(a), 4), round(float(fp), 4)

def _disjoint_folds(X, y, vic, victims, cols, pairs):
    aucs = []; fprs = []
    parts = (map(frozenset, itertools.combinations(sorted(victims), 2)) if pairs
             else [frozenset({v}) for v in sorted(victims)])
    for testset in parts:
        side = np.array([bool(v & testset) for v in vic])
        if not side.any() or side.all() or len(set(y[~side])) < 2 or len(set(y[side])) < 2: continue
        a, fp, seen = fit_eval(X[~side], y[~side], X[side], y[side], cols)
        if seen is None or not (0.005 <= seen <= 0.02): continue
        aucs.append(a); fprs.append(fp)
    return aucs, fprs

def run(name, members, victims, single_victim_only, pairs):
    print(f"\n=== {name} ({'2-victim holdout' if pairs else 'LOVO'}) ===", flush=True)
    X, y, vic = build(list(members), victims, single_victim_only=single_victim_only)
    out = {}
    for mask in MASKS:
        cols = cols_for(mask)
        rand_auc, rand_fpr = random_eval(X, y, cols)
        aucs, fprs = _disjoint_folds(X, y, vic, victims, cols, pairs)
        out[mask] = {"random_auc": rand_auc, "random_fpr": rand_fpr,
                     "disjoint_auc_med": round(float(np.median(aucs)), 4) if aucs else None,
                     "disjoint_fpr_med": round(float(np.median(fprs)), 4) if fprs else None, "n_folds": len(aucs)}
        print(f"  {mask:12s} random AUC {rand_auc}  |  disjoint AUC {out[mask]['disjoint_auc_med']} "
              f"FPR {out[mask]['disjoint_fpr_med']} ({out[mask]['n_folds']} folds)", flush=True)
    return out

results = {"note": "device-disjoint AUC/FPR by feature mask; XGB 3.3.0 hist seed42; thr=1%FPR on seen. "
                   "Masks drop TTL/window (stack) and/or source port (allocation) = device-identity fields.",
           "masks": MASKS, "families": {}}
results["families"]["CICIoT2023_MITM"] = run("CICIoT2023 MITM", ["MITM-ArpSpoofing.csv"], LOVO["MITM-ArpSpoofing.csv"], True, False)
results["families"]["CICIoT2023_DNS"] = run("CICIoT2023 DNS", ["DNS_Spoofing.csv"], LOVO["DNS_Spoofing.csv"], True, False)
results["families"]["CICIoT2023_Mirai"] = run("CICIoT2023 Mirai", MIRAI, MIRAI_VIC, False, True)
json.dump(results, open(os.path.join(HERE, "feature_ablation_ciciot2023.json"), "w"), indent=2)
print(f"\nWrote feature_ablation_ciciot2023.json")
print("READ: if AUC barely moves as identity features are removed, the family had no attack signal beyond identity.")
