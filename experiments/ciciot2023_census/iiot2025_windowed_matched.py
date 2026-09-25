#!/usr/bin/env python3
"""
Matched windowed comparison (reviewer §3.3). The abstract credits WINDOWED features with fixing the
operating point (a stable low FPR), but the headline 2.8% came from an all-family binary over 38 devices on
the full corpus, while the packet-2 udp-frag number is a single attack over 26 victims — not like-for-like.
This runs the SAME windowed representation that produced the 0.989 per-family number
(combined_dataset.csv / DataSense windows, identity dropped) but on the MATCHED population: udp-frag windows
+ benign windows, victim-disjoint over the SAME udp-frag victim devices as the packet-2 run, same protocol
(hold ~30% victims, 5 seeds, threshold = 1% FPR on seen). Reports windowed AUC/FPR median+range to put beside
packet-2 udp-frag (AUC 0.997, FPR median 22.6%).

HYGIENE TENSION (state in the paper): windowed features are per-device aggregates over a time window, so the
detector must buffer a whole window before deciding -> it is NOT a packet-2 inline decision, and its latency is
the window length, not two packets. The identity-is-never-a-feature rule still holds (device_mac/name dropped),
but per-device windowing means the feature values are computed within one device's traffic.

Feature/label logic mirrors iiot2025_per_family.py exactly. XGBoost 3.3.0 hist, seed 42, n_jobs 1.
"""
import os, json, statistics as stt
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

CSV = os.environ.get("IIOT2025_CSV", "./data/iiot/combined_dataset.csv")
HERE = os.path.dirname(os.path.abspath(__file__))
ABL_CACHE = "./cache/iiot_udpfrag_ablation_cache.json"
DROP = ("label", "timestamp", "network_ips", "network_macs", "network_ports", "network_protocols", "log_data-types")
ID = {"device_name", "device_mac"}

hdr = pd.read_csv(CSV, nrows=0).columns.tolist()
feat_cols = [c for c in hdr if c not in ID and not c.startswith(DROP)]
print(f"feature columns (windowed): {len(feat_cols)}", flush=True)
df = pd.read_csv(CSV, usecols=feat_cols + ["device_mac", "label_full", "label1"], low_memory=False)
for c in feat_cols: df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.fillna(0.0)

lf = df["label_full"].astype(str)
udp = lf.str.contains("udp-frag", case=False, na=False)
benign = df["label1"].astype(str).str.strip() == "benign"
sub = df[udp | benign].copy()
y = udp[udp | benign].astype(int).to_numpy()
X = sub[feat_cols].to_numpy(np.float32)
dev = sub["device_mac"].astype(str).str.strip().str.lower().to_numpy()
print(f"udp-frag windows={int(udp.sum())}  benign windows={int(benign.sum())}", flush=True)

# match to the packet-2 udp-frag victim MACs
vm = set()
if os.path.exists(ABL_CACHE):
    vm = {str(v).strip().lower() for v in json.load(open(ABL_CACHE))["vic_mac"].values() if v}
udp_devs = sorted(set(dev[y == 1]))
matched = sorted(set(udp_devs) & vm) if vm else udp_devs
print(f"udp-frag victim devices (windowed): {len(udp_devs)}; packet-2 victims: {len(vm)}; "
      f"matched: {len(matched)}", flush=True)
# use the matched victims as the disjoint axis (fall back to windowed udp devs if no cache)
axis = matched if matched else udp_devs

def model(): return XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, eval_metric="logloss",
                                  random_state=42, n_jobs=1, tree_method="hist")
def ev(Xtr, ytr, Xte, yte):
    Xf, Xc, yf, yc = train_test_split(Xtr, ytr, test_size=0.2, random_state=7, stratify=ytr)
    clf = model(); clf.fit(Xf, yf); sc = clf.predict_proba(Xc)[:, 1]; st = clf.predict_proba(Xte)[:, 1]
    thr = float(np.quantile(sc[yc == 0], 0.99)) if (yc == 0).any() else 0.5
    pd_ = (st >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(yte, pd_, labels=[0, 1]).ravel()
    return (roc_auc_score(yte, st) if len(set(yte)) > 1 else float("nan"),
            fp / (fp + tn) if (fp + tn) else float("nan"))

# RANDOM
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
ra, rf = ev(Xtr, ytr, Xte, yte)
print(f"\n[RANDOM] windowed AUC {ra:.3f} FPR {rf:.3f}", flush=True)

# VICTIM-DISJOINT over the matched axis (hold ~30%, 5 seeds) — held device's attack AND benign -> test
aucs = []; fprs = []
for seed in [1, 7, 42, 100, 1729]:
    rng = np.random.default_rng(seed); perm = list(axis); rng.shuffle(perm)
    held = set(perm[:max(1, round(len(perm) * 0.3))])
    te = np.isin(dev, list(held))
    if te.sum() == 0 or (~te).sum() == 0 or y[te].sum() == 0 or (y[te] == 0).sum() == 0: continue
    if y[~te].sum() == 0 or (y[~te] == 0).sum() == 0: continue
    a, f = ev(X[~te], y[~te], X[te], y[te]); aucs.append(a); fprs.append(f)
    print(f"[DISJOINT seed={seed:4d}] held {len(held)} victims  AUC {a:.3f}  FPR {f:.3f}", flush=True)

disjoint = {"auc_median": round(stt.median(aucs), 4), "auc_range": [round(min(aucs), 4), round(max(aucs), 4)],
            "fpr_median": round(stt.median(fprs), 4), "fpr_range": [round(min(fprs), 4), round(max(fprs), 4)],
            "n_seeds": len(aucs)} if aucs else None
out = {"note": "MATCHED windowed comparison: same windowed representation (combined_dataset.csv, identity "
               "dropped) as the 0.989 per-family run, restricted to udp-frag windows + benign, victim-disjoint "
               "over the udp-frag victims matched to the packet-2 run. Compare to packet-2 udp-frag AUC 0.997 / "
               "FPR median 0.226. Windowed = per-device windows (buffer a window, not packet-2 inline).",
       "n_feature_cols": len(feat_cols), "udp_windows": int(udp.sum()), "benign_windows": int(benign.sum()),
       "udp_victims_windowed": len(udp_devs), "packet2_victims": len(vm), "matched_victims": len(matched),
       "random": {"auc": round(ra, 4), "fpr": round(rf, 4)}, "disjoint": disjoint,
       "packet2_udpfrag_reference": {"auc": 0.997, "fpr_median": 0.226}}
json.dump(out, open(os.path.join(HERE, "iiot2025_windowed_matched.json"), "w"), indent=2)
print(f"\nMATCHED windowed disjoint: {disjoint}")
print("READ: if windowed disjoint FPR << packet-2 22.6% and stable, windowed fixes the operating point on the "
      "SAME population -> abstract claim matched-validated. If comparable/unstable -> reframe the 2.8% claim.")
