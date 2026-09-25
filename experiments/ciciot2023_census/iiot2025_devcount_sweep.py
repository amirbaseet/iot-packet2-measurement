#!/usr/bin/env python3
"""
Controlled device-count sweep, WITHIN CIC IIoT 2025 only (removes the dataset confound that the
cross-dataset "0.66 at 5 victims -> 0.997 at 26" comparison carried: dataset, capture year, and
attack subtype all changed together there).

Design: udp-fragmentation flood, one pcap per victim device. Hold out a FIXED set of test victims
(their flood AND their benign -> test, never in training). Then vary only the number of TRAINING
victims (their flood + benign), sub-sampled at several pool sizes, and measure device-disjoint AUC
and FPR on the SAME held-out victims. Everything else is held constant: features (packet-2 only,
no MAC/IP), model (XGB 150/6/0.1, seed 42, n_jobs 1), threshold = 1% FPR on seen. Multiple
sub-sample seeds per pool size -> report median and spread.

READ: if AUC climbs with training-victim count on the fixed test set, contribution (ii) is a real
controlled effect. If it is flat, device count is necessary but not an isolable ranking driver here,
and the paper must say so.
"""
import subprocess, os, glob, collections, json, statistics as stt, re
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

EX = "./data/iiot/extract"
FLOODS = sorted(p for p in glob.glob(EX + "/ddos/*udp-frag-flood*.pcap") if "/._" not in p)
BENIGN = EX + "/benign/benign_whole-network3.pcap"
HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["frame.time_epoch", "eth.src", "eth.dst", "ip.src", "ip.dst", "ip.proto", "ip.len", "ip.ttl",
          "tcp.srcport", "tcp.dstport", "udp.srcport", "udp.dstport", "tcp.flags", "tcp.window_size", "frame.len"]
HELD_TEST = 5            # fixed held-out test victims
POOL_SIZES = [5, 8, 12, 16, 20]
SUBSAMPLE_SEEDS = [1, 7, 42, 100, 1729]

def tshark(pcap):
    p = subprocess.run(["tshark", "-r", pcap, "-T", "fields"] + sum([["-e", f] for f in FIELDS], []),
                       capture_output=True, text=True)
    for line in p.stdout.splitlines():
        yield line.split("\t")

def fnum(x):
    try:
        v = float(x); return v if v == v else 0.0
    except: return 0.0

def feats(c1, c, ts):
    proto, iplen, ttl, flags, win, flen = c[5], c[6], c[7], c[12], c[13], c[14]
    sport = c[8] or c[10]; dport = c[9] or c[11]
    t1, t2 = fnum(c1[0]), fnum(ts)
    return [fnum(proto), fnum(sport), fnum(dport), fnum(c1[6]), fnum(iplen), fnum(c1[7]), fnum(ttl),
            fnum(c1[13]), fnum(win), fnum(c1[12]), fnum(flags), fnum(c1[14]), fnum(flen), (t2 - t1) if t2 >= t1 else 0.0]

def flows_from(pcap):
    first = {}; done = set(); rows = []; dstmac = collections.Counter()
    for c in tshark(pcap):
        if len(c) < 15: continue
        ips, ipd, proto = c[3], c[4], c[5]
        sport = c[8] or c[10]; dport = c[9] or c[11]
        key = (ips, ipd, sport, dport, proto)
        if key in done: continue
        dstmac[c[2]] += 1
        if key not in first: first[key] = c; continue
        rows.append(feats(first[key], c, c[0])); done.add(key); del first[key]
    vic = dstmac.most_common(1)[0][0] if dstmac else None
    return rows, vic

# ---- extract once ----
print(f"flood pcaps: {len(FLOODS)}", flush=True)
per_dev = {}; vic_mac = {}
for p in FLOODS:
    name = re.sub(r".*udp-frag-flood_|\.pcap$", "", os.path.basename(p))
    rows, vmac = flows_from(p); vic_mac[name] = vmac; per_dev[name] = rows
    print(f"  {name:24s} flows={len(rows):>6} mac={vmac}", flush=True)

macset = {m for m in vic_mac.values() if m}
first = {}; done = set(); ben_rows = []; ben_touch = []
for c in tshark(BENIGN):
    if len(c) < 15: continue
    ips, ipd, proto = c[3], c[4], c[5]
    sport = c[8] or c[10]; dport = c[9] or c[11]
    key = (ips, ipd, sport, dport, proto)
    if key in done: continue
    if key not in first: first[key] = (c, {c[1], c[2]}); continue
    c1, m1 = first[key]
    ben_rows.append(feats(c1, c, c[0])); ben_touch.append((m1 | {c[1], c[2]}) & macset); done.add(key); del first[key]
Xb = np.array(ben_rows, float)
print(f"benign flows={len(ben_rows)}  victim devices={len(vic_mac)}", flush=True)

devices = sorted(vic_mac)
def model(): return XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, eval_metric="logloss",
                                  random_state=42, n_jobs=1, tree_method="hist")
def ev(Xtr, ytr, Xte, yte):
    Xf, Xc, yf, yc = train_test_split(Xtr, ytr, test_size=0.2, random_state=7, stratify=ytr)
    clf = model(); clf.fit(Xf, yf); sc = clf.predict_proba(Xc)[:, 1]; st = clf.predict_proba(Xte)[:, 1]
    thr = float(np.quantile(sc[yc == 0], 0.99)) if (yc == 0).any() else 0.5
    seen_fpr = float((sc[yc == 0] >= thr).mean()) if (yc == 0).any() else None
    pd_ = (st >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(yte, pd_, labels=[0, 1]).ravel()
    return (roc_auc_score(yte, st) if len(set(yte)) > 1 else float('nan'),
            fp / (fp + tn) if (fp + tn) else float('nan'), seen_fpr)

def rows_for(devset):
    Xf = np.array([r for d in devset for r in per_dev[d]], float)
    macs = {vic_mac[d] for d in devset if vic_mac[d]}
    bmask = np.array([bool(t & macs) for t in ben_touch])
    return Xf, bmask, macs

# ---- fixed held-out TEST set (seed 2024); ensure it has benign coverage ----
rng0 = np.random.default_rng(2024); perm = list(devices); rng0.shuffle(perm)
held = perm[:HELD_TEST]
Xte_a, te_bmask, held_macs = rows_for(held)
pool_all = [d for d in devices if d not in held]
max_pool = len(pool_all)
print(f"\nHeld-out TEST victims (fixed): {held}\n  test flood flows={len(Xte_a)} test benign flows={int(te_bmask.sum())}", flush=True)
print(f"Training pool available: {max_pool} victims", flush=True)
Xte = np.vstack([Xte_a, Xb[te_bmask]]); yte = np.concatenate([np.ones(len(Xte_a)), np.zeros(int(te_bmask.sum()))])

results = []
for k in [s for s in POOL_SIZES if s <= max_pool] + ([max_pool] if max_pool not in POOL_SIZES else []):
    aucs = []; fprs = []; seens = []
    for seed in SUBSAMPLE_SEEDS:
        rng = np.random.default_rng(seed); pool = list(pool_all); rng.shuffle(pool)
        train_devs = pool[:k]
        Xtr_a, tr_bmask, _ = rows_for(train_devs)
        tr_bmask = tr_bmask & ~te_bmask                      # never train on benign that touches a held-out victim
        if len(Xtr_a) == 0 or tr_bmask.sum() == 0: continue
        Xtr = np.vstack([Xtr_a, Xb[tr_bmask]]); ytr = np.concatenate([np.ones(len(Xtr_a)), np.zeros(int(tr_bmask.sum()))])
        a, f, sfpr = ev(Xtr, ytr, Xte, yte); aucs.append(a); fprs.append(f); seens.append(sfpr)
    row = {"train_victims": k, "n_seeds": len(aucs),
           "auc_median": round(stt.median(aucs), 4), "auc_min": round(min(aucs), 4), "auc_max": round(max(aucs), 4),
           "fpr_median": round(stt.median(fprs), 4),
           "seen_fpr_median": round(stt.median([s for s in seens if s is not None]), 4)}
    results.append(row)
    print(f"[k={k:2d} train victims] AUC median {row['auc_median']} ({row['auc_min']}-{row['auc_max']})  "
          f"FPR median {row['fpr_median']}  seen-FPR {row['seen_fpr_median']}", flush=True)

out = {"note": "WITHIN CIC IIoT 2025 udp-frag; fixed held-out test victims; vary training-victim count; "
               "packet-2 features only; XGB 150/6/0.1 seed42 n_jobs1; threshold=1%FPR on seen",
       "held_test_victims": held, "n_held_test": HELD_TEST, "training_pool_available": max_pool,
       "test_flood_flows": len(Xte_a), "test_benign_flows": int(te_bmask.sum()),
       "sweep": results}
json.dump(out, open(os.path.join(HERE, "iiot2025_devcount_sweep.json"), "w"), indent=2)
aucs_only = [r["auc_median"] for r in results]
print(f"\nSUMMARY: AUC medians by train-victim count {[(r['train_victims'], r['auc_median']) for r in results]}")
print(f"VERDICT: {'CLIMBS' if aucs_only[-1] - aucs_only[0] >= 0.05 else 'FLAT'} "
      f"(delta {round(aucs_only[-1]-aucs_only[0],3)} from {results[0]['train_victims']} to {results[-1]['train_victims']} victims)")
print("Wrote iiot2025_devcount_sweep.json")
