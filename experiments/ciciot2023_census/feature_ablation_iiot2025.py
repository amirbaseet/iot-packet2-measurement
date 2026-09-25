#!/usr/bin/env python3
"""
Feature ablation, FLOOD arm (reviewer §3.1) — CIC IIoT 2025 udp-frag-flood, packet-2, victim-disjoint.
Companion to feature_ablation.py (CICIoT2023 stealthy families). The question: does the flood's
device-disjoint SUCCESS survive removing identity features? For a flood the answer is expected YES —
but that alone is uninformative, because a udp-frag flood's *attack* signature lives in packet SIZE
(ip.len / frame.len), which is constant per capture and saturates AUC on its own. So we add a `no_size`
mask that removes the attack channel: the discriminating contrast is
    no_identity -> AUC stays high (attack signal intact)  vs  no_size -> AUC drops (attack signal gone).
That pair, not "survives" alone, is the claim.

Feature vector (14, packet-2), IIoT flavour — NOTE it differs from the CICIoT2023 file: cols 11/12 are
frame.len (framelen1/framelen2) here, payload length (pay1/pay2) there. Same INDICES for the masked
identity fields (sport=1, ttl=5,6, win=7,8), so the ablation is index-valid; each JSON carries its own FEATS.

Reuses iiot2025_ddos_pkt2.py extraction exactly. tshark -> flows -> first 2 packets. Extraction is cached
to the cache (16 GB of pcaps -> tshark once); the mask loop then runs from the cache.
XGBoost 3.3.0 hist, seed 42, n_jobs 1.
"""
import subprocess, os, glob, collections, itertools, json, statistics as stt, re
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

EX = "./data/iiot/extract"
FLOODS = sorted(p for p in glob.glob(EX + "/ddos/*udp-frag-flood*.pcap") if "/._" not in p)
BENIGN = EX + "/benign/benign_whole-network3.pcap"
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.environ.get("ABL_CACHE",
    "./cache/iiot_udpfrag_ablation_cache.json")
FIELDS = ["frame.time_epoch","eth.src","eth.dst","ip.src","ip.dst","ip.proto","ip.len","ip.ttl",
          "tcp.srcport","tcp.dstport","udp.srcport","udp.dstport","tcp.flags","tcp.window_size","frame.len"]

FEATS = ["ip_p","sport","dport","iplen1","iplen2","ttl1","ttl2","win1","win2","flags1","flags2",
         "framelen1","framelen2","iat"]
MASKS = {
    "all14":        [],
    "no_stack":     ["ttl1","ttl2","win1","win2"],      # OS/stack fingerprint
    "no_sport":     ["sport"],                            # allocation-policy fingerprint
    "no_identity":  ["ttl1","ttl2","win1","win2","sport"],
    "no_size":      ["iplen1","iplen2","framelen1","framelen2"],  # the udp-frag ATTACK channel (control)
}
IDX = {f: i for i, f in enumerate(FEATS)}

def tshark(pcap):
    p = subprocess.run(["tshark","-r",pcap,"-T","fields"] + sum([["-e",f] for f in FIELDS], []),
                       capture_output=True, text=True)
    for line in p.stdout.splitlines():
        yield line.split("\t")

def fnum(x):
    try:
        v = float(x); return v if v == v else 0.0
    except: return 0.0

def flows_from(pcap):
    first = {}; done = set(); rows = []; dstmac = collections.Counter()
    for c in tshark(pcap):
        if len(c) < 15: continue
        ts,esrc,edst,ips,ipd,proto,iplen,ttl,tsp,tdp,usp,udp_,flags,win,flen = c[:15]
        sport = tsp or usp; dport = tdp or udp_
        key = (ips,ipd,sport,dport,proto)
        if key in done: continue
        dstmac[edst] += 1
        if key not in first: first[key] = c; continue
        c1 = first[key]; t1,t2 = fnum(c1[0]), fnum(ts)
        feat = [fnum(proto),fnum(sport),fnum(dport),fnum(c1[6]),fnum(iplen),fnum(c1[7]),fnum(ttl),
                fnum(c1[13]),fnum(win),fnum(c1[12]),fnum(flags),fnum(c1[14]),fnum(flen),(t2-t1) if t2>=t1 else 0.0]
        rows.append(feat); done.add(key); del first[key]
    vic = dstmac.most_common(1)[0][0] if dstmac else None
    return rows, vic

def extract():
    print(f"flood pcaps: {len(FLOODS)}", flush=True)
    X = []; dev = []; vic_mac = {}
    for p in FLOODS:
        name = re.sub(r".*udp-frag-flood_|\.pcap$", "", os.path.basename(p))
        rows, vmac = flows_from(p); vic_mac[name] = vmac
        for r in rows: X.append(r); dev.append(name)
        print(f"  {name:22s} flows={len(rows):>6} victim_mac={vmac}", flush=True)
    macset = {m for m in vic_mac.values() if m}
    ben_rows = []; ben_touch = []; first = {}; done = set()
    for c in tshark(BENIGN):
        if len(c) < 15: continue
        ts,esrc,edst,ips,ipd,proto,iplen,ttl,tsp,tdp,usp,udp_,flags,win,flen = c[:15]
        sport = tsp or usp; dport = tdp or udp_; key = (ips,ipd,sport,dport,proto)
        if key in done: continue
        if key not in first: first[key] = (c, {esrc,edst}); continue
        c1, m1 = first[key]; t1,t2 = fnum(c1[0]), fnum(ts)
        feat = [fnum(proto),fnum(sport),fnum(dport),fnum(c1[6]),fnum(iplen),fnum(c1[7]),fnum(ttl),
                fnum(c1[13]),fnum(win),fnum(c1[12]),fnum(flags),fnum(c1[14]),fnum(flen),(t2-t1) if t2>=t1 else 0.0]
        ben_rows.append(feat); ben_touch.append((m1 | {esrc,edst}) & macset); done.add(key); del first[key]
    print(f"benign flows={len(ben_rows)}  victim devices={len(vic_mac)}", flush=True)
    data = {"X": X, "dev": dev, "ben_rows": ben_rows,
            "ben_touch": [sorted(s) for s in ben_touch], "vic_mac": vic_mac}
    json.dump(data, open(CACHE, "w"))          # self-produced local cache; JSON, no pickle
    print(f"cached extraction -> {CACHE}", flush=True)
    return data

def load():
    if os.path.exists(CACHE):
        print(f"loading cached extraction <- {CACHE}", flush=True)
        return json.load(open(CACHE))
    return extract()

def model(): return XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, eval_metric="logloss",
                                  random_state=42, n_jobs=1, tree_method="hist")

def ev(Xtr, ytr, Xte, yte, cols):
    Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    Xf, Xc, yf, yc = train_test_split(Xtr, ytr, test_size=0.2, random_state=7, stratify=ytr)
    clf = model(); clf.fit(Xf, yf)
    sc = clf.predict_proba(Xc)[:, 1]; st = clf.predict_proba(Xte)[:, 1]
    thr = float(np.quantile(sc[yc == 0], 0.99)) if (yc == 0).any() else 0.5
    pd_ = (st >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(yte, pd_, labels=[0, 1]).ravel()
    return (roc_auc_score(yte, st) if len(set(yte)) > 1 else float("nan"),
            fp / (fp + tn) if (fp + tn) else float("nan"))

def cols_for(mask): return [i for i in range(len(FEATS)) if FEATS[i] not in MASKS[mask]]

def run():
    d = load()
    Xa = np.array(d["X"], float); Xb = np.array(d["ben_rows"], float)
    dev_arr = np.array(d["dev"]); vic_mac = d["vic_mac"]
    ben_touch = [set(s) for s in d["ben_touch"]]
    devices = sorted(vic_mac)
    ya = np.ones(len(Xa))
    print(f"\nattack flows={len(Xa)} benign flows={len(Xb)} victim devices={len(devices)}", flush=True)
    out = {}
    for mask in MASKS:
        cols = cols_for(mask)
        # RANDOM
        Xall = np.vstack([Xa, Xb]); yall = np.concatenate([ya, np.zeros(len(Xb))])
        Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.3, random_state=42, stratify=yall)
        ra, rf = ev(Xtr, ytr, Xte, yte, cols)
        # VICTIM-DISJOINT (hold 30% devices; their flood + their benign -> test), 5 seeds
        aucs = []; fprs = []
        for seed in [1, 7, 42, 100, 1729]:
            rng = np.random.default_rng(seed); perm = list(devices); rng.shuffle(perm)
            held = set(perm[:max(1, round(len(perm) * 0.3))])
            heldmac = {vic_mac[dd] for dd in held if vic_mac[dd]}
            ta = np.isin(dev_arr, list(held)); tb = np.array([bool(t & heldmac) for t in ben_touch])
            if ta.sum() == 0 or (~ta).sum() == 0 or tb.sum() == 0 or (~tb).sum() == 0: continue
            Xtr = np.vstack([Xa[~ta], Xb[~tb]]); ytr = np.concatenate([np.ones((~ta).sum()), np.zeros((~tb).sum())])
            Xte = np.vstack([Xa[ta], Xb[tb]]);  yte = np.concatenate([np.ones(ta.sum()), np.zeros(tb.sum())])
            a, fp = ev(Xtr, ytr, Xte, yte, cols); aucs.append(a); fprs.append(fp)
        out[mask] = {"random_auc": round(ra, 4), "random_fpr": round(rf, 4),
                     "disjoint_auc_med": round(stt.median(aucs), 4) if aucs else None,
                     "disjoint_fpr_med": round(stt.median(fprs), 4) if fprs else None,
                     "disjoint_auc_range": [round(min(aucs), 4), round(max(aucs), 4)] if aucs else None,
                     "n_seeds": len(aucs)}
        print(f"  {mask:12s} random AUC {out[mask]['random_auc']}  |  disjoint AUC "
              f"{out[mask]['disjoint_auc_med']} FPR {out[mask]['disjoint_fpr_med']} ({len(aucs)} seeds)", flush=True)
    results = {"note": "CIC IIoT 2025 udp-frag-flood packet-2 victim-disjoint AUC/FPR by feature mask; "
                       "XGB 3.3.0 hist seed42; thr=1%FPR on seen. no_size drops the udp-frag ATTACK channel "
                       "(ip.len/frame.len) as a control: no_identity high + no_size low = attack signal, not identity.",
               "FEATS": FEATS, "masks": MASKS, "family": "CICIIoT2025_udpfrag", "results": out}
    json.dump(results, open(os.path.join(HERE, "feature_ablation_iiot2025.json"), "w"), indent=2)
    print("\nWrote feature_ablation_iiot2025.json")
    print("READ: no_identity ~unchanged + no_size collapse => the flood rides its ATTACK channel, not identity.")

if __name__ == "__main__":
    run()
