#!/usr/bin/env python3
"""
WS3 (exploratory MEASUREMENT, no detection claim) — per-device anomaly baseline.
Flip the architecture: instead of one global detector that must generalise across devices (which
fingerprints, #52), give EACH victim device its own 'normal' baseline (IsolationForest on that device's
OWN benign) and ask whether Mirai flood traffic to that device flags as anomalous WITHOUT any cross-device
training. Threshold set to 1% FPR on the device's HELD-OUT benign (so the false-alarm rate is honest).

The question (per the fit-assessment): does a per-device baseline ESCAPE the cross-device fingerprint, or
TRADE it for the benign-drift false-alarm wall the spec predicts? We report both the flood detection rate
AND the held-out-benign FPR per device. NO claim that this 'detects attacks' — a measurement of the limit.
Reads LOCAL zip. Deterministic.
"""
import zipfile, io, json, os, collections
import numpy as np
from sklearn.ensemble import IsolationForest

ZIP=os.environ.get("CICIOT_ZIP",
    "./data/CICIOT23/archive.zip")
HERE=os.path.dirname(os.path.abspath(__file__))
MIRAI=["Mirai-greeth_flood.csv","Mirai-greip_flood.csv","Mirai-udpplain.csv"]
BENIGN=["BenignTraffic.csv","BenignTraffic1.csv","BenignTraffic2.csv","BenignTraffic3.csv"]
VICTIMS={"08:7c:39:ce:6e:2a","1c:12:b0:9b:0c:ec","1c:fe:2b:98:16:dd","9c:8e:cd:1d:ab:9f","cc:f4:11:9c:d0:00"}
MIN_BENIGN=400
def f(x):
    try:
        v=float(x); return v if v==v else 0.0
    except: return 0.0
def feats(r1,p2):
    t1,t2=f(r1[0]),f(p2[0])
    return [f(r1[6]),f(r1[10]),f(r1[11]),f(r1[7]),f(p2[7]),f(r1[8]),f(p2[8]),f(r1[15]),f(p2[15]),
            f(r1[14]),f(p2[14]),f(r1[17]),f(p2[17]),(t2-t1) if t2>=t1 else 0.0]

def collect(member,attack):
    """return per-device (dst for attack; the victim endpoint for benign) list of packet-2 feature rows."""
    by_dev=collections.defaultdict(list); first={}; seen=set(); zf=zipfile.ZipFile(ZIP)
    with zf.open(member) as fh:
        text=io.TextIOWrapper(fh,encoding="utf-8",errors="replace",newline=""); text.readline()
        for line in text:
            p=line.rstrip("\n").split(",")
            if len(p)<=17: continue
            es,ed=p[1],p[2]
            dev=(ed if ed in VICTIMS else None) if attack else next(iter({es,ed}&VICTIMS),None)
            if dev is None: continue
            key=(p[4],p[5],p[10],p[11],p[6])
            if key in seen: continue
            if key not in first: first[key]=(p,dev); continue
            r1,d=first[key]; by_dev[d].append(feats(r1,p2=p)); seen.add(key); del first[key]
    return by_dev

print("Extracting per-device flows...",flush=True)
atk=collections.defaultdict(list); ben=collections.defaultdict(list)
for m in MIRAI:
    for d,rows in collect(m,True).items(): atk[d]+=rows
for m in BENIGN:
    for d,rows in collect(m,False).items(): ben[d]+=rows

rng=np.random.default_rng(42)
results={}
print(f"\n{'device':22s} {'benign':>7s} {'flood':>7s} {'FPR@thr':>8s} {'flood_detect':>12s}")
for dev in sorted(VICTIMS):
    B=np.array(ben.get(dev,[]),float); A=np.array(atk.get(dev,[]),float)
    if len(B)<MIN_BENIGN or len(A)<50:
        results[dev]={"skip":f"benign={len(B)} flood={len(A)}"};
        print(f"{dev:22s} {len(B):7d} {len(A):7d}   SKIP (insufficient)"); continue
    idx=rng.permutation(len(B)); cut=int(0.7*len(B)); tr,te=B[idx[:cut]],B[idx[cut:]]
    iso=IsolationForest(n_estimators=200,random_state=42,contamination='auto').fit(tr)
    s_te=-iso.score_samples(te); s_at=-iso.score_samples(A)      # higher = more anomalous
    thr=float(np.quantile(s_te,0.99))                            # 1% FPR on held-out benign of THIS device
    fpr=float(np.mean(s_te>=thr)); det=float(np.mean(s_at>=thr))
    results[dev]={"benign":len(B),"flood":len(A),"fpr_heldout_benign":round(fpr,4),"flood_detect_rate":round(det,4)}
    print(f"{dev:22s} {len(B):7d} {len(A):7d} {fpr:8.3f} {det:12.3f}")

ok=[r for r in results.values() if "flood_detect_rate" in r]
if ok:
    import statistics as stt
    md=round(stt.median(r["flood_detect_rate"] for r in ok),3); mf=round(stt.median(r["fpr_heldout_benign"] for r in ok),3)
    print(f"\nPer-device baseline (median over {len(ok)} devices): flood detection {md} at held-out-benign FPR {mf}")
    print("READ: high detection at ~1% FPR => per-device baseline ESCAPES the cross-device fingerprint (same-session benign).")
    print("      CAVEAT (spec's no-novelty wall): held-out benign here is SAME-SESSION; real benign DRIFT (new")
    print("      firmware/app) is untested and is where anomaly baselines historically blow up the FPR.")
json.dump({"note":"per-device IsolationForest; thr=1%FPR on device's held-out benign; NO detection claim; same-session benign only",
           "min_benign":MIN_BENIGN,"per_device":results},open(os.path.join(HERE,"baseline_spike.json"),"w"),indent=2)
print("Wrote baseline_spike.json")
