#!/usr/bin/env python3
"""
Pooled-Mirai: combine CICIoT2023 + CIC IIoT 2025 Mirai victims into ONE device-disjoint test.
Question: does a bigger victim pool (5 + 8 = 13) rescue Mirai, or does it stay victim-bound even pooled?

Common packet-2 schema BOTH datasets can produce identically (payload/frame.len dropped — not comparable):
  proto, sport, dport, iplen1, iplen2, ttl1, ttl2, win1, win2, flags1, flags2, iat   (12 features)
No MAC/IP as features; device identity only builds the split; device IDs are dataset-prefixed so pools don't
collide. TRUE victim holdout mixing BOTH datasets on both sides. Threshold = 1% FPR on seen. Deterministic.

ORIGIN CONTROL (mandatory): train a classifier to predict dataset-of-origin from BENIGN flows only. If it
can (AUC high), the two labs are trivially distinguishable -> any pooled 'success' is the lab-fingerprint
cheat, not attack detection. We want this AUC LOW.
"""
import zipfile, io, subprocess, glob, os, collections, json, statistics as stt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

C23_ZIP="./data/CICIOT23/archive.zip"
I25_MIRAI=sorted(p for p in glob.glob("./data/iiot/extract/malware/*mirai*.pcap") if "/._" not in p)
I25_BENIGN="./data/iiot/extract/benign/benign_whole-network3.pcap"
HERE=os.path.dirname(os.path.abspath(__file__))
C23_MIRAI_CSVS=["Mirai-greeth_flood.csv","Mirai-greip_flood.csv","Mirai-udpplain.csv"]
C23_BENIGN_CSVS=["BenignTraffic.csv","BenignTraffic1.csv","BenignTraffic2.csv","BenignTraffic3.csv"]
C23_VICTIMS={"08:7c:39:ce:6e:2a","1c:12:b0:9b:0c:ec","1c:fe:2b:98:16:dd","9c:8e:cd:1d:ab:9f","cc:f4:11:9c:d0:00"}

def fnum(x):
    try:
        v=float(x); return v if v==v else 0.0
    except: return 0.0
def flagnum(x):          # CSV "16.0" or tshark "0x0010" -> int
    x=(x or "").strip()
    if not x: return 0.0
    try: return float(int(x,16)) if x.lower().startswith("0x") else float(x)
    except: return 0.0
def is_real(m):
    if not m or ":" not in m or m.lower()=="ff:ff:ff:ff:ff:ff": return False
    try: return not (int(m.split(":")[0],16)&1)
    except ValueError: return False

# feature vector from two packets' raw dicts {ts,proto,sport,dport,iplen,ttl,win,flags}
def feat(p1,p2):
    return [p1["proto"],p1["sport"],p1["dport"],p1["iplen"],p2["iplen"],p1["ttl"],p2["ttl"],
            p1["win"],p2["win"],p1["flags"],p2["flags"],(p2["ts"]-p1["ts"]) if p2["ts"]>=p1["ts"] else 0.0]

# ---- CICIoT2023 side: read CSVs from the zip ----
# cols: 0 ts,1 eth_src,2 eth_dst,4 ip_src,5 ip_dst,6 ip_p,7 ip_len,8 ip_ttl,10 sport,11 dport,14 flags,15 win
def c23_rows(members, attack):
    zf=zipfile.ZipFile(C23_ZIP); out=[]
    for m in members:
        first={}; done=set()
        with zf.open(m) as fh:
            t=io.TextIOWrapper(fh,encoding="utf-8",errors="replace",newline=""); t.readline()
            for line in t:
                c=line.rstrip("\n").split(",")
                if len(c)<16: continue
                es,ed=c[1],c[2]
                dev = ed if (attack and ed in C23_VICTIMS) else (next(iter(({es,ed}&C23_VICTIMS)),None) if not attack else None)
                if dev is None: continue
                key=(c[4],c[5],c[10],c[11],c[6])
                if key in done: continue
                pk={"ts":fnum(c[0]),"proto":fnum(c[6]),"sport":fnum(c[10]),"dport":fnum(c[11]),
                    "iplen":fnum(c[7]),"ttl":fnum(c[8]),"win":fnum(c[15]),"flags":flagnum(c[14])}
                if key not in first: first[key]=(pk,dev); continue
                p1,d=first[key]; out.append((feat(p1,pk),"c23:"+d)); done.add(key); del first[key]
    return out

# ---- CIC IIoT 2025 side: tshark the pcaps ----
FIELDS=["frame.time_epoch","eth.src","eth.dst","ip.src","ip.dst","ip.proto","ip.len","ip.ttl",
        "tcp.srcport","tcp.dstport","udp.srcport","udp.dstport","tcp.flags","tcp.window_size"]
def tshark(pcap):
    r=subprocess.run(["tshark","-r",pcap,"-T","fields"]+sum([["-e",f] for f in FIELDS],[]),capture_output=True,text=True)
    for line in r.stdout.splitlines(): yield line.split("\t")
def i25_pcap_rows(pcap, attack, victim_hint=None):
    first={}; done=set(); out=[]; dstc=collections.Counter()
    recs=list(tshark(pcap))
    for c in recs:
        if len(c)<14: continue
        dstc[c[2]]+=1
    vic = victim_hint or (dstc.most_common(1)[0][0] if dstc else None)
    for c in recs:
        if len(c)<14: continue
        es,ed=c[1],c[2]
        if attack:
            if ed!=vic: continue
            dev=vic
        else:
            dev=next(iter(({es,ed}&I25_BENIGN_VICS)),None)
            if dev is None: continue
        sp=c[8] or c[10]; dp=c[9] or c[11]
        key=(c[3],c[4],sp,dp,c[5])
        if key in done: continue
        pk={"ts":fnum(c[0]),"proto":fnum(c[5]),"sport":fnum(sp),"dport":fnum(dp),
            "iplen":fnum(c[6]),"ttl":fnum(c[7]),"win":fnum(c[13]),"flags":flagnum(c[12])}
        if key not in first: first[key]=(pk,dev); continue
        p1,d=first[key]; out.append((feat(p1,pk),"i25:"+d)); done.add(key); del first[key]
    return out, vic

print("Reading CICIoT2023 Mirai + benign (zip)...",flush=True)
c23_atk=c23_rows(C23_MIRAI_CSVS,True); c23_ben=c23_rows(C23_BENIGN_CSVS,False)
print(f"  c23 attack={len(c23_atk)} benign={len(c23_ben)}")

print("Reading CIC IIoT 2025 Mirai (tshark)...",flush=True)
i25_atk=[]; I25_VICS=set()
for p in I25_MIRAI:
    rows,vic=i25_pcap_rows(p,True);
    if vic: I25_VICS.add(vic)
    i25_atk+=rows
I25_BENIGN_VICS=I25_VICS
i25_ben,_=i25_pcap_rows(I25_BENIGN,False)
print(f"  i25 attack={len(i25_atk)} benign={len(i25_ben)} victims={len(I25_VICS)}")

# assemble
atk=c23_atk+i25_atk; ben=c23_ben+i25_ben
Xa=np.array([r[0] for r in atk],float); da=np.array([r[1] for r in atk])
Xb=np.array([r[0] for r in ben],float); db=np.array([r[1] for r in ben])
victims=sorted(set(da)); print(f"\nPOOLED victim devices: {len(victims)} ({sum(v.startswith('c23') for v in victims)} c23 + {sum(v.startswith('i25') for v in victims)} i25)")

def model(): return XGBClassifier(n_estimators=150,max_depth=6,learning_rate=0.1,eval_metric="logloss",random_state=42,n_jobs=1,tree_method="hist")
def ev(Xtr,ytr,Xte,yte):
    Xf,Xc,yf,yc=train_test_split(Xtr,ytr,test_size=0.2,random_state=7,stratify=ytr)
    clf=model(); clf.fit(Xf,yf); sc=clf.predict_proba(Xc)[:,1]; st=clf.predict_proba(Xte)[:,1]
    thr=float(np.quantile(sc[yc==0],0.99)) if (yc==0).any() else 0.5
    pd_=(st>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pd_,labels=[0,1]).ravel()
    return (roc_auc_score(yte,st) if len(set(yte))>1 else float('nan'), fp/(fp+tn) if (fp+tn) else float('nan'))

# ---- ORIGIN CONTROL: predict dataset from BENIGN only ----
yo=np.array([0 if d.startswith("c23") else 1 for d in db])
if len(set(yo))>1:
    Xtr,Xte,ytr,yte=train_test_split(Xb,yo,test_size=0.3,random_state=42,stratify=yo)
    oa,_=ev(Xtr,ytr,Xte,yte)
    print(f"\n[ORIGIN CONTROL] predict dataset-of-origin from BENIGN only: AUC {oa:.3f}  (want LOW; ~0.5 = labs indistinguishable)")
else:
    oa=None; print("\n[ORIGIN CONTROL] skipped (benign from one dataset only)")

# ---- POOLED device-disjoint (mix both datasets both sides via random holdout of the combined pool) ----
Xall=np.vstack([Xa,Xb]); yall=np.concatenate([np.ones(len(Xa)),np.zeros(len(Xb))])
Xtr,Xte,ytr,yte=train_test_split(Xall,yall,test_size=0.3,random_state=42,stratify=yall)
ra,rf=ev(Xtr,ytr,Xte,yte); print(f"\n[RANDOM pooled] AUC {ra:.3f} FPR {rf:.3f}")
aucs=[];fprs=[];mix=[]
ben_dev=db
for seed in [1,7,42,100,1729]:
    rng=np.random.default_rng(seed); perm=list(victims); rng.shuffle(perm)
    held=set(perm[:max(1,round(len(perm)*0.3))])
    ta=np.isin(da,list(held)); tb=np.isin(ben_dev,list(held))
    if ta.sum()==0 or (~ta).sum()==0 or tb.sum()==0 or (~tb).sum()==0: continue
    Xtr=np.vstack([Xa[~ta],Xb[~tb]]); ytr=np.concatenate([np.ones((~ta).sum()),np.zeros((~tb).sum())])
    Xte=np.vstack([Xa[ta],Xb[tb]]);  yte=np.concatenate([np.ones(ta.sum()),np.zeros(tb.sum())])
    a,fp=ev(Xtr,ytr,Xte,yte); aucs.append(a); fprs.append(fp)
    heldmix=f"{sum(h.startswith('c23') for h in held)}c23+{sum(h.startswith('i25') for h in held)}i25"
    mix.append(heldmix)
    print(f"[POOLED DISJOINT seed={seed:4d}] held {len(held)} dev ({heldmix})  AUC {a:.3f} FPR {fp:.3f}")
print(f"\n[POOLED] DISJOINT AUC {min(aucs):.3f}-{max(aucs):.3f} (med {stt.median(aucs):.3f})  FPR {min(fprs):.3f}-{max(fprs):.3f} (med {stt.median(fprs):.3f})")
print("vs Mirai ALONE: CICIoT2023 AUC 0.66 · CIC IIoT 2025 (8 dev) AUC 0.52-0.98/FPR 97%")
print("READ: pooled AUC high + FPR low => more devices rescued Mirai. Still failing => Mirai fundamentally victim-bound.")
print("      BUT only trust it if ORIGIN CONTROL AUC is near 0.5 (else the model may be reading the lab, not the attack).")
json.dump({"pooled_victims":len(victims),"c23_victims":sum(v.startswith('c23') for v in victims),
           "i25_victims":sum(v.startswith('i25') for v in victims),
           "origin_control_auc_benign":round(oa,4) if oa is not None else None,
           "random":{"auc":round(ra,4),"fpr":round(rf,4)},
           "pooled_disjoint_auc":[round(min(aucs),4),round(stt.median(aucs),4),round(max(aucs),4)],
           "pooled_disjoint_fpr":[round(min(fprs),4),round(stt.median(fprs),4),round(max(fprs),4)],
           "held_mix_per_seed":mix},open(os.path.join(HERE,"pooled_mirai.json"),"w"),indent=2)
print("Wrote pooled_mirai.json")
