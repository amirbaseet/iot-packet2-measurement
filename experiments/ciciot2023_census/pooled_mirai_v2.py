#!/usr/bin/env python3
"""
Pooled-Mirai v2 — the feature-drop rescue.
Fixes v1's two problems: (1) i25 victims tagged by pcap FILENAME device (8 victims, not 1 -> pool = 13);
(2) greedily DROP the most lab-discriminative features until the ORIGIN CONTROL (predict lab from benign
only) falls toward 0.5 — then run the pooled device-disjoint Mirai test on the SURVIVING lab-invariant
features. If Mirai generalises on lab-invariant features AND origin control is ~0.5, pooling is honest.

Honest caveat baked in: dropping features to defeat the origin control may also drop attack signal. A high
pooled AUC is only trustworthy when the origin control is genuinely low; we report BOTH at every step.
"""
import zipfile, io, subprocess, glob, os, re, collections, json, statistics as stt
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
FEATNAMES=["proto","sport","dport","iplen1","iplen2","ttl1","ttl2","win1","win2","flags1","flags2","iat"]

def fnum(x):
    try:
        v=float(x); return v if v==v else 0.0
    except: return 0.0
def flagnum(x):
    x=(x or "").strip()
    if not x: return 0.0
    try: return float(int(x,16)) if x.lower().startswith("0x") else float(x)
    except: return 0.0
def feat(p1,p2):
    return [p1["proto"],p1["sport"],p1["dport"],p1["iplen"],p2["iplen"],p1["ttl"],p2["ttl"],
            p1["win"],p2["win"],p1["flags"],p2["flags"],(p2["ts"]-p1["ts"]) if p2["ts"]>=p1["ts"] else 0.0]

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

FIELDS=["frame.time_epoch","eth.src","eth.dst","ip.src","ip.dst","ip.proto","ip.len","ip.ttl",
        "tcp.srcport","tcp.dstport","udp.srcport","udp.dstport","tcp.flags","tcp.window_size"]
def tshark(pcap):
    r=subprocess.run(["tshark","-r",pcap,"-T","fields"]+sum([["-e",f] for f in FIELDS],[]),capture_output=True,text=True)
    for line in r.stdout.splitlines(): yield line.split("\t")
def i25_attack_rows(pcap, devname):    # tag EVERY flow in this victim's pcap by the filename device
    first={}; done=set(); out=[]
    for c in tshark(pcap):
        if len(c)<14: continue
        sp=c[8] or c[10]; dp=c[9] or c[11]; key=(c[3],c[4],sp,dp,c[5])
        if key in done: continue
        pk={"ts":fnum(c[0]),"proto":fnum(c[5]),"sport":fnum(sp),"dport":fnum(dp),
            "iplen":fnum(c[6]),"ttl":fnum(c[7]),"win":fnum(c[13]),"flags":flagnum(c[12])}
        if key not in first: first[key]=pk; continue
        out.append((feat(first[key],pk),"i25:"+devname)); done.add(key); del first[key]
    return out
def i25_benign_rows(pcap, vic_devnames):
    first={}; done=set(); out=[]
    for c in tshark(pcap):
        if len(c)<14: continue
        sp=c[8] or c[10]; dp=c[9] or c[11]; key=(c[3],c[4],sp,dp,c[5])
        if key in done: continue
        pk={"ts":fnum(c[0]),"proto":fnum(c[5]),"sport":fnum(sp),"dport":fnum(dp),
            "iplen":fnum(c[6]),"ttl":fnum(c[7]),"win":fnum(c[13]),"flags":flagnum(c[12])}
        if key not in first: first[key]=pk; continue
        out.append((feat(first[key],pk),"i25:benign")); done.add(key); del first[key]  # benign not device-split for i25
    return out

print("Reading CICIoT2023 (zip)...",flush=True)
c23_atk=c23_rows(C23_MIRAI_CSVS,True); c23_ben=c23_rows(C23_BENIGN_CSVS,False)
print(f"  c23 attack={len(c23_atk)} benign={len(c23_ben)}")
print("Reading CIC IIoT 2025 Mirai (tshark, per-file victim)...",flush=True)
i25_atk=[]; i25_vics=[]
for p in I25_MIRAI:
    dev=re.sub(r".*mirai-[a-z]+-flood_|--.*|\.pcap$","",os.path.basename(p))
    rows=i25_attack_rows(p,dev); i25_atk+=rows; i25_vics.append(dev)
i25_vics=sorted(set(i25_vics)); i25_ben=i25_benign_rows(I25_BENIGN,i25_vics)
print(f"  i25 attack={len(i25_atk)} benign={len(i25_ben)} victims={len(i25_vics)} -> {i25_vics}")

atk=c23_atk+i25_atk; ben=c23_ben+i25_ben
Xa=np.array([r[0] for r in atk],float); da=np.array([r[1] for r in atk])
Xb=np.array([r[0] for r in ben],float); db=np.array([r[1] for r in ben])
victims=sorted(set(da))
print(f"\nPOOLED victims: {len(victims)} ({sum(v.startswith('c23') for v in victims)} c23 + {sum(v.startswith('i25') for v in victims)} i25)")

def model(): return XGBClassifier(n_estimators=150,max_depth=6,learning_rate=0.1,eval_metric="logloss",random_state=42,n_jobs=1,tree_method="hist")
def ev(Xtr,ytr,Xte,yte):
    Xf,Xc,yf,yc=train_test_split(Xtr,ytr,test_size=0.2,random_state=7,stratify=ytr)
    clf=model(); clf.fit(Xf,yf); sc=clf.predict_proba(Xc)[:,1]; st=clf.predict_proba(Xte)[:,1]
    thr=float(np.quantile(sc[yc==0],0.99)) if (yc==0).any() else 0.5
    pd_=(st>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pd_,labels=[0,1]).ravel()
    return (roc_auc_score(yte,st) if len(set(yte))>1 else float('nan'), fp/(fp+tn) if (fp+tn) else float('nan'))

def origin_auc(cols):
    yo=np.array([0 if d.startswith("c23") else 1 for d in db])
    Xtr,Xte,ytr,yte=train_test_split(Xb[:,cols],yo,test_size=0.3,random_state=42,stratify=yo)
    clf=model(); clf.fit(Xtr,ytr)
    return roc_auc_score(yte,clf.predict_proba(Xte)[:,1]), clf.feature_importances_

# ---- greedy feature drop until origin control <= 0.6 ----
cols=list(range(len(FEATNAMES))); dropped=[]
print("\n[FEATURE-DROP to defeat the lab confound]")
while True:
    oa,imp=origin_auc(cols)
    print(f"  features={[FEATNAMES[c] for c in cols]}  origin AUC={oa:.3f}")
    if oa<=0.60 or len(cols)<=3: break
    worst=cols[int(np.argmax(imp))]; dropped.append(FEATNAMES[worst]); cols.remove(worst)
    print(f"    -> drop '{FEATNAMES[worst]}' (most lab-discriminative)")
surv=[FEATNAMES[c] for c in cols]
print(f"\nSurviving lab-invariant features: {surv}  (origin AUC {oa:.3f})  dropped: {dropped}")

# ---- pooled device-disjoint on surviving features ----
Xa2,Xb2=Xa[:,cols],Xb[:,cols]
Xall=np.vstack([Xa2,Xb2]); yall=np.concatenate([np.ones(len(Xa2)),np.zeros(len(Xb2))])
Xtr,Xte,ytr,yte=train_test_split(Xall,yall,test_size=0.3,random_state=42,stratify=yall)
ra,rf=ev(Xtr,ytr,Xte,yte); print(f"\n[RANDOM pooled, surviving feats] AUC {ra:.3f} FPR {rf:.3f}")
aucs=[];fprs=[]
for seed in [1,7,42,100,1729]:
    rng=np.random.default_rng(seed); perm=list(victims); rng.shuffle(perm)
    held=set(perm[:max(1,round(len(perm)*0.3))])
    ta=np.isin(da,list(held)); tb=np.isin(db,list(held))
    if ta.sum()==0 or (~ta).sum()==0 or tb.sum()==0 or (~tb).sum()==0:
        # i25 benign isn't device-split; fall back to random benign holdout for coverage
        tb=np.zeros(len(db),bool); idx=rng.choice(len(db),size=int(0.3*len(db)),replace=False); tb[idx]=True
    Xtr=np.vstack([Xa2[~ta],Xb2[~tb]]); ytr=np.concatenate([np.ones((~ta).sum()),np.zeros((~tb).sum())])
    Xte=np.vstack([Xa2[ta],Xb2[tb]]);  yte=np.concatenate([np.ones(ta.sum()),np.zeros(tb.sum())])
    a,fp=ev(Xtr,ytr,Xte,yte); aucs.append(a); fprs.append(fp)
    hm=f"{sum(h.startswith('c23') for h in held)}c23+{sum(h.startswith('i25') for h in held)}i25"
    print(f"[POOLED DISJOINT seed={seed:4d}] held {len(held)} ({hm})  AUC {a:.3f} FPR {fp:.3f}")
print(f"\n[POOLED, lab-invariant feats] DISJOINT AUC {min(aucs):.3f}-{max(aucs):.3f} (med {stt.median(aucs):.3f}) FPR {min(fprs):.3f}-{max(fprs):.3f} (med {stt.median(fprs):.3f})")
print(f"ONLY trust if origin AUC ({oa:.3f}) is near 0.5. Mirai alone: c23 0.66 / i25 0.52-0.98 FPR 97%.")
json.dump({"pooled_victims":len(victims),"surviving_features":surv,"dropped_features":dropped,
           "origin_auc_final":round(oa,4),"random":{"auc":round(ra,4),"fpr":round(rf,4)},
           "pooled_disjoint_auc":[round(min(aucs),4),round(stt.median(aucs),4),round(max(aucs),4)],
           "pooled_disjoint_fpr":[round(min(fprs),4),round(stt.median(fprs),4),round(max(fprs),4)]},
          open(os.path.join(HERE,"pooled_mirai_v2.json"),"w"),indent=2)
print("Wrote pooled_mirai_v2.json")
