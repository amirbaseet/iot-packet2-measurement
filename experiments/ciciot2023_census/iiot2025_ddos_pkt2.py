#!/usr/bin/env python3
"""
THE gating experiment: PACKET-2 victim-disjoint DDoS detection on CIC IIoT 2025 (many victim devices).
Resolves whether the fingerprinting was PACKET-2 (too thin) or TOO FEW DEVICES.
  - CICIoT2023 Mirai (packet-2, ~5 victims) -> collapsed (AUC 0.66, FPR 10-99%).
  - CIC IIoT 2025 udp-frag-flood is captured PER TARGET DEVICE (~10+ victims).
  - If packet-2 GENERALISES here -> it was device count. If it FAILS -> packet-2 is genuinely too thin.

Attack = udp-frag-flood pcaps, one per target device (victim = filename device; its MAC = dominant eth.dst).
Benign = the whole-network benign pcap, per-device by endpoint MAC. Packet-2 behavioural features only
(no MAC/IP as features; identity only builds the split). TRUE victim holdout (held device's flood AND benign
in test). RANDOM vs victim-disjoint, multi-seed, threshold = 1% FPR on seen. tshark -> flows -> first 2 packets.
"""
import subprocess, os, glob, collections, itertools, json, statistics as stt, re
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score

EX="./data/iiot/extract"
FLOODS=sorted(p for p in glob.glob(EX+"/ddos/*udp-frag-flood*.pcap") if "/._" not in p)
BENIGN=EX+"/benign/benign_whole-network3.pcap"
HERE=os.path.dirname(os.path.abspath(__file__))
FIELDS=["frame.time_epoch","eth.src","eth.dst","ip.src","ip.dst","ip.proto","ip.len","ip.ttl",
        "tcp.srcport","tcp.dstport","udp.srcport","udp.dstport","tcp.flags","tcp.window_size","frame.len"]

def tshark(pcap):
    p=subprocess.run(["tshark","-r",pcap,"-T","fields"]+sum([["-e",f] for f in FIELDS],[]),
                     capture_output=True,text=True)
    for line in p.stdout.splitlines():
        yield line.split("\t")

def fnum(x):
    try:
        v=float(x); return v if v==v else 0.0
    except: return 0.0

def flows_from(pcap):
    """assemble flows, return (feature-list, victim_dst_mac) per flow from first 2 packets."""
    first={}; done=set(); rows=[]; dstmac=collections.Counter()
    for c in tshark(pcap):
        if len(c)<15: continue
        ts,esrc,edst,ips,ipd,proto,iplen,ttl,tsp,tdp,usp,udp_,flags,win,flen=c[:15]
        sport=tsp or usp; dport=tdp or udp_
        key=(ips,ipd,sport,dport,proto)
        if key in done: continue
        dstmac[edst]+=1
        if key not in first: first[key]=c; continue
        c1=first[key]
        t1,t2=fnum(c1[0]),fnum(ts)
        feat=[fnum(proto),fnum(sport),fnum(dport),fnum(c1[6]),fnum(iplen),fnum(c1[7]),fnum(ttl),
              fnum(c1[13]),fnum(win),fnum(c1[12]),fnum(flags),fnum(c1[14]),fnum(flen),(t2-t1) if t2>=t1 else 0.0]
        rows.append(feat); done.add(key); del first[key]
    vic=dstmac.most_common(1)[0][0] if dstmac else None
    return rows, vic

print(f"flood pcaps: {len(FLOODS)}",flush=True)
X=[]; y=[]; dev=[]           # attack rows tagged by victim device name
vic_mac={}
for p in FLOODS:
    name=re.sub(r".*udp-frag-flood_|\.pcap$","",os.path.basename(p))
    rows,vmac=flows_from(p); vic_mac[name]=vmac
    for r in rows: X.append(r); y.append(1); dev.append(name)
    print(f"  {name:22s} flows={len(rows):>6} victim_mac={vmac}",flush=True)

# benign: per-flow, record which victim devices (by MAC) it touches
macset={m for m in vic_mac.values() if m}
ben_rows=[]; ben_touch=[]
first={}; done=set()          # benign: assemble flows once, capturing endpoint macs per flow
for c in tshark(BENIGN):
    if len(c)<15: continue
    ts,esrc,edst,ips,ipd,proto,iplen,ttl,tsp,tdp,usp,udp_,flags,win,flen=c[:15]
    sport=tsp or usp; dport=tdp or udp_; key=(ips,ipd,sport,dport,proto)
    if key in done: continue
    if key not in first: first[key]=(c,{esrc,edst}); continue
    c1,m1=first[key]; t1,t2=fnum(c1[0]),fnum(ts)
    feat=[fnum(proto),fnum(sport),fnum(dport),fnum(c1[6]),fnum(iplen),fnum(c1[7]),fnum(ttl),
          fnum(c1[13]),fnum(win),fnum(c1[12]),fnum(flags),fnum(c1[14]),fnum(flen),(t2-t1) if t2>=t1 else 0.0]
    ben_rows.append(feat); ben_touch.append((m1|{esrc,edst})&macset); done.add(key); del first[key]
print(f"benign flows={len(ben_rows)}  victim devices={len(vic_mac)}",flush=True)

Xa=np.array(X,float); ya=np.ones(len(X)); Xb=np.array(ben_rows,float)
devices=sorted(vic_mac)
def model(): return XGBClassifier(n_estimators=150,max_depth=6,learning_rate=0.1,eval_metric="logloss",
                                  random_state=42,n_jobs=1,tree_method="hist")
def ev(Xtr,ytr,Xte,yte):
    Xf,Xc,yf,yc=train_test_split(Xtr,ytr,test_size=0.2,random_state=7,stratify=ytr)
    clf=model(); clf.fit(Xf,yf); sc=clf.predict_proba(Xc)[:,1]; st=clf.predict_proba(Xte)[:,1]
    thr=float(np.quantile(sc[yc==0],0.99)) if (yc==0).any() else 0.5
    pd_=(st>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pd_,labels=[0,1]).ravel()
    return (roc_auc_score(yte,st) if len(set(yte))>1 else float('nan'), fp/(fp+tn) if (fp+tn) else float('nan'))

dev_arr=np.array(dev)
# RANDOM
Xall=np.vstack([Xa,Xb]); yall=np.concatenate([ya,np.zeros(len(Xb))])
Xtr,Xte,ytr,yte=train_test_split(Xall,yall,test_size=0.3,random_state=42,stratify=yall)
ra,rf=ev(Xtr,ytr,Xte,yte)
print(f"\n[RANDOM]  AUC {ra:.3f}  FPR {rf:.3f}")
# VICTIM-DISJOINT (hold out ~30% of target devices; their flood + their benign -> test)
aucs=[];fprs=[]
for seed in [1,7,42,100,1729]:
    rng=np.random.default_rng(seed); perm=list(devices); rng.shuffle(perm)
    held=set(perm[:max(1,round(len(perm)*0.3))]); heldmac={vic_mac[d] for d in held if vic_mac[d]}
    ta=np.isin(dev_arr,list(held)); tb=np.array([bool(t & heldmac) for t in ben_touch])
    if ta.sum()==0 or (~ta).sum()==0 or tb.sum()==0 or (~tb).sum()==0: continue
    Xtr=np.vstack([Xa[~ta],Xb[~tb]]); ytr=np.concatenate([np.ones((~ta).sum()),np.zeros((~tb).sum())])
    Xte=np.vstack([Xa[ta],Xb[tb]]);  yte=np.concatenate([np.ones(ta.sum()),np.zeros(tb.sum())])
    a,fp=ev(Xtr,ytr,Xte,yte); aucs.append(a); fprs.append(fp)
    print(f"[VICTIM-DISJOINT seed={seed:4d}] held {len(held)} devices  AUC {a:.3f}  FPR {fp:.3f}")
print(f"\n[VICTIM-DISJOINT] AUC {min(aucs):.3f}-{max(aucs):.3f} (median {stt.median(aucs):.3f})  FPR {min(fprs):.3f}-{max(fprs):.3f} (median {stt.median(fprs):.3f})")
print(f"\nCONTRAST: RANDOM {ra:.3f}/{rf:.3f}  vs  DISJOINT median {stt.median(aucs):.3f}/{stt.median(fprs):.3f}")
print("vs CICIoT2023 Mirai packet-2 (5 victims): AUC 0.66, FPR 10-99%.")
print("READ: generalises here => it was DEVICE COUNT, not packet-2. Fails => packet-2 genuinely too thin.")
json.dump({"n_victim_devices":len(devices),"attack_flows":len(X),"benign_flows":len(ben_rows),
           "random":{"auc":round(ra,4),"fpr":round(rf,4)},
           "victim_disjoint":{"auc":[round(min(aucs),4),round(stt.median(aucs),4),round(max(aucs),4)],
                              "fpr":[round(min(fprs),4),round(stt.median(fprs),4),round(max(fprs),4)],"n_seeds":len(aucs)}},
          open(os.path.join(HERE,"iiot2025_ddos_pkt2.json"),"w"),indent=2)
print("Wrote iiot2025_ddos_pkt2.json")
