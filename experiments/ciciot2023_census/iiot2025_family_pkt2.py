#!/usr/bin/env python3
"""
Generalized packet-2 victim-disjoint test, one CIC IIoT 2025 family per run.
Usage: iiot2025_family_pkt2.py <family_dir_under_extract>   e.g. dos, mitm, recon, web, bruteforce

Same method as iiot2025_ddos_pkt2.py / _mirai_: attack = per-target-device pcaps (victim = filename device,
its MAC = dominant eth.dst), benign = whole-network benign pcap per device, packet-2 behavioural features only
(no MAC/IP as features), TRUE victim holdout, RANDOM vs victim-disjoint, 5 seeds, thr = 1% FPR on seen.
Reports device count so thin families are visibly bounded. NOTE: for recon (attacker sprays many hosts) the
victim-disjoint frame is the WRONG axis — run reports it but the honest recon test is attacker-disjoint
(see recon_attacker_disjoint.py). Appends one row to iiot2025_family_summary.json.
"""
import subprocess, os, glob, collections, json, statistics as stt, re, sys
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

FAM=sys.argv[1] if len(sys.argv)>1 else "dos"
EX="./data/iiot/extract"
HERE=os.path.dirname(os.path.abspath(__file__))
PCAPS=sorted(p for p in glob.glob(f"{EX}/{FAM}/*.pcap") if "/._" not in p)
BENIGN=EX+"/benign/benign_whole-network3.pcap"
FIELDS=["frame.time_epoch","eth.src","eth.dst","ip.src","ip.dst","ip.proto","ip.len","ip.ttl",
        "tcp.srcport","tcp.dstport","udp.srcport","udp.dstport","tcp.flags","tcp.window_size","frame.len"]

PKT_CAP=os.environ.get("TSHARK_CAP","400000")   # read only first N packets/pcap -> bounds time on huge files
def tshark(pcap):
    p=subprocess.run(["tshark","-r",pcap,"-c",PKT_CAP,"-T","fields"]+sum([["-e",f] for f in FIELDS],[]),
                     capture_output=True,text=True)
    for line in p.stdout.splitlines(): yield line.split("\t")
def fnum(x):
    try:
        v=float(x); return v if v==v else 0.0
    except: return 0.0
def flows(pcap, cap_macs=False):
    first={}; done=set(); rows=[]; dstc=collections.Counter()
    for c in tshark(pcap):
        if len(c)<15: continue
        ts,esrc,edst,ips,ipd,proto,iplen,ttl,tsp,tdp,usp,udp_,flags,win,flen=c[:15]
        sp=tsp or usp; dp=tdp or udp_; key=(ips,ipd,sp,dp,proto)
        if key in done: continue
        dstc[edst]+=1
        if key not in first:
            first[key]=(c,{esrc,edst}) if cap_macs else c; continue
        c1=first[key][0] if cap_macs else first[key]
        t1,t2=fnum(c1[0]),fnum(ts)
        feat=[fnum(proto),fnum(sp),fnum(dp),fnum(c1[6]),fnum(iplen),fnum(c1[7]),fnum(ttl),
              fnum(c1[13]),fnum(win),fnum(c1[12]),fnum(flags),fnum(c1[14]),fnum(flen),(t2-t1) if t2>=t1 else 0.0]
        rows.append((feat, first[key][1]|{esrc,edst}) if cap_macs else feat)
        done.add(key); del first[key]
    return rows,(dstc.most_common(1)[0][0] if dstc else None)

if not PCAPS:
    print(f"no pcaps for family '{FAM}' under {EX}/{FAM}/"); sys.exit(1)
print(f"family={FAM}  pcaps={len(PCAPS)}",flush=True)
X=[];dev=[];vic_mac={}
for p in PCAPS:
    name=re.sub(r".*_|--.*|\.pcap$","",os.path.basename(p))
    rows,vmac=flows(p); vic_mac[name]=vmac
    for r in rows: X.append(r); dev.append(name)
    print(f"  {name:26s} flows={len(rows):>6} victim_mac={vmac}",flush=True)
macset={m for m in vic_mac.values() if m}
brows,_=flows(BENIGN,cap_macs=True)
Xb=np.array([b[0] for b in brows],float); ben_touch=[b[1]&macset for b in brows]
Xa=np.array(X,float); dev_arr=np.array(dev); devices=sorted(vic_mac)
print(f"attack flows={len(Xa)} benign flows={len(Xb)} victim devices={len(devices)}",flush=True)

def model(): return XGBClassifier(n_estimators=150,max_depth=6,learning_rate=0.1,eval_metric="logloss",
                                  random_state=42,n_jobs=1,tree_method="hist")
def ev(Xtr,ytr,Xte,yte):
    Xf,Xc,yf,yc=train_test_split(Xtr,ytr,test_size=0.2,random_state=7,stratify=ytr)
    clf=model(); clf.fit(Xf,yf); sc=clf.predict_proba(Xc)[:,1]; st=clf.predict_proba(Xte)[:,1]
    thr=float(np.quantile(sc[yc==0],0.99)) if (yc==0).any() else 0.5
    pd_=(st>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pd_,labels=[0,1]).ravel()
    return (roc_auc_score(yte,st) if len(set(yte))>1 else float('nan'), fp/(fp+tn) if (fp+tn) else float('nan'))

Xall=np.vstack([Xa,Xb]); yall=np.concatenate([np.ones(len(Xa)),np.zeros(len(Xb))])
Xtr,Xte,ytr,yte=train_test_split(Xall,yall,test_size=0.3,random_state=42,stratify=yall)
ra,rf=ev(Xtr,ytr,Xte,yte)
print(f"\n[RANDOM] AUC {ra:.3f} FPR {rf:.3f}")
aucs=[];fprs=[]
if len(devices)>=2:
    for seed in [1,7,42,100,1729]:
        rng=np.random.default_rng(seed); perm=list(devices); rng.shuffle(perm)
        held=set(perm[:max(1,round(len(perm)*0.3))]); hm={vic_mac[d] for d in held if vic_mac[d]}
        ta=np.isin(dev_arr,list(held)); tb=np.array([bool(t&hm) for t in ben_touch])
        if ta.sum()==0 or (~ta).sum()==0 or tb.sum()==0 or (~tb).sum()==0: continue
        Xtr=np.vstack([Xa[~ta],Xb[~tb]]); ytr=np.concatenate([np.ones((~ta).sum()),np.zeros((~tb).sum())])
        Xte=np.vstack([Xa[ta],Xb[tb]]);  yte=np.concatenate([np.ones(ta.sum()),np.zeros(tb.sum())])
        a,fp=ev(Xtr,ytr,Xte,yte); aucs.append(a); fprs.append(fp)
        print(f"[DISJOINT seed={seed:4d}] held {len(held)} dev  AUC {a:.3f} FPR {fp:.3f}")
row={"family":FAM,"n_victim_devices":len(devices),"attack_flows":len(Xa),"benign_flows":len(Xb),
     "random_auc":round(ra,4),"random_fpr":round(rf,4),
     "disjoint_auc":[round(min(aucs),4),round(stt.median(aucs),4),round(max(aucs),4)] if aucs else None,
     "disjoint_fpr":[round(min(fprs),4),round(stt.median(fprs),4),round(max(fprs),4)] if aucs else None,
     "n_seeds":len(aucs),"note":"victim-disjoint frame; for recon the honest axis is attacker-disjoint" if FAM=="recon" else ""}
if aucs:
    print(f"\n[{FAM}] DISJOINT AUC {min(aucs):.3f}-{max(aucs):.3f} (med {stt.median(aucs):.3f}) FPR {min(fprs):.3f}-{max(fprs):.3f} (med {stt.median(fprs):.3f}) | {len(devices)} devices")
else:
    print(f"\n[{FAM}] too thin for victim-disjoint ({len(devices)} devices)")
sf=os.path.join(HERE,"iiot2025_family_summary.json")
allrows=json.load(open(sf)) if os.path.exists(sf) else []
allrows=[r for r in allrows if r["family"]!=FAM]+[row]
json.dump(allrows,open(sf,"w"),indent=2)
print(f"Appended to iiot2025_family_summary.json")
