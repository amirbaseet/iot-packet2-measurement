#!/usr/bin/env python3
"""
WS2 Step 1 — attacker-disjoint recon test. Does packet-2 scan detection transfer to an UNSEEN SCANNER,
or does it fingerprint the specific scanner machine (the attacker-side analogue of #52's victim fingerprint)?

Per class: ATTACK flows = flows whose src is a real scanner (fan-out >=50 distinct dst IPs in that class,
non-infra) — this cuts the per-file label noise (the CSV is labeled by scenario, not per-flow). BENIGN =
device-matched benign flows (touching the same devices), random-split (benign has no attacker identity, so
it cannot be attacker-split; the claim under test is generalization to unseen ATTACKERS).

Diagnostic = RANDOM split (scanners shared) vs ATTACKER-DISJOINT (train on some scanners, test on unseen ones),
swept over scanner partitions. Behavioural packet-2 features only; PORTS-DROPPED ablation; threshold = 1% FPR
on seen data; deterministic. Reads LOCAL zip.
"""
import zipfile, io, itertools, json, os, collections, statistics as stt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix, roc_auc_score

ZIP=os.environ.get("CICIOT_ZIP",
    "./data/CICIOT23/archive.zip")
HERE=os.path.dirname(os.path.abspath(__file__))
RECON={"Recon-OSScan.csv":"OSScan","Recon-PortScan.csv":"PortScan","Recon-PingSweep.csv":"PingSweep",
       "Recon-HostDiscovery.csv":"HostDiscovery","VulnerabilityScan.csv":"VulnerabilityScan"}
BENIGN=["BenignTraffic.csv","BenignTraffic1.csv","BenignTraffic2.csv","BenignTraffic3.csv"]
INFRA={"3c:18:a0:41:c3:a0","44:bb:3b:00:39:07"}
SCANNER_FANOUT=50          # a source is a "scanner" for a class if it hits >= this many distinct dst IPs
FEATS_FULL=["ip_p","sport","dport","iplen1","iplen2","ttl1","ttl2","win1","win2","flags1","flags2","pay1","pay2","iat"]
PORT_IDX=[1,2]

def f(x):
    try:
        v=float(x); return v if v==v else 0.0
    except: return 0.0
def is_real(mac):
    if not mac or ":" not in mac or mac=="ff:ff:ff:ff:ff:ff": return False
    try: return not (int(mac.split(":")[0],16)&1)
    except ValueError: return False
def feats(r1,p2):
    t1,t2=f(r1[0]),f(p2[0])
    return [f(r1[6]),f(r1[10]),f(r1[11]),f(r1[7]),f(p2[7]),f(r1[8]),f(p2[8]),f(r1[15]),f(p2[15]),
            f(r1[14]),f(p2[14]),f(r1[17]),f(p2[17]),(t2-t1) if t2>=t1 else 0.0]

def scanners_for(member):
    """first pass: identify scanner source MACs (fan-out>=SCANNER_FANOUT) for this class."""
    src_dsts=collections.defaultdict(set); zf=zipfile.ZipFile(ZIP)
    with zf.open(member) as fh:
        text=io.TextIOWrapper(fh,encoding="utf-8",errors="replace",newline=""); text.readline()
        for line in text:
            p=line.rstrip("\n").split(",")
            if len(p)<=11: continue
            if is_real(p[1]) and p[1] not in INFRA: src_dsts[p[1]].add(p[5])
    return {s for s,d in src_dsts.items() if len(d)>=SCANNER_FANOUT}

def collect_attack(member,scanners):
    """packet-2 features for flows whose src is a scanner; tag flow with its scanner source."""
    rows=[]; first={}; seen=set(); zf=zipfile.ZipFile(ZIP)
    with zf.open(member) as fh:
        text=io.TextIOWrapper(fh,encoding="utf-8",errors="replace",newline=""); text.readline()
        for line in text:
            p=line.rstrip("\n").split(",")
            if len(p)<=17 or p[1] not in scanners: continue
            key=(p[4],p[5],p[10],p[11],p[6])
            if key in seen: continue
            if key not in first: first[key]=p; continue
            rows.append((feats(first[key],p),p[1])); seen.add(key); del first[key]
    return rows      # (feature, scanner_src)

# benign pool: flows touching any recon-scanner device; TAG which scanner(s) each benign flow touches
# so the held-out scanner's benign can be held out too (a true device holdout, not just attacker holdout).
def collect_benign(all_scanners):
    rows=[]; zf=zipfile.ZipFile(ZIP)
    for member in BENIGN:
        first={}; seen=set()
        with zf.open(member) as fh:
            text=io.TextIOWrapper(fh,encoding="utf-8",errors="replace",newline=""); text.readline()
            for line in text:
                p=line.rstrip("\n").split(",")
                if len(p)<=17: continue
                touch={p[1],p[2]} & all_scanners
                if not touch: continue
                key=(p[4],p[5],p[10],p[11],p[6])
                if key in seen: continue
                if key not in first: first[key]=p; continue
                rows.append((feats(first[key],p),frozenset(touch))); seen.add(key); del first[key]
    return rows

def model(): return XGBClassifier(n_estimators=150,max_depth=6,learning_rate=0.1,eval_metric="logloss",
                                  random_state=42,n_jobs=1,tree_method="hist")

def evalu(Xtr,ytr,Xte,yte):
    Xf,Xc,yf,yc=train_test_split(Xtr,ytr,test_size=0.2,random_state=7,stratify=ytr)
    clf=model(); clf.fit(Xf,yf); sc=clf.predict_proba(Xc)[:,1]; st=clf.predict_proba(Xte)[:,1]
    thr=float(np.quantile(sc[yc==0],0.99)) if (yc==0).any() else 0.5
    pd_=(st>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pd_,labels=[0,1]).ravel()
    return {"auc":round(roc_auc_score(yte,st),4) if len(set(yte))>1 else None,
            "recall":round(recall_score(yte,pd_,zero_division=0),4),
            "fpr":round(fp/(fp+tn),4) if (fp+tn) else None}

print("Identifying scanners per class + extracting flows...",flush=True)
all_scanners=set(); per_class_scanners={}
for m,name in RECON.items():
    sc=scanners_for(m); per_class_scanners[name]=sc; all_scanners|=sc
_ben=collect_benign(all_scanners)
benign=np.array([b[0] for b in _ben],float); benign_touch=[b[1] for b in _ben]
print(f"  benign pool: {len(benign):,} flows; total distinct scanners across recon: {len(all_scanners)}")

def run(feat_idx,tag):
    print(f"\n===== {tag} =====")
    results={}
    for m,name in RECON.items():
        scanners=sorted(per_class_scanners[name])
        if len(scanners)<2:
            results[name]={"skip":"<2 scanners"}; print(f"  {name}: SKIP (<2 scanners)"); continue
        atk=collect_attack(m,set(scanners))
        Xa=np.array([a[0] for a in atk],float)[:,feat_idx]; srcs=np.array([a[1] for a in atk])
        Xb=benign[:,feat_idx]
        # RANDOM split (scanners shared)
        X=np.vstack([Xa,Xb]); yv=np.concatenate([np.ones(len(Xa)),np.zeros(len(Xb))])
        Xtr,Xte,ytr,yte=train_test_split(X,yv,test_size=0.3,random_state=42,stratify=yv)
        rnd=evalu(Xtr,ytr,Xte,yte)
        # ATTACKER-DISJOINT (TRUE device holdout): held scanner's ATTACK *and* BENIGN go to test only.
        aucs=[];fprs=[];rc=[]
        combos=list(itertools.combinations(scanners,max(1,round(len(scanners)*0.3))))
        import random as _r; _r.Random(1).shuffle(combos); combos=combos[:8]   # cap partitions
        for held in combos:
            held=set(held); te_a=np.isin(srcs,list(held))
            b_te=np.array([bool(t & held) for t in benign_touch])   # benign touching a held scanner -> test
            if te_a.sum()==0 or (~te_a).sum()==0 or b_te.sum()==0 or (~b_te).sum()==0: continue
            Xtr=np.vstack([Xa[~te_a],Xb[~b_te]]); ytr=np.concatenate([np.ones((~te_a).sum()),np.zeros((~b_te).sum())])
            Xte=np.vstack([Xa[te_a],Xb[b_te]]);  yte=np.concatenate([np.ones(te_a.sum()),np.zeros(b_te.sum())])
            r=evalu(Xtr,ytr,Xte,yte); aucs.append(r["auc"]); fprs.append(r["fpr"]); rc.append(r["recall"])
        results[name]={"n_scanners":len(scanners),"n_attack_flows":len(Xa),
                       "random":rnd,
                       "attacker_disjoint":{"auc_range":[min(aucs),max(aucs)],"auc_median":round(stt.median(aucs),4),
                                            "fpr_range":[min(fprs),max(fprs)],"fpr_median":round(stt.median(fprs),4),
                                            "recall_median":round(stt.median(rc),4),"n_partitions":len(aucs)}}
        rd=results[name]
        print(f"  {name:18s} scanners={len(scanners):2d} atk_flows={len(Xa):>6,}  "
              f"RANDOM auc={rnd['auc']} fpr={rnd['fpr']}  |  DISJOINT auc={rd['attacker_disjoint']['auc_median']} "
              f"(range {rd['attacker_disjoint']['auc_range']}) fpr={rd['attacker_disjoint']['fpr_median']}")
    return results

full=run(list(range(len(FEATS_FULL))),"FULL feature set")
noports=run([i for i in range(len(FEATS_FULL)) if i not in PORT_IDX],"PORTS-DROPPED ablation")
json.dump({"scanner_fanout_threshold":SCANNER_FANOUT,"note":"attack=scanner-src flows; benign device-matched random-split; thr=1%FPR seen; seed42 n_jobs1",
           "full":full,"ports_dropped":noports},open(os.path.join(HERE,"recon_result.json"),"w"),indent=2)
print("\nREAD: RANDOM auc>>DISJOINT auc (with DISJOINT fpr high) => scan detection fingerprints the SCANNER (attacker-disjoint fails, like #52 on the victim side).")
print("      RANDOM ~ DISJOINT => scan detection GENERALIZES to unseen scanners (a positive result DDoS could not give).")
print("Wrote recon_result.json")
