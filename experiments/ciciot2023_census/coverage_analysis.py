#!/usr/bin/env python3
"""
WS1 — selective-prediction / risk-coverage reanalysis of the Mirai device-disjoint result.
Binary AUC/FPR asked "is it wrong on unseen devices?" (yes). This asks: when it is wrong, is the error
CONFIDENT (far from the decision boundary) or NEAR-THRESHOLD (an abstention band would catch it)?

NOTE ON TERMS: the spec's DEFEATED/DEGRADED/CAUGHT are defined for the EVASION suite (an attacker
deliberately forcing abstention), NOT for benign false-alarms. To avoid misusing that vocabulary we report
our own labels here: "confident-error" vs "near-threshold-error", and a threshold-free risk-coverage curve.
The analogy to the spec's abstention idea is noted, not claimed as the same measurement.

Hardened (pre-flight fixes): reads a LOCAL copy of the zip (no mid-run SSD disconnect); NO improvised band
(error breakdown reported as a CURVE over bands); deterministic (seed + n_jobs=1); PORTS-DROPPED ablation.
Threshold = 1% FPR on SEEN devices (same operating point as the headline result).
"""
import zipfile, io, itertools, json, os, statistics as stt
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix, roc_auc_score

ZIP=os.environ.get("CICIOT_ZIP",
    "./data/CICIOT23/archive.zip")
HERE=os.path.dirname(os.path.abspath(__file__))
MIRAI={"Mirai-greeth_flood.csv":"greeth","Mirai-greip_flood.csv":"greip","Mirai-udpplain.csv":"udpplain"}
BENIGN=["BenignTraffic.csv","BenignTraffic1.csv","BenignTraffic2.csv","BenignTraffic3.csv"]
VICTIMS={"08:7c:39:ce:6e:2a","1c:12:b0:9b:0c:ec","1c:fe:2b:98:16:dd","9c:8e:cd:1d:ab:9f","cc:f4:11:9c:d0:00"}
CLASSV={"greeth":{"1c:fe:2b:98:16:dd","9c:8e:cd:1d:ab:9f"},
        "greip":{"08:7c:39:ce:6e:2a","1c:12:b0:9b:0c:ec","1c:fe:2b:98:16:dd","9c:8e:cd:1d:ab:9f"},
        "udpplain":{"08:7c:39:ce:6e:2a","1c:fe:2b:98:16:dd","cc:f4:11:9c:d0:00"}}
COVS=[1.0,0.9,0.8,0.7,0.6,0.5]
BANDS=[0.05,0.10,0.15,0.20,0.30]                       # error-breakdown reported ACROSS bands, not one
FEATS_FULL=["ip_p","sport","dport","iplen1","iplen2","ttl1","ttl2","win1","win2","flags1","flags2","pay1","pay2","iat"]
PORT_IDX=[1,2]                                         # sport,dport -> dropped in the ablation

def f(x):
    try:
        v=float(x); return v if v==v else 0.0
    except: return 0.0

def collect(member,is_attack):
    rows=[]; first={}; seen2=set(); zf=zipfile.ZipFile(ZIP)
    with zf.open(member) as fh:
        text=io.TextIOWrapper(fh,encoding="utf-8",errors="replace",newline="")
        text.readline()
        for line in text:
            p=line.rstrip("\n").split(",")
            if len(p)<=17: continue
            es,ed=p[1],p[2]
            if not is_attack and es not in VICTIMS and ed not in VICTIMS: continue
            key=(p[4],p[5],p[10],p[11],p[6])
            if key in seen2: continue
            if key not in first: first[key]=p; continue
            r1,p2=first[key],p; seen2.add(key); del first[key]
            t1,t2=f(r1[0]),f(p2[0])
            feat=[f(r1[6]),f(r1[10]),f(r1[11]),f(r1[7]),f(p2[7]),f(r1[8]),f(p2[8]),f(r1[15]),f(p2[15]),
                  f(r1[14]),f(p2[14]),f(r1[17]),f(p2[17]),(t2-t1) if t2>=t1 else 0.0]
            vt=({ed}&VICTIMS) if is_attack else ({es,ed}&VICTIMS)
            rows.append((feat,1 if is_attack else 0,frozenset(vt)))
    return rows

print(f"Extracting flows from {os.path.basename(ZIP)} ...",flush=True)
data=[]
for m in MIRAI: data+=collect(m,True)
for m in BENIGN: data+=collect(m,False)
Xall=np.array([d[0] for d in data],float); y=np.array([d[1] for d in data]); vic=[d[2] for d in data]
print(f"  {len(data):,} flows ({int(y.sum()):,} attack / {int((y==0).sum()):,} benign)")

def model(): return XGBClassifier(n_estimators=150,max_depth=6,learning_rate=0.1,eval_metric="logloss",
                                  random_state=42,n_jobs=1,tree_method="hist")   # deterministic
def ok(ts): return all((cv&ts) and (cv-ts) for cv in CLASSV.values())

def risk_coverage(st,yte,thr):
    margin=np.abs(st-thr); order=np.argsort(-margin)      # most-confident (far from boundary) first
    out=[]
    for cov in COVS:
        k=max(1,int(round(cov*len(order)))); idx=order[:k]
        pred=(st[idx]>=thr).astype(int); yk=yte[idx]
        tn,fp,fn,tp=confusion_matrix(yk,pred,labels=[0,1]).ravel()
        out.append({"coverage":cov,"fpr":round(fp/(fp+tn),4) if (fp+tn) else None,
                    "recall":round(recall_score(yk,pred,zero_division=0),4)})
    return out

def error_band_curve(st,yte,thr):
    """Of benign flagged as attack (s>=thr), fraction within [thr,thr+band) for each band = near-threshold."""
    fp=st[(yte==0)&(st>=thr)]
    if len(fp)==0: return {"n_fp":0}
    return {"n_fp":int(len(fp)),
            "pct_near_thr":{f"{b}":round(100*np.mean(fp<thr+b),1) for b in BANDS}}

def run(feat_idx,tag):
    X=Xall[:,feat_idx]
    print(f"\n===== {tag} ({len(feat_idx)} features) =====")
    rows=[]
    for ts in map(frozenset,itertools.combinations(sorted(VICTIMS),2)):
        if not ok(ts): continue
        s=np.array([bool(vic[i]&ts) for i in range(len(data))])
        Xf,Xc,yf,yc=train_test_split(X[~s],y[~s],test_size=0.2,random_state=7,stratify=y[~s])
        clf=model(); clf.fit(Xf,yf)
        sc=clf.predict_proba(Xc)[:,1]; st=clf.predict_proba(X[s])[:,1]; yte=y[s]
        thr=float(np.quantile(sc[yc==0],0.99)) if (yc==0).any() else 0.5
        auc=roc_auc_score(yte,st) if len(set(yte))>1 else None
        rc=risk_coverage(st,yte,thr); eb=error_band_curve(st,yte,thr)
        rows.append({"test_victims":sorted(ts),"auc":round(auc,4) if auc else None,"thr":round(thr,4),
                     "risk_coverage":rc,"error_band_curve":eb})
    def med(i,k):
        v=[r["risk_coverage"][i][k] for r in rows if r["risk_coverage"][i][k] is not None]
        return round(stt.median(v),4) if v else None
    print(f"  {'coverage':>9s} {'median FPR':>11s} {'median recall':>13s}")
    for i,cov in enumerate(COVS): print(f"  {int(cov*100):8d}% {med(i,'fpr'):>11} {med(i,'recall'):>13}")
    # near-threshold error, median across partitions, per band
    for b in BANDS:
        vals=[r["error_band_curve"]["pct_near_thr"][f"{b}"] for r in rows if r["error_band_curve"].get("pct_near_thr")]
        if vals: print(f"  band +{b}: median {round(stt.median(vals),1)}% of false-alarms are near-threshold")
    aucs=[r["auc"] for r in rows if r["auc"] is not None]
    print(f"  (AUC range {min(aucs)}-{max(aucs)}; partitions with AUC<0.5 have meaningless confidence ordering)")
    return {"tag":tag,"features":[FEATS_FULL[i] for i in feat_idx],
            "median_fpr_by_coverage":{f"{int(c*100)}pct":med(i,'fpr') for i,c in enumerate(COVS)},
            "partitions":rows}

full=run(list(range(len(FEATS_FULL))),"FULL feature set")
noports=run([i for i in range(len(FEATS_FULL)) if i not in PORT_IDX],"PORTS-DROPPED ablation")
json.dump({"note":"selective prediction on unseen-device Mirai; thr=1%FPR on seen; deterministic seed=42 n_jobs=1",
           "full":full,"ports_dropped":noports},
          open(os.path.join(HERE,"coverage_result.json"),"w"),indent=2)
print("\nREAD:")
print(" - FPR falling sharply as coverage drops => errors are near-threshold => model honestly uncertain (abstention helps).")
print(" - FPR flat as coverage drops => errors are confident => genuinely fooled (abstention does NOT help).")
print(" - FULL vs PORTS-DROPPED close => ports were not carrying the signal (behavioural claim holds).")
print("Wrote coverage_result.json")
