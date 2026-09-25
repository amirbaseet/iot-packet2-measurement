#!/usr/bin/env python3
"""
Mirai #52 test (robust): does packet-2 detection of CICIoT2023 Mirai floods HOLD on UNSEEN victim
devices, or COLLAPSE (victim fingerprinting -> #52 generalizes to IoT DDoS)?

Diagnostic = RANDOM split vs DEVICE-DISJOINT split (the #52 method), swept over ALL valid
test-victim partitions (extract once, loop partitions) so the verdict is not one lucky draw.
Model input = behavioral packet-2 features ONLY (no MAC/IP as features); device identity is used
ONLY to build the split. Test-victim devices appear NOWHERE in training (benign included).
"""
import zipfile, io, itertools, json, os
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix

ZIP = "./data/CICIOT23/archive.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
MIRAI = {"Mirai-greeth_flood.csv":"greeth","Mirai-greip_flood.csv":"greip","Mirai-udpplain.csv":"udpplain"}
BENIGN = ["BenignTraffic.csv","BenignTraffic1.csv","BenignTraffic2.csv","BenignTraffic3.csv"]
VICTIMS = {"08:7c:39:ce:6e:2a","1c:12:b0:9b:0c:ec","1c:fe:2b:98:16:dd","9c:8e:cd:1d:ab:9f","cc:f4:11:9c:d0:00"}
FEATS = ["ip_p","sport","dport","iplen1","iplen2","ttl1","ttl2","win1","win2","flags1","flags2","pay1","pay2","iat"]

def f(x):
    try: return float(x)
    except: return np.nan

def feats(p1,p2):
    t1,t2=f(p1[0]),f(p2[0])
    return [f(p1[6]),f(p1[10]),f(p1[11]),f(p1[7]),f(p2[7]),f(p1[8]),f(p2[8]),
            f(p1[15]),f(p2[15]),f(p1[14]),f(p2[14]),f(p1[17]),f(p2[17]),
            (t2-t1) if (np.isfinite(t1) and np.isfinite(t2)) else np.nan]

def collect(member, is_attack):
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
            row=feats(first[key],p); seen2.add(key); del first[key]
            vt = {ed} & VICTIMS if is_attack else ({es,ed} & VICTIMS)  # victim device(s) this flow touches
            rows.append((row, 1 if is_attack else 0, MIRAI.get(member,"benign"), frozenset(vt)))
    return rows

print("Extracting flows once...", flush=True)
data=[]
for m in MIRAI:  r=collect(m,True);  print(f"  {m}: {len(r):,}",flush=True); data+=r
for m in BENIGN: r=collect(m,False); print(f"  {m}: {len(r):,} benign",flush=True); data+=r
X=np.array([d[0] for d in data],dtype=float)
y=np.array([d[1] for d in data])
cls=np.array([d[2] for d in data])
vic=[d[3] for d in data]     # set of victim devices each flow touches

def fit_eval(Xtr_all,ytr_all,Xte,yte):
    """Train, pick threshold for 1% FPR on a held-out slice of TRAINING data (seen devices),
    apply unchanged to test. AUC is threshold-free. This removes the base-rate/0.5 confound."""
    Xf,Xc,yf,yc=train_test_split(Xtr_all,ytr_all,test_size=0.2,random_state=7,stratify=ytr_all)
    clf=XGBClassifier(n_estimators=150,max_depth=6,learning_rate=0.1,eval_metric="logloss",
                      n_jobs=4,tree_method="hist")
    clf.fit(Xf,yf)
    sc=clf.predict_proba(Xc)[:,1]; st=clf.predict_proba(Xte)[:,1]
    thr=float(np.quantile(sc[yc==0],0.99)) if (yc==0).any() else 0.5   # 1% FPR on seen devices
    pd_=(st>=thr).astype(int); pdc=(sc>=thr).astype(int)
    tn,fp,fn,tp=confusion_matrix(yte,pd_,labels=[0,1]).ravel()
    tnc,fpc,_,_=confusion_matrix(yc,pdc,labels=[0,1]).ravel()
    return {"auc":round(roc_auc_score(yte,st),4) if len(set(yte))>1 else None,
            "recall_at_thr":round(recall_score(yte,pd_,zero_division=0),4),
            "fpr_at_thr":round(fp/(fp+tn),4) if (fp+tn) else None,
            "fpr_calib_seen":round(fpc/(fpc+tnc),4) if (fpc+tnc) else None,  # sanity ~0.01
            "precision_at_thr":round(precision_score(yte,pd_,zero_division=0),4),
            "n_train":int(len(ytr_all)),"n_test":int(len(yte)),
            "test_attack_frac":round(float(yte.mean()),3),"train_attack_frac":round(float(ytr_all.mean()),3),
            "clf":clf,"thr":thr}

def strip(d): return {k:v for k,v in d.items() if k not in ("clf",)}

# ---- RANDOM reference ----
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
rand=strip(fit_eval(Xtr,ytr,Xte,yte))
print(f"\n[RANDOM] {rand}")

# ---- DEVICE-DISJOINT over ALL valid test-victim pairs ----
def valid(testset):
    for c in ["greeth","greip","udpplain"]:
        cv={v for i in range(len(data)) if cls[i]==c for v in vic[i]}
        if not (cv & testset) or not (cv - testset):   # need >=1 victim each side
            return False
    return True

print("\n[DEVICE-DISJOINT] sweep over valid 2-victim test partitions (threshold = 1% FPR on seen devices):")
rows=[]
for testset in map(frozenset, itertools.combinations(sorted(VICTIMS),2)):
    if not valid(testset): continue
    side_test=np.array([bool(vic[i] & testset) for i in range(len(data))])
    full=fit_eval(X[~side_test],y[~side_test],X[side_test],y[side_test])
    clf=full["clf"]; thr=full["thr"]
    st=clf.predict_proba(X[side_test])[:,1]; pd_te=(st>=thr).astype(int)
    yte=y[side_test]; cte=cls[side_test]
    per={}
    for c in ["greeth","greip","udpplain"]:
        m=(cte==c)
        if m.sum(): per[c]=round(recall_score(yte[m],pd_te[m],zero_division=0),4)
    r=strip(full); r["test_victims"]=sorted(testset); r["per_class_recall_at_thr"]=per
    rows.append(r)
    print(f"  test={sorted(testset)}  AUC={r['auc']}  recall={r['recall_at_thr']}  FPR={r['fpr_at_thr']} "
          f"(seen-FPR {r['fpr_calib_seen']})  test_atk={r['test_attack_frac']}  per_class={per}")

aucs=[r["auc"] for r in rows]; fprs=[r["fpr_at_thr"] for r in rows]
print(f"\nSUMMARY across {len(rows)} partitions (threshold calibrated to 1% FPR on SEEN devices):")
print(f"  RANDOM   : AUC {rand['auc']}  test-FPR {rand['fpr_at_thr']}  (seen-FPR {rand['fpr_calib_seen']})")
print(f"  DISJOINT : AUC {min(aucs)}-{max(aucs)}  test-FPR {min(fprs)}-{max(fprs)}")
below_chance=sum(1 for a in aucs if a<=0.55)
print(f"  partitions with AUC <= 0.55 (ranking at/below chance on unseen devices): {below_chance}/{len(rows)}")

json.dump({"features":FEATS,"note":"threshold set to 1% FPR on held-out SEEN(training) devices, applied to unseen",
           "random":rand,"disjoint_partitions":rows,
           "disjoint_auc_range":[min(aucs),max(aucs)],"disjoint_test_fpr_range":[min(fprs),max(fprs)]},
          open(os.path.join(HERE,"mirai_disjoint_result.json"),"w"),indent=2)
print(f"\nWrote mirai_disjoint_result.json")
