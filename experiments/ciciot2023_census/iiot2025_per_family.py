#!/usr/bin/env python3
"""
Verify the CIC IIoT 2025 device-disjoint generalization result (AUC 0.98) PER ATTACK FAMILY.
Is it real across DDoS/DoS/Recon/MITM/Web/Malware/BruteForce, or just the easy whole-network recon
(38 devices, near-identical across the fleet) carrying the aggregate?

Per family F: positives = F's attack windows, negatives = benign windows. RANDOM split (devices shared)
vs DEVICE-DISJOINT (hold out whole devices), multi-seed. Behavioural features only (identity dropped).
Threshold = 1% FPR on seen devices. Reports device count per family (thin floods vs whole-network recon).
"""
import zipfile, io, json, os, statistics as stt
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score

ZIP=os.environ.get("IIOT2025_ZIP",
    "./data/iiot/iiot2025.zip")
HERE=os.path.dirname(os.path.abspath(__file__))
zf=zipfile.ZipFile(ZIP); member=[m for m in zf.namelist() if m.endswith(".csv")][0]
with zf.open(member) as fh:
    header=io.TextIOWrapper(fh,encoding="utf-8",errors="replace").readline().rstrip("\n").split(",")
DROP=("label","timestamp","network_ips","network_macs","network_ports","network_protocols","log_data-types")
ID={"device_name","device_mac"}
feat_cols=[c for c in header if c not in ID and not c.startswith(DROP)]
with zf.open(member) as fh:
    df=pd.read_csv(fh,usecols=feat_cols+["device_mac","label_full","label1"],low_memory=False)
for c in feat_cols: df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.fillna(0.0)

def family(lf,l1):
    l1=str(l1).strip()
    if l1=="benign": return "benign"
    t=str(lf).split("_")
    return t[1] if len(t)>1 and t[0]=="attack" else "other"
df["fam"]=[family(a,b) for a,b in zip(df["label_full"],df["label1"])]
X=df[feat_cols].to_numpy(np.float32); dev=df["device_mac"].astype(str).str.strip().str.lower().to_numpy(); fam=df["fam"].to_numpy()
benign_mask=fam=="benign"
fams=[f for f in ["ddos","dos","recon","mitm","malware","bruteforce","web"] if (fam==f).any()]
print(f"rows={len(df):,}  benign={int(benign_mask.sum()):,}  families={fams}")

def model(): return XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,eval_metric="logloss",
                                  random_state=42,n_jobs=1,tree_method="hist")
def ev(Xtr,ytr,Xte,yte):
    Xf,Xc,yf,yc=train_test_split(Xtr,ytr,test_size=0.2,random_state=7,stratify=ytr)
    clf=model(); clf.fit(Xf,yf); sc=clf.predict_proba(Xc)[:,1]; st=clf.predict_proba(Xte)[:,1]
    thr=float(np.quantile(sc[yc==0],0.99)) if (yc==0).any() else 0.5
    pd_=(st>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pd_,labels=[0,1]).ravel()
    return (roc_auc_score(yte,st) if len(set(yte))>1 else float('nan'),
            fp/(fp+tn) if (fp+tn) else float('nan'))

out={}
print(f"\n{'family':12s} {'atk_dev':>7s} {'rows':>8s} {'RANDOM auc/fpr':>16s} {'DISJOINT auc(med)/fpr':>22s}")
for F in fams:
    fmask=fam==F
    atk_devs=sorted(set(dev[fmask])); dual=[d for d in atk_devs if (fmask&(dev==d)).sum()>0 and (benign_mask&(dev==d)).sum()>0]
    idx=np.where(fmask|benign_mask)[0]
    Xf_,yf_,devf=X[idx],fmask[idx].astype(int),dev[idx]
    # RANDOM
    Xtr,Xte,ytr,yte=train_test_split(Xf_,yf_,test_size=0.3,random_state=42,stratify=yf_)
    ra,rf=ev(Xtr,ytr,Xte,yte)
    # DEVICE-DISJOINT over seeds (hold out ~30% of dual-label attack devices)
    aucs=[];fprs=[]
    if len(dual)>=2:
        for seed in [1,7,42,100,1729]:
            rng=np.random.default_rng(seed); perm=list(dual); rng.shuffle(perm)
            held=set(perm[:max(1,int(round(len(perm)*0.3)))])
            te=np.isin(devf,list(held))
            if te.sum()==0 or (~te).sum()==0 or yf_[~te].sum()==0 or (yf_[~te]==0).sum()==0: continue
            if yf_[te].sum()==0: continue
            a,fp=ev(Xf_[~te],yf_[~te],Xf_[te],yf_[te]); aucs.append(a); fprs.append(fp)
    da=round(stt.median(aucs),3) if aucs else None; dfp=round(stt.median(fprs),3) if fprs else None
    out[F]={"attack_devices":len(atk_devs),"dual_label_devices":len(dual),"n_attack_rows":int(fmask.sum()),
            "random_auc":round(ra,3),"random_fpr":round(rf,3),
            "disjoint_auc_median":da,"disjoint_fpr_median":dfp,
            "disjoint_auc_range":[round(min(aucs),3),round(max(aucs),3)] if aucs else None,"n_seeds":len(aucs)}
    print(f"{F:12s} {len(atk_devs):7d} {int(fmask.sum()):8d}   {ra:.3f}/{rf:.3f}      "
          f"{('%.3f'%da) if da is not None else '  n/a':>7}/{('%.3f'%dfp) if dfp is not None else 'n/a':<6} "
          f"{'(devices<2: cannot hold out)' if len(dual)<2 else ''}")

json.dump(out,open(os.path.join(HERE,"iiot2025_per_family.json"),"w"),indent=2)
gen=[F for F,r in out.items() if r["disjoint_auc_median"] is not None and r["disjoint_auc_median"]>=0.85 and (r["disjoint_fpr_median"] or 1)<=0.10]
fail=[F for F,r in out.items() if r["disjoint_auc_median"] is not None and (r["disjoint_auc_median"]<0.75 or (r["disjoint_fpr_median"] or 0)>0.20)]
print(f"\nGENERALIZE (disjoint AUC>=0.85 & FPR<=10%): {gen}")
print(f"FINGERPRINT/FAIL (AUC<0.75 or FPR>20%): {fail}")
print("READ: if only recon generalizes and floods/web fail/can't-split -> the 0.98 aggregate was carried by recon.")
print("Wrote iiot2025_per_family.json")
