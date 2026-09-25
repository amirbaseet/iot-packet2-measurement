#!/usr/bin/env python3
"""
CIC IIoT 2025 — the well-powered device-disjoint test (38 devices, the diversity CICIoT2023 lacked).
Does attack detection on the windowed features transfer to UNSEEN devices, or fingerprint them?
NOTE: windowed per-device aggregate features (NOT packet-2) — complements the packet-2 map, doesn't replace it.

Features = behavioural aggregates ONLY. Every identity column dropped: device_name/device_mac, and any
network_ips*/network_macs*/network_ports*/network_protocols* (names/cardinalities), labels, timestamps.
Task = binary attack(label1) vs benign. Diagnostic = RANDOM split vs DEVICE-DISJOINT (hold out whole
devices), multi-seed. Threshold = 1% FPR on seen devices. Deterministic. Reads LOCAL zip.
"""
import zipfile, io, json, os, statistics as stt
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, confusion_matrix, roc_auc_score

ZIP=os.environ.get("IIOT2025_ZIP",
    "./data/iiot/iiot2025.zip")
HERE=os.path.dirname(os.path.abspath(__file__))
zf=zipfile.ZipFile(ZIP); member=[m for m in zf.namelist() if m.endswith(".csv")][0]

# header -> pick behavioural feature columns (drop identity)
with zf.open(member) as fh:
    header=io.TextIOWrapper(fh,encoding="utf-8",errors="replace").readline().rstrip("\n").split(",")
DROP_PREFIX=("label","timestamp","network_ips","network_macs","network_ports","network_protocols","log_data-types")
ID={"device_name","device_mac"}
feat_cols=[c for c in header if c not in ID and not c.startswith(DROP_PREFIX)]
usecols=feat_cols+["device_mac","label1"]
print(f"{len(feat_cols)} behavioural feature columns kept (identity dropped).")

with zf.open(member) as fh:
    df=pd.read_csv(fh,usecols=usecols,low_memory=False)
for c in feat_cols:
    df[c]=pd.to_numeric(df[c],errors="coerce")
df=df.fillna(0.0)
X=df[feat_cols].to_numpy(np.float32)
y=(df["label1"].astype(str).str.strip()=="attack").to_numpy().astype(int)
dev=df["device_mac"].astype(str).str.strip().str.lower().to_numpy()
devices=sorted(set(dev))
print(f"rows={len(df):,}  attack={int(y.sum()):,} benign={int((y==0).sum()):,}  devices={len(devices)}")

# devices usable as held-out test = have BOTH attack and benign windows
dev_has_both=[d for d in devices if y[dev==d].sum()>0 and (y[dev==d]==0).sum()>0]
print(f"devices with both attack+benign windows (valid to hold out): {len(dev_has_both)}/{len(devices)}")

def model(): return XGBClassifier(n_estimators=200,max_depth=6,learning_rate=0.1,eval_metric="logloss",
                                  random_state=42,n_jobs=1,tree_method="hist")
def evalu(Xtr,ytr,Xte,yte):
    Xf,Xc,yf,yc=train_test_split(Xtr,ytr,test_size=0.2,random_state=7,stratify=ytr)
    clf=model(); clf.fit(Xf,yf); sc=clf.predict_proba(Xc)[:,1]; st=clf.predict_proba(Xte)[:,1]
    thr=float(np.quantile(sc[yc==0],0.99)) if (yc==0).any() else 0.5
    pd_=(st>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pd_,labels=[0,1]).ravel()
    return {"auc":round(roc_auc_score(yte,st),4) if len(set(yte))>1 else None,
            "recall":round(recall_score(yte,pd_,zero_division=0),4),
            "fpr":round(fp/(fp+tn),4) if (fp+tn) else None}

# RANDOM split (devices shared)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
rnd=evalu(Xtr,ytr,Xte,yte)
print(f"\n[RANDOM split, devices shared]      {rnd}")

# DEVICE-DISJOINT: hold out ~30% of dual-label devices, all their rows -> test; multi-seed
aucs=[];fprs=[];recs=[]
for seed in [1,7,42,100,1729]:
    rng=np.random.default_rng(seed); perm=list(dev_has_both); rng.shuffle(perm)
    nhold=max(1,int(round(len(perm)*0.3))); held=set(perm[:nhold])
    te=np.isin(dev,list(held))
    if te.sum()==0 or (~te).sum()==0: continue
    r=evalu(X[~te],y[~te],X[te],y[te]); aucs.append(r["auc"]); fprs.append(r["fpr"]); recs.append(r["recall"])
    print(f"[DEVICE-DISJOINT seed={seed:4d}] held {nhold} devices  {r}")
print(f"\n[DEVICE-DISJOINT] AUC {min(aucs):.3f}-{max(aucs):.3f} (median {stt.median(aucs):.3f})  "
      f"FPR {min(fprs):.3f}-{max(fprs):.3f} (median {stt.median(fprs):.3f})")
print(f"\nCONTRAST — RANDOM AUC {rnd['auc']} / FPR {rnd['fpr']}  vs  DEVICE-DISJOINT median AUC {stt.median(aucs):.3f} / FPR {stt.median(fprs):.3f}")
print("READ: AUC holds + FPR stays low on unseen devices => detection GENERALIZES (rich features + 38 devices);")
print("      AUC drops / FPR explodes => still fingerprints devices even here.")
json.dump({"note":"CIC IIoT 2025 windowed features (NOT packet-2); behavioural only; binary attack/benign; thr=1%FPR seen; seed-swept",
           "n_features":len(feat_cols),"n_rows":int(len(df)),"n_devices":len(devices),"n_devices_dual_label":len(dev_has_both),
           "random":rnd,"disjoint_auc":[round(min(aucs),4),round(float(stt.median(aucs)),4),round(max(aucs),4)],
           "disjoint_fpr":[round(min(fprs),4),round(float(stt.median(fprs)),4),round(max(fprs),4)],
           "disjoint_recall_median":round(float(stt.median(recs)),4)},
          open(os.path.join(HERE,"iiot2025_disjoint.json"),"w"),indent=2)
print("Wrote iiot2025_disjoint.json")
