#!/usr/bin/env python3
"""
Earliness vs fingerprint: run the SAME device-disjoint diagnostic at packet-2, packet-20, full-flow.
Same flows, same victim splits, three feature depths. If generalization to UNSEEN devices RISES with
more packets -> earliness problem. If it stays broken at every budget -> fingerprinting is fundamental.

pkt-2  = first-2-packet behavioral features (14).
pkt-20 = aggregate stats over first <=20 packets (22).
full   = aggregate stats over all packets (22).
No MAC/IP as features; device identity only builds the split. Threshold = 1% FPR on SEEN devices.
"""
import zipfile, io, itertools, json, os, math
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix

ZIP="./data/CICIOT23/archive.zip"; HERE=os.path.dirname(os.path.abspath(__file__))
MIRAI={"Mirai-greeth_flood.csv":"greeth","Mirai-greip_flood.csv":"greip","Mirai-udpplain.csv":"udpplain"}
BENIGN=["BenignTraffic.csv","BenignTraffic1.csv","BenignTraffic2.csv","BenignTraffic3.csv"]
VICTIMS={"08:7c:39:ce:6e:2a","1c:12:b0:9b:0c:ec","1c:fe:2b:98:16:dd","9c:8e:cd:1d:ab:9f","cc:f4:11:9c:d0:00"}

def f(x):
    try:
        v=float(x); return v if v==v else 0.0
    except: return 0.0

def new_acc(): return dict(n=0,t0=None,tl=None,ls=0.0,lss=0.0,lmn=1e18,lmx=-1e18,ts=0.0,tss=0.0,
                           ws=0.0,wss=0.0,ps=0.0,pt=0.0,is_=0.0,iss=0.0,imn=1e18,imx=-1e18,ni=0,fs=0.0)
def upd(a,t,ln,ttl,win,pay,flg):
    if a['t0'] is None: a['t0']=t
    if a['tl'] is not None:
        iat=t-a['tl']
        if iat>=0: a['is_']+=iat;a['iss']+=iat*iat;a['ni']+=1;a['imn']=min(a['imn'],iat);a['imx']=max(a['imx'],iat)
    a['tl']=t; a['n']+=1
    a['ls']+=ln;a['lss']+=ln*ln;a['lmn']=min(a['lmn'],ln);a['lmx']=max(a['lmx'],ln)
    a['ts']+=ttl;a['tss']+=ttl*ttl;a['ws']+=win;a['wss']+=win*win;a['ps']+=pay;a['pt']+=pay;a['fs']+=flg
def fin(a,proto,sp,dp):
    n=a['n'] or 1
    dur=(a['tl']-a['t0']) if a['t0'] is not None else 0.0
    def sd(s,ss,k):
        v=ss/k-(s/k)**2; return math.sqrt(v) if v>0 else 0.0
    ni=a['ni'] or 1
    return [a['n'],a['ls']/n,sd(a['ls'],a['lss'],n),a['lmn'],a['lmx'],a['ts']/n,sd(a['ts'],a['tss'],n),
            a['ws']/n,sd(a['ws'],a['wss'],n),a['ps']/n,a['pt'],
            (a['is_']/ni) if a['ni'] else 0.0, sd(a['is_'],a['iss'],ni) if a['ni'] else 0.0,
            a['imn'] if a['ni'] else 0.0, a['imx'] if a['ni'] else 0.0, dur, a['fs']/n,
            (a['pt']/dur) if dur>0 else 0.0,(n/dur) if dur>0 else 0.0, proto,sp,dp]

def collect(member,is_attack):
    F={}; zf=zipfile.ZipFile(ZIP)
    with zf.open(member) as fh:
        text=io.TextIOWrapper(fh,encoding="utf-8",errors="replace",newline="")
        text.readline()
        for line in text:
            p=line.rstrip("\n").split(",")
            if len(p)<=17: continue
            es,ed=p[1],p[2]
            if not is_attack and es not in VICTIMS and ed not in VICTIMS: continue
            key=(p[4],p[5],p[10],p[11],p[6])
            t,ln,ttl,win,pay,flg=f(p[0]),f(p[7]),f(p[8]),f(p[15]),f(p[17]),f(p[14])
            proto,sp,dp=f(p[6]),f(p[10]),f(p[11])
            if key not in F:
                F[key]=dict(a20=new_acc(),af=new_acc(),rows=[],ed=ed,es=es,proto=proto,sp=sp,dp=dp)
            fl=F[key]
            if len(fl['rows'])<2: fl['rows'].append((t,ln,ttl,win,pay,flg,proto,sp,dp))
            if fl['a20']['n']<20: upd(fl['a20'],t,ln,ttl,win,pay,flg)
            upd(fl['af'],t,ln,ttl,win,pay,flg)
    out=[]
    for key,fl in F.items():
        if len(fl['rows'])<2: continue          # need packet 2 to exist (matches the gate)
        r1,r2=fl['rows'][0],fl['rows'][1]
        feat2=[r1[6],r1[7],r1[8],r1[1],r2[1],r1[2],r2[2],r1[3],r2[3],r1[5],r2[5],r1[4],r2[4],
               (r2[0]-r1[0]) if (r2[0]>=r1[0]) else 0.0]
        feat20=fin(fl['a20'],fl['proto'],fl['sp'],fl['dp'])
        featf =fin(fl['af'], fl['proto'],fl['sp'],fl['dp'])
        vt=({fl['ed']}&VICTIMS) if is_attack else ({fl['es'],fl['ed']}&VICTIMS)
        out.append((feat2,feat20,featf,1 if is_attack else 0,MIRAI.get(member,"benign"),frozenset(vt),fl['af']['n']))
    return out

print("Extracting flows (3 depths) once...",flush=True)
data=[]
for m in MIRAI:  r=collect(m,True);  print(f"  {m}: {len(r):,}",flush=True); data+=r
for m in BENIGN: r=collect(m,False); print(f"  {m}: {len(r):,} benign",flush=True); data+=r
X2=np.array([d[0] for d in data],float); X20=np.array([d[1] for d in data],float); XF=np.array([d[2] for d in data],float)
y=np.array([d[3] for d in data]); cls=np.array([d[4] for d in data]); vic=[d[5] for d in data]
nfull=np.array([d[6] for d in data])
for c in ["greeth","greip","udpplain"]:
    m=(cls==c); print(f"  {c}: {m.sum()} flows, median len {int(np.median(nfull[m]))}, %>=20pkts {100*np.mean(nfull[m]>=20):.0f}%")

def sweep(X,tag):
    def fe(Xtr,ytr,Xte,yte):
        Xf,Xc,yf,yc=train_test_split(Xtr,ytr,test_size=0.2,random_state=7,stratify=ytr)
        clf=XGBClassifier(n_estimators=150,max_depth=6,learning_rate=0.1,eval_metric="logloss",n_jobs=4,tree_method="hist")
        clf.fit(Xf,yf); sc=clf.predict_proba(Xc)[:,1]; st=clf.predict_proba(Xte)[:,1]
        thr=float(np.quantile(sc[yc==0],0.99)) if (yc==0).any() else 0.5
        pd_=(st>=thr).astype(int); tn,fp,fn,tp=confusion_matrix(yte,pd_,labels=[0,1]).ravel()
        return (roc_auc_score(yte,st) if len(set(yte))>1 else float("nan"),
                fp/(fp+tn) if (fp+tn) else float("nan"), recall_score(yte,pd_,zero_division=0))
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
    ra,rf,rr=fe(Xtr,ytr,Xte,yte)
    def valid(ts):
        for c in ["greeth","greip","udpplain"]:
            cv={v for i in range(len(data)) if cls[i]==c for v in vic[i]}
            if not (cv&ts) or not (cv-ts): return False
        return True
    A=[];Fp=[]
    for ts in map(frozenset,itertools.combinations(sorted(VICTIMS),2)):
        if not valid(ts): continue
        s=np.array([bool(vic[i]&ts) for i in range(len(data))])
        a,fpr,_=fe(X[~s],y[~s],X[s],y[s]); A.append(a);Fp.append(fpr)
    print(f"\n[{tag}]  RANDOM: AUC {ra:.3f} FPR {rf:.3f}   |  DISJOINT: AUC {min(A):.3f}-{max(A):.3f} (median {np.median(A):.3f}) FPR {min(Fp):.3f}-{max(Fp):.3f} (median {np.median(Fp):.3f})  atchance<=0.55: {sum(1 for a in A if a<=0.55)}/{len(A)}")
    return dict(tag=tag,random_auc=round(ra,4),random_fpr=round(rf,4),
                disjoint_auc=[round(min(A),4),round(float(np.median(A)),4),round(max(A),4)],
                disjoint_fpr=[round(min(Fp),4),round(float(np.median(Fp)),4),round(max(Fp),4)],
                at_chance=sum(1 for a in A if a<=0.55),n_partitions=len(A))

res=[sweep(X2,"packet-2"),sweep(X20,"packet-20"),sweep(XF,"full-flow")]
json.dump(res,open(os.path.join(HERE,"earliness_result.json"),"w"),indent=2)
print("\n=== EARLINESS READ ===")
print("If DISJOINT AUC rises 2->20->full and FPR falls -> earliness (navigable).")
print("If DISJOINT stays broken at every depth -> fingerprinting is fundamental to flow-stats.")
print(f"\nWrote earliness_result.json")
