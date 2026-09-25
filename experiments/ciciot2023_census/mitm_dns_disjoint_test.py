#!/usr/bin/env python3
"""
CICIoT2023 MITM-ArpSpoofing + DNS_Spoofing device-disjoint packet-2 test — completes the
CICIoT2023 family map (Mirai and Recon were already done; these two were the gap).

Method is IDENTICAL to mirai_disjoint_test.py (the reviewed template):
  - model input = behavioral packet-1/packet-2 features ONLY (no MAC/IP as features);
  - device identity is used ONLY to build the split;
  - threshold pinned to 1% FPR on a held-out slice of SEEN (training) devices, applied unchanged;
  - RANDOM split is reported alongside DEVICE-DISJOINT as the diagnostic contrast.

Difference vs Mirai: each class has exactly 3 non-infra victims at the census 5% basis, so the
partition scheme is LEAVE-ONE-VICTIM-OUT (test = 1 held-out victim, train = other 2), swept over
all 3. Victim MAC lists are the census's own victims_5pct_by_class (not improvised here).

Victim attribution (differs from Mirai, deliberately): Mirai floods go TO the victim, so dst-only
attribution is correct there. MITM and DNS_Spoofing are NOT floods — the census victim appears as
SRC about as often as DST (measured), so attack attribution here is BOTH directions,
{eth_src, eth_dst} ∩ victims. A flow attributable to exactly ONE victim is held out in that
victim's fold; a flow touching ZERO victims (≈40-55% of each class — attacks on non-census
devices) or ≥2 victims (ambiguous) stays in TRAINING in every fold and is never in a test set —
so "the held-out victim's flows appear nowhere in training" holds exactly.

Scope label: MITM-ArpSpoofing.csv contains 0 ARP frames (all rows are IP; measured) — the
extractor emitted only the intercepted IP traffic. So this measures detection of the redirected
IP traffic, NOT of the ARP spoof itself. The flow key is not direction-normalized, so packet-1
and packet-2 are the first two packets in the SAME direction of a 5-tuple flow.

Determinism: XGBoost n_jobs=1 + explicit random_state (the Mirai script used n_jobs=4, which is
not bit-reproducible; this is a deliberate, documented tightening, not a method change).
"""
import zipfile, io, json, os
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, roc_auc_score, confusion_matrix

ZIP = "./data/CICIOT23/archive.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
BENIGN = ["BenignTraffic.csv", "BenignTraffic1.csv", "BenignTraffic2.csv", "BenignTraffic3.csv"]
FEATS = ["ip_p","sport","dport","iplen1","iplen2","ttl1","ttl2","win1","win2","flags1","flags2","pay1","pay2","iat"]

# census victims_5pct_by_class (non-infra dst with >=5% class share) — same basis as the Mirai test
CLASSES = {
    "MITM-ArpSpoofing.csv": {"08:7c:39:ce:6e:2a", "56:4f:8a:e1:f3:2d", "94:39:e5:5d:27:a6"},
    "DNS_Spoofing.csv":     {"dc:a6:32:dc:27:d5", "56:4f:8a:e1:f3:2d", "24:05:88:30:6f:89"},
}

def f(x):
    try: return float(x)
    except: return np.nan

def feats(p1, p2):
    t1, t2 = f(p1[0]), f(p2[0])
    return [f(p1[6]),f(p1[10]),f(p1[11]),f(p1[7]),f(p2[7]),f(p1[8]),f(p2[8]),
            f(p1[15]),f(p2[15]),f(p1[14]),f(p2[14]),f(p1[17]),f(p2[17]),
            (t2 - t1) if (np.isfinite(t1) and np.isfinite(t2)) else np.nan]

def collect(member, is_attack, victims):
    """Same flow dedup + packet-2 assembly as mirai_disjoint_test.py."""
    rows = []; first = {}; seen2 = set(); zf = zipfile.ZipFile(ZIP)
    with zf.open(member) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="")
        text.readline()
        for line in text:
            p = line.rstrip("\n").split(",")
            if len(p) <= 17: continue
            es, ed = p[1], p[2]
            if not is_attack and es not in victims and ed not in victims: continue
            key = (p[4], p[5], p[10], p[11], p[6])
            if key in seen2: continue
            if key not in first: first[key] = p; continue
            row = feats(first[key], p); seen2.add(key); del first[key]
            touched = {es, ed} & victims          # both directions, attack and benign alike
            vt = touched if len(touched) == 1 else set()   # 0 or >=2 victims -> training-always
            rows.append((row, 1 if is_attack else 0, frozenset(vt)))
    return rows

def fit_eval(Xtr_all, ytr_all, Xte, yte):
    Xf, Xc, yf, yc = train_test_split(Xtr_all, ytr_all, test_size=0.2, random_state=7, stratify=ytr_all)
    clf = XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, eval_metric="logloss",
                        n_jobs=1, tree_method="hist", random_state=42)
    clf.fit(Xf, yf)
    sc = clf.predict_proba(Xc)[:, 1]; st = clf.predict_proba(Xte)[:, 1]
    thr = float(np.quantile(sc[yc == 0], 0.99)) if (yc == 0).any() else 0.5
    pd_ = (st >= thr).astype(int); pdc = (sc >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pd_, labels=[0, 1]).ravel()
    tnc, fpc, _, _ = confusion_matrix(yc, pdc, labels=[0, 1]).ravel()
    return {"auc": round(roc_auc_score(yte, st), 4) if len(set(yte)) > 1 else None,
            "recall_at_thr": round(recall_score(yte, pd_, zero_division=0), 4),
            "fpr_at_thr": round(fp / (fp + tn), 4) if (fp + tn) else None,
            "fpr_calib_seen": round(fpc / (fpc + tnc), 4) if (fpc + tnc) else None,
            "precision_at_thr": round(precision_score(yte, pd_, zero_division=0), 4),
            "n_train": int(len(ytr_all)), "n_test": int(len(yte)),
            "test_attack_frac": round(float(yte.mean()), 3), "train_attack_frac": round(float(ytr_all.mean()), 3),
            "thr": thr}

def run_class(member, victims):
    victims = sorted(victims)
    print(f"\n==================== {member}  victims={victims} ====================", flush=True)
    data = collect(member, True, set(victims)); print(f"  {member}: {len(data):,} attack flows", flush=True)
    for b in BENIGN:
        r = collect(b, False, set(victims)); print(f"  {b}: {len(r):,} benign (touching victims)", flush=True); data += r
    X = np.array([d[0] for d in data], dtype=float)
    y = np.array([d[1] for d in data])
    vic = [d[2] for d in data]

    # attribution breakdown (post-dedup): attack flows testable per victim vs always-train
    atk_single = {v: sum(1 for i in range(len(data)) if y[i] == 1 and vic[i] == frozenset({v})) for v in victims}
    atk_train_always = sum(1 for i in range(len(data)) if y[i] == 1 and len(vic[i]) == 0)
    print(f"  attribution: attack flows testable per victim {atk_single}  "
          f"| attack flows always-train (0 or >=2 victims) {atk_train_always:,}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    rand = fit_eval(Xtr, ytr, Xte, yte)
    print(f"  [RANDOM]  AUC={rand['auc']}  recall={rand['recall_at_thr']}  FPR={rand['fpr_at_thr']} (seen {rand['fpr_calib_seen']})")

    loo = []
    for held in victims:
        side_test = np.array([held in v for v in vic])
        if not side_test.any() or side_test.all():
            print(f"  [DISJOINT held={held}] SKIP (no testable flows for this victim)"); continue
        ytr_s, yte_s = y[~side_test], y[side_test]
        if len(set(ytr_s)) < 2 or len(set(yte_s)) < 2:
            print(f"  [DISJOINT held={held}] SKIP (a side lacks both classes)"); continue
        r = fit_eval(X[~side_test], ytr_s, X[side_test], yte_s)
        r["held_out_victim"] = held
        # void the fold if the 1%-FPR threshold did not actually calibrate on seen devices
        r["calibrated"] = bool(r["fpr_calib_seen"] is not None and 0.005 <= r["fpr_calib_seen"] <= 0.02)
        loo.append(r)
        flag = "" if r["calibrated"] else "  [VOID: seen-FPR off-target, test-FPR meaningless]"
        print(f"  [DISJOINT held={held}]  AUC={r['auc']}  recall={r['recall_at_thr']}  FPR={r['fpr_at_thr']} "
              f"(seen {r['fpr_calib_seen']})  test_atk={r['test_attack_frac']}  n_test={r['n_test']}{flag}")
    valid = [r for r in loo if r["calibrated"]]
    if len(valid) < 2:
        raise SystemExit(f"ABORT {member}: only {len(valid)} calibrated fold(s) — cannot conclude at n<2.")
    return {"victims": victims, "attack_testable_per_victim": atk_single,
            "attack_always_train": atk_train_always, "random": rand, "leave_one_victim_out": loo}

results = {}
for member, victims in CLASSES.items():
    results[member.replace(".csv", "")] = run_class(member, victims)

out = {"features": FEATS,
       "note": "leave-one-victim-out; threshold = 1% FPR on held-out SEEN(training) devices; "
               "victims = census victims_5pct_by_class; XGBoost n_jobs=1 random_state=42",
       "n_victims_per_class": 3,
       "results": results}
json.dump(out, open(os.path.join(HERE, "mitm_dns_disjoint_result.json"), "w"), indent=2)

print("\n==================== SUMMARY (n=3 victims; folds reported individually, NOT averaged) ====================")
for name, r in results.items():
    ra = r["random"]
    print(f"\n{name}   RANDOM: AUC {ra['auc']}  test-FPR {ra['fpr_at_thr']} (seen {ra['fpr_calib_seen']})")
    for p in r["leave_one_victim_out"]:
        tag = "" if p["calibrated"] else "  [VOID]"
        print(f"    held={p['held_out_victim']}  AUC {p['auc']}  recall {p['recall_at_thr']}  "
              f"test-FPR {p['fpr_at_thr']} (seen {p['fpr_calib_seen']})  n_test {p['n_test']}{tag}")
    valid = [p for p in r["leave_one_victim_out"] if p["calibrated"]]
    directions = {("hold" if (p["auc"] or 0) >= 0.9 and (p["fpr_at_thr"] or 1) <= 0.05 else "collapse") for p in valid}
    verdict = ("consistent: " + directions.pop()) if len(directions) == 1 else "INDETERMINATE (folds disagree)"
    print(f"    -> across {len(valid)} calibrated folds: {verdict}")
print(f"\nWrote {os.path.join(HERE, 'mitm_dns_disjoint_result.json')}")
