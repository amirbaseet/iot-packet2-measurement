#!/usr/bin/env python3
"""
CLEAN CIC IIoT 2025 MITM packet-2 device-disjoint + training-victim sweep (reviewer §3.2, and a fix for a
DEFECT in the paper's cited 0.53/91% number).

The paper's MITM row came from iiot2025_family_pkt2.py FAM=mitm, which had two problems:
  (1) it grouped the 60 pcaps by filename name-token -> 28 mqtt-broker + 18 router + 14 sensors = 16 lopsided
      groups, and
  (2) `vic_mac[name]=vmac` was last-write-wins, so holding out "mqtt-broker" removed all 28 attack pcaps by
      name but only the LAST pcap's single dominant eth.dst from benign -> an ASYMMETRIC holdout that inflates
      the 91% FPR.
This script fixes both: every pcap keeps its OWN dominant eth.dst (no overwrite); the split holds out a set of
victim MACs and removes, symmetrically, every attack flow whose victim MAC is held AND every benign flow
touching any held MAC. An assertion enforces the symmetry. It prints the MAC-grouping histogram first
(advisor gate: if dominant-dst collapses to 2 giant broker/router groups, set GROUP_BY=sensor to group by the
`--<sensor>` endpoint instead). Extraction is cached so regrouping/sweeping needs no re-tshark.

SCOPE (carried into the paper's Limitations regardless of the number): attack_type_map.json holds that MITM's
honest frame is ATTACKER-disjoint, not victim-disjoint; this is a victim-disjoint test like the paper's other
families, reported as such. XGBoost 3.3.0 hist, seed 42, n_jobs 1.
"""
import subprocess, os, glob, collections, json, statistics as stt, re
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

EX = "./data/iiot/extract"
PCAPS = sorted(p for p in glob.glob(EX + "/mitm/*.pcap") if "/._" not in p)
BENIGN = EX + "/benign/benign_whole-network3.pcap"
HERE = os.path.dirname(os.path.abspath(__file__))
SC = "./cache/"
CACHE = os.environ.get("MITM_CACHE", SC + "iiot_mitm_clean_cache.json")
UDP_CACHE = os.environ.get("UDP_CACHE", SC + "iiot_udpfrag_ablation_cache.json")
# "sensor_realmac" (default, CORRECT): group by the --<sensor> victim device, and hold out its benign by the
#   device's REAL MAC taken from the udp-frag capture (where the flood targets the device, so dst IS the victim).
#   The MITM capture's dominant eth.dst is the ATTACKER, so "mac" grouping is degenerate (2 groups) — kept only
#   to document that. "sensor" groups attack by token but has no clean benign MAC, so it is not the honest test.
GROUP_BY = os.environ.get("GROUP_BY", "sensor_realmac")
FIELDS = ["frame.time_epoch","eth.src","eth.dst","ip.src","ip.dst","ip.proto","ip.len","ip.ttl",
          "tcp.srcport","tcp.dstport","udp.srcport","udp.dstport","tcp.flags","tcp.window_size","frame.len"]

def tshark(pcap, cap=None):
    cmd = ["tshark","-r",pcap] + (["-c",cap] if cap else []) + ["-T","fields"] + sum([["-e",f] for f in FIELDS], [])
    p = subprocess.run(cmd, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        yield line.split("\t")

def fnum(x):
    try:
        v = float(x); return v if v == v else 0.0
    except: return 0.0

def feats(c1, c, ts):
    proto, iplen, ttl, flags, win, flen = c[5], c[6], c[7], c[12], c[13], c[14]
    sport = c[8] or c[10]; dport = c[9] or c[11]
    t1, t2 = fnum(c1[0]), fnum(ts)
    return [fnum(proto),fnum(sport),fnum(dport),fnum(c1[6]),fnum(iplen),fnum(c1[7]),fnum(ttl),
            fnum(c1[13]),fnum(win),fnum(c1[12]),fnum(flags),fnum(c1[14]),fnum(flen),(t2-t1) if t2>=t1 else 0.0]

def sensor_token(basename):
    n = re.sub(r"\.pcap$", "", basename)
    return n.split("--")[-1] if "--" in n else re.sub(r".*_", "", n)

def flows_from(pcap):
    first = {}; done = set(); rows = []; dstmac = collections.Counter()
    for c in tshark(pcap):
        if len(c) < 15: continue
        ips, ipd, proto = c[3], c[4], c[5]
        sport = c[8] or c[10]; dport = c[9] or c[11]
        key = (ips, ipd, sport, dport, proto)
        if key in done: continue
        dstmac[c[2]] += 1
        if key not in first: first[key] = c; continue
        rows.append(feats(first[key], c, c[0])); done.add(key); del first[key]
    vic = dstmac.most_common(1)[0][0] if dstmac else None
    return rows, vic

def extract():
    print(f"mitm pcaps: {len(PCAPS)}", flush=True)
    atk = []          # (feat, victim_mac, sensor_token) per pcap group
    hist = collections.Counter()
    for p in PCAPS:
        base = os.path.basename(p); sens = sensor_token(base)
        rows, vmac = flows_from(p); hist[vmac] += 1
        for r in rows: atk.append((r, vmac, sens))
        print(f"  {base[:52]:52s} flows={len(rows):>5} dst_mac={vmac} sensor={sens}", flush=True)
    macset = {v for (_, v, _) in atk if v}
    ben = []; first = {}; done = set()
    for c in tshark(BENIGN):
        if len(c) < 15: continue
        ips, ipd, proto = c[3], c[4], c[5]
        sport = c[8] or c[10]; dport = c[9] or c[11]
        key = (ips, ipd, sport, dport, proto)
        if key in done: continue
        if key not in first: first[key] = (c, {c[1], c[2]}); continue
        c1, m1 = first[key]
        touch = sorted((m1 | {c[1], c[2]}) & macset)
        ben.append((feats(c1, c, c[0]), touch)); done.add(key); del first[key]
    data = {"atk": atk, "ben": ben, "hist": dict(hist)}
    json.dump(data, open(CACHE, "w"))
    print(f"cached -> {CACHE}", flush=True)
    return data

def load():
    if os.path.exists(CACHE):
        print(f"loading cache <- {CACHE}", flush=True); return json.load(open(CACHE))
    return extract()

def model(): return XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.1, eval_metric="logloss",
                                  random_state=42, n_jobs=1, tree_method="hist")

def ev(Xtr, ytr, Xte, yte):
    Xf, Xc, yf, yc = train_test_split(Xtr, ytr, test_size=0.2, random_state=7, stratify=ytr)
    clf = model(); clf.fit(Xf, yf); sc = clf.predict_proba(Xc)[:, 1]; st = clf.predict_proba(Xte)[:, 1]
    thr = float(np.quantile(sc[yc == 0], 0.99)) if (yc == 0).any() else 0.5
    seen_fpr = float((sc[yc == 0] >= thr).mean()) if (yc == 0).any() else None
    pd_ = (st >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(yte, pd_, labels=[0, 1]).ravel()
    return (roc_auc_score(yte, st) if len(set(yte)) > 1 else float("nan"),
            fp / (fp + tn) if (fp + tn) else float("nan"), seen_fpr)

def run():
    d = load()
    atk = d["atk"]; ben = d["ben"]; hist = d["hist"]
    # sensor -> REAL device MAC, learned from the udp-frag capture (dst = victim device there)
    real_mac = {}
    if os.path.exists(UDP_CACHE):
        real_mac = {k.lower(): v for k, v in json.load(open(UDP_CACHE))["vic_mac"].items() if v}
    # grouping key per attack flow + the MAC(s) that key maps to for the SYMMETRIC benign holdout
    def gkey(vmac, sens):
        return {"mac": vmac, "sensor": sens, "sensor_realmac": sens}[GROUP_BY]
    group_macs = collections.defaultdict(set)
    skipped = set()
    for (_, vmac, sens) in atk:
        g = gkey(vmac, sens)
        if not g: continue
        if GROUP_BY == "sensor_realmac":
            rm = real_mac.get(sens.lower())
            if not rm: skipped.add(sens); continue      # no real device MAC -> no clean benign holdout
            group_macs[g].add(rm)                        # benign side uses the device's OWN mac, not the attacker's
        elif vmac:
            group_macs[g].add(vmac)
    groups = sorted(g for g in group_macs if g)
    if skipped: print(f"skipped {len(skipped)} sensors with no udp-frag MAC: {sorted(skipped)}", flush=True)
    print(f"\n=== MAC-grouping histogram (dominant eth.dst per pcap) ===", flush=True)
    for m, n in collections.Counter(hist).most_common():
        print(f"  {m}: {n} pcaps", flush=True)
    print(f"\nGROUP_BY={GROUP_BY}: {len(groups)} groups; pcaps/group balance from histogram above.", flush=True)

    Xa = np.array([r for (r, _, _) in atk], float)
    ga = np.array([gkey(v, s) for (_, v, s) in atk])
    # Benign side: for sensor_realmac the MITM cache's benign touch was recorded against ATTACKER MACs (useless
    # for a real-device holdout). Reuse the udp-frag cache's benign — SAME benign_whole-network3 capture, SAME
    # 14-feature extraction, but its ben_touch is against the REAL device MACs. So the holdout can match devices.
    if GROUP_BY == "sensor_realmac" and os.path.exists(UDP_CACHE):
        ud = json.load(open(UDP_CACHE))
        Xb = np.array(ud["ben_rows"], float); btouch = [set(t) for t in ud["ben_touch"]]
        print(f"benign from udp-frag cache (real-MAC touch): {len(Xb)} flows", flush=True)
    else:
        Xb = np.array([r for (r, _) in ben], float); btouch = [set(t) for (_, t) in ben]

    def split_eval(test_groups, train_groups):
        held_macs = set().union(*[group_macs[g] for g in test_groups]) if test_groups else set()
        train_macs = set().union(*[group_macs[g] for g in train_groups]) if train_groups else set()
        ta = np.isin(ga, list(test_groups)); tr = np.isin(ga, list(train_groups))
        te_b = np.array([bool(t & held_macs) for t in btouch])
        trn_b = np.array([bool(t & train_macs) for t in btouch]) & ~te_b   # never train on benign touching a held MAC
        # symmetry assertion: benign-test MACs are exactly the held groups' MACs
        assert held_macs == set().union(*[group_macs[g] for g in test_groups]) if test_groups else True
        if ta.sum() == 0 or tr.sum() == 0 or te_b.sum() == 0 or trn_b.sum() == 0: return None
        Xtr = np.vstack([Xa[tr], Xb[trn_b]]); ytr = np.concatenate([np.ones(tr.sum()), np.zeros(int(trn_b.sum()))])
        Xte = np.vstack([Xa[ta], Xb[te_b]]);  yte = np.concatenate([np.ones(ta.sum()), np.zeros(int(te_b.sum()))])
        return ev(Xtr, ytr, Xte, yte)

    # ---- clean device-disjoint: hold 30% groups, 5 seeds ----
    print(f"\n=== CLEAN device-disjoint (symmetric holdout, hold ~30% groups) ===", flush=True)
    aucs = []; fprs = []
    for seed in [1, 7, 42, 100, 1729]:
        rng = np.random.default_rng(seed); perm = list(groups); rng.shuffle(perm)
        held = set(perm[:max(1, round(len(perm) * 0.3))]); train = set(groups) - held
        r = split_eval(held, train)
        if r is None: continue
        a, f, _ = r; aucs.append(a); fprs.append(f)
        print(f"  seed={seed:4d} held {len(held)} groups  AUC {a:.3f}  FPR {f:.3f}", flush=True)
    disjoint = {"auc_median": round(stt.median(aucs), 4), "auc_range": [round(min(aucs), 4), round(max(aucs), 4)],
                "fpr_median": round(stt.median(fprs), 4), "fpr_range": [round(min(fprs), 4), round(max(fprs), 4)],
                "n_seeds": len(aucs)} if aucs else None
    # random baseline
    Xall = np.vstack([Xa, Xb]); yall = np.concatenate([np.ones(len(Xa)), np.zeros(len(Xb))])
    Xtr, Xte, ytr, yte = train_test_split(Xall, yall, test_size=0.3, random_state=42, stratify=yall)
    ra, rf, _ = ev(Xtr, ytr, Xte, yte)

    # ---- training-victim sweep (§3.2): fixed held-out test groups, vary # training groups ----
    print(f"\n=== training-group sweep (fixed test groups) ===", flush=True)
    rng0 = np.random.default_rng(2024); perm = list(groups); rng0.shuffle(perm)
    n_held = max(1, round(len(groups) * 0.3)); held_test = set(perm[:n_held]); pool = perm[n_held:]
    sweep = []
    for k in sorted(set([s for s in [3, 5, 8, 12, len(pool)] if 1 <= s <= len(pool)])):
        sa = []; sf = []
        for seed in [1, 7, 42, 100, 1729]:
            rng = np.random.default_rng(seed); pp = list(pool); rng.shuffle(pp)
            r = split_eval(held_test, set(pp[:k]))
            if r is None: continue
            a, f, _ = r; sa.append(a); sf.append(f)
        if sa:
            sweep.append({"train_groups": k, "auc_median": round(stt.median(sa), 4),
                          "auc_range": [round(min(sa), 4), round(max(sa), 4)],
                          "fpr_median": round(stt.median(sf), 4), "n_seeds": len(sa)})
            print(f"  k={k:2d} train groups  AUC median {sweep[-1]['auc_median']} "
                  f"({sweep[-1]['auc_range'][0]}-{sweep[-1]['auc_range'][1]})  FPR {sweep[-1]['fpr_median']}", flush=True)

    out = {"note": "CLEAN CIC IIoT 2025 MITM packet-2 victim-disjoint (fixes the family-map name-token grouping "
                   "+ asymmetric-benign-holdout bug). GROUP_BY=" + GROUP_BY + ". XGB 150/6/0.1 seed42 n_jobs1; "
                   "threshold=1%FPR on seen. Victim axis is dst/device, NOT attacker-disjoint (scope caveat).",
           "group_by": GROUP_BY, "n_groups": len(groups), "n_attack_flows": len(atk), "n_benign_flows": len(ben),
           "grouping_histogram": collections.Counter(hist).most_common(),
           "random": {"auc": round(ra, 4), "fpr": round(rf, 4)},
           "clean_disjoint": disjoint,
           "held_test_groups": sorted(held_test), "training_pool": len(pool), "sweep": sweep}
    json.dump(out, open(os.path.join(HERE, "iiot2025_mitm_clean.json"), "w"), indent=2)
    print(f"\nrandom AUC {ra:.3f}/FPR {rf:.3f}  |  clean disjoint {disjoint}")
    print("Wrote iiot2025_mitm_clean.json")

if __name__ == "__main__":
    run()
