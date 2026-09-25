#!/usr/bin/env python3
"""
Corrected Option-2 test (addresses reconstruction artifact + criterion + worst-vs-best).
1. RE-STREAM the zip and dump the COMPLETE per-class {dst: packets} for every real device
   (no top-5 truncation) -> recompute the true victim pool at >=5%.
2. Split criterion split into two severities:
     - evaluable(class)  = has >=1 TEST victim  (can be evaluated at all)
     - usable(class)     = has >=1 TRAIN and >=1 TEST victim (trained + tested on disjoint devices)
3. Random 30% holdout (worst case, multi-seed) AND a greedy CONSTRUCTED partition
   (upper bound on what the data could support; labelled a methodological smell, not a shipping split).
Victim = dst absorbing >=5% of a class's real-device packets (the definition that gave VIABLE).
"""
import zipfile, io, collections, json, os, random

ZIP = "./data/CICIOT23/archive.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
IDX_SRC, IDX_DST = 1, 2
SHARE = 0.05
SEEDS = [1, 7, 42, 100, 1729]
HOLDOUT_FRAC = 0.30

def to_class(m):
    b = m[:-4] if m.endswith(".csv") else m
    return "Benign" if b.startswith("BenignTraffic") else b

def is_real_device(mac):
    if not mac or ":" not in mac or mac == "ff:ff:ff:ff:ff:ff":
        return False
    try:
        return not (int(mac.split(":")[0], 16) & 1)   # drop multicast/broadcast
    except ValueError:
        return False

# ── 1. full per-class dst dump (real devices) ──
class_dst = collections.defaultdict(collections.Counter)
class_total = collections.Counter()
zf = zipfile.ZipFile(ZIP)
for member in sorted(m for m in zf.namelist() if m.endswith(".csv") and "/" not in m and not m.startswith("__")):
    cls = to_class(member); n = 0
    with zf.open(member) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="")
        text.readline()
        for line in text:
            p = line.split(",", 4)
            if len(p) < 3:
                continue
            d = p[IDX_DST]
            class_total[cls] += 1
            if is_real_device(d):
                class_dst[cls][d] += 1
            n += 1
    print(f"  {member:30s} class={cls:20s} rows={n:,}", flush=True)

classes = sorted(class_total)
# victims at >=5% of the class's TOTAL packets, from FULL counts (no truncation)
per_class_victims = {c: {m: k for m, k in class_dst[c].items() if class_total[c] and k / class_total[c] >= SHARE}
                     for c in classes}
pool = sorted({m for v in per_class_victims.values() for m in v})
nvic = {c: len(per_class_victims[c]) for c in classes}
print(f"\n=== TRUE victim pool (full counts, >={SHARE:.0%}): {len(pool)} distinct devices ===")
for c in classes:
    print(f"  {c:24s} victims>=5%={nvic[c]:2d}   distinct_real_dst_total={len(class_dst[c]):3d}")

def evaluate(train_v, test_v):
    assert set(train_v).isdisjoint(test_v)
    evaluable, usable = [], []
    for c in classes:
        tr = [m for m in per_class_victims[c] if m in train_v]
        te = [m for m in per_class_victims[c] if m in test_v]
        if te:
            evaluable.append(c)
        if tr and te:
            usable.append(c)
    return evaluable, usable

# ── 2. random 30% holdout, multi-seed ──
print(f"\n=== RANDOM 30% holdout (worst case), {len(SEEDS)} seeds ===")
print(f"{'seed':>5s} {'evaluable':>9s} {'usable':>6s}")
rand_eval, rand_use = {}, {}
for s in SEEDS:
    rng = random.Random(s); perm = list(pool); rng.shuffle(perm)
    nt = max(1, round(len(perm) * HOLDOUT_FRAC))
    ev, us = evaluate(set(perm[nt:]), set(perm[:nt]))
    rand_eval[s], rand_use[s] = ev, us
    print(f"{s:5d} {len(ev):9d} {len(us):6d}")

# ── 3. greedy constructed partition (upper bound) ──
# start all train; for each class with >=2 victims and none in test, move its most-shared victim to test.
train_v, test_v = set(pool), set()
share_count = collections.Counter()
for c in classes:
    for m in per_class_victims[c]:
        share_count[m] += 1
for c in sorted(classes, key=lambda c: nvic[c]):   # help thin classes first
    vics = list(per_class_victims[c])
    if len(vics) >= 2 and not any(m in test_v for m in vics):
        move = max(vics, key=lambda m: share_count[m])  # prefer a shared device -> helps many classes
        if len(train_v) > 1:
            train_v.discard(move); test_v.add(move)
g_eval, g_use = evaluate(train_v, test_v)

structurally_dead = [c for c in classes if nvic[c] <= 1]
print(f"\n=== GREEDY constructed partition (upper bound; NOT a shipping split) ===")
print(f"  train devices={len(train_v)}  test devices={len(test_v)}")
print(f"  evaluable (>=1 test victim): {len(g_eval)} / {len(classes)}")
print(f"  usable (train+test disjoint victims): {len(g_use)} / {len(classes)}")
print(f"  structurally dead (<=1 victim, unsplittable): {structurally_dead}")

out = {
    "victim_definition": f"dst-concentration >= {SHARE:.0%} of class packets, FULL counts (no truncation)",
    "true_victim_pool_size": len(pool),
    "per_class_victims_ge5pct": nvic,
    "per_class_distinct_real_dst_total": {c: len(class_dst[c]) for c in classes},
    "random_worstcase": {"seeds": SEEDS,
                          "evaluable_per_seed": {str(s): len(v) for s, v in rand_eval.items()},
                          "usable_per_seed": {str(s): len(v) for s, v in rand_use.items()}},
    "greedy_upperbound": {"evaluable": len(g_eval), "usable": len(g_use),
                          "evaluable_classes": sorted(g_eval), "usable_classes": sorted(g_use)},
    "structurally_dead_classes": structurally_dead,
}
json.dump(out, open(os.path.join(HERE, "disjoint_split_corrected.json"), "w"), indent=2)
print(f"\nWrote {os.path.join(HERE, 'disjoint_split_corrected.json')}")
