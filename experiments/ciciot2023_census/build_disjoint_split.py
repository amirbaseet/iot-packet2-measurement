#!/usr/bin/env python3
"""
Option-2 test: does a REAL victim-hold-out device-disjoint split survive on CICIoT2023?
The census counted victims; this checks that holding out ~30% of victim DEVICES leaves each
attack class with attack traffic on BOTH sides (train AND test). A class whose victims all
land on one side VANISHES from the other and cannot be evaluated there.

Victim definition = dst-concentration >=5% of a class's packets (the definition that gave VIABLE).
Partition is GLOBAL and DISJOINT: a device is train-victim or test-victim across all classes.
Multi-seed (stage2 precedent). Reads the saved census JSON; no re-streaming of the 4.6GB zip.
"""
import json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
CENSUS = json.load(open(os.path.join(HERE, "census_ciciot2023.json")))
SEEDS = [1, 7, 42, 100, 1729]
HOLDOUT_FRAC = 0.30
SHARE = 0.05  # victim = dst absorbing >= this share of the class (the VIABLE-at-5% definition)

# ── per-class victim sets (mac -> packets) from saved top_real_dst (share >= 5%) ──
per_class_victims = {}   # class -> {mac: packets}
for cls, row in CENSUS["B_dst_concentration"]["per_class"].items():
    vics = {mac: cnt for (mac, cnt, share) in row["top_real_dst"] if share >= SHARE}
    per_class_victims[cls] = vics

classes = CENSUS["classes"]
pool = sorted({m for v in per_class_victims.values() for m in v})   # global victim pool
print(f"Global victim pool: {len(pool)} distinct devices across {len(classes)} classes")
print(f"Classes with >=3 victims (the VIABLE bar): "
      f"{sum(1 for c in classes if len(per_class_victims[c])>=3)}\n")

def run_seed(seed):
    rng = random.Random(seed)
    perm = list(pool); rng.shuffle(perm)
    n_test = max(1, int(round(len(perm) * HOLDOUT_FRAC)))
    test_v = set(perm[:n_test]); train_v = set(perm[n_test:])
    assert train_v.isdisjoint(test_v), "LEAK: a device is in both train and test"
    surviving = []
    table = []
    for c in classes:
        vics = per_class_victims[c]
        tr = {m: k for m, k in vics.items() if m in train_v}
        te = {m: k for m, k in vics.items() if m in test_v}
        ok = len(tr) >= 1 and len(te) >= 1           # class present on BOTH sides
        if ok:
            surviving.append(c)
        table.append((c, len(vics), len(tr), len(te), sum(tr.values()), sum(te.values()), ok))
    return train_v, test_v, surviving, table

# ── multi-seed summary ──
print(f"{'seed':>5s} {'train_dev':>9s} {'test_dev':>8s} {'classes_both_sides':>18s}")
per_seed_surv = {}
for s in SEEDS:
    tr, te, surv, _ = run_seed(s)
    per_seed_surv[s] = surv
    print(f"{s:5d} {len(tr):9d} {len(te):8d} {len(surv):18d}")

counts = [len(v) for v in per_seed_surv.values()]
print(f"\nClasses surviving on BOTH sides: min={min(counts)} max={max(counts)} "
      f"(bar for a usable eval: as many of the 18 as possible; >=10 was the census VIABLE bar)")

# ── representative per-class table (seed 42) ──
_, _, _, table = run_seed(42)
print(f"\n=== per-class split (seed 42, 30% held out) ===")
print(f"{'class':24s} {'vics':>4s} {'trainV':>6s} {'testV':>5s} {'train_pkts':>11s} {'test_pkts':>10s} {'BOTH?':>6s}")
for (c, nv, ntr, nte, trp, tep, ok) in table:
    print(f"{c:24s} {nv:4d} {ntr:6d} {nte:5d} {trp:11,} {tep:10,} {'YES' if ok else 'NO':>6s}")

# which classes fail on some seed?
always = set(classes)
for s in SEEDS:
    always &= set(per_seed_surv[s])
never_all = [c for c in classes if c not in always]
print(f"\nSurvive on ALL {len(SEEDS)} seeds: {len(always)} classes")
print(f"Fail on >=1 seed (fragile/unsplittable): {sorted(never_all)}")

out = {
    "victim_definition": f"dst-concentration >= {SHARE:.0%} of class packets (VIABLE-at-5% definition)",
    "holdout_frac": HOLDOUT_FRAC, "seeds": SEEDS,
    "global_victim_pool_size": len(pool),
    "classes_surviving_both_sides_per_seed": {str(s): sorted(v) for s, v in per_seed_surv.items()},
    "surviving_all_seeds": sorted(always),
    "fragile_or_unsplittable": sorted(never_all),
    "min_surviving": min(counts), "max_surviving": max(counts),
}
json.dump(out, open(os.path.join(HERE, "disjoint_split_test.json"), "w"), indent=2)
print(f"\nWrote {os.path.join(HERE,'disjoint_split_test.json')}")
