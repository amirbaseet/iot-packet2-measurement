#!/usr/bin/env python3
"""
CORRECTED CICIoT2023 census — fixes the senior-review CRITICALs:
  #1/#2  RESTORE the infrastructure filter (the frozen stage2 holds infra/cross-class CONSTANT).
         A device is INFRASTRUCTURE if it is a meaningful destination (>=1% share) in >80% of
         classes (gateway/router/resolver) -> EXCLUDED from victim candidates.
  #3     No single improvised threshold: report viability as a SENSITIVITY CURVE over the share cut.
  #4     Bar is >=10 ATTACK classes (Benign excluded from the count).
Also dumps per-class victim MAC lists (non-infra) so the Mirai #52 test needs no re-stream.
Re-streams archive.zip (per-class victim MAC lists were never stored).
"""
import zipfile, io, collections, json, os

ZIP = "./data/CICIOT23/archive.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
IDX_DST = 2
INFRA_APPEAR_FRAC = 0.80     # meaningful-appearance fraction above which a device is infrastructure
APPEAR_SHARE = 0.01          # "meaningful" = >=1% of a class's packets
SHARE_CURVE = [0.01, 0.02, 0.05, 0.10, 0.20]
MIN_VICTIMS, MIN_CLASSES = 3, 10

def to_class(m):
    b = m[:-4] if m.endswith(".csv") else m
    return "Benign" if b.startswith("BenignTraffic") else b

def is_real_device(mac):
    if not mac or ":" not in mac or mac == "ff:ff:ff:ff:ff:ff":
        return False
    try:
        return not (int(mac.split(":")[0], 16) & 1)
    except ValueError:
        return False

# ── re-stream: full per-class dst packet counts (real devices) ──
class_dst = collections.defaultdict(collections.Counter)
class_total = collections.Counter()
zf = zipfile.ZipFile(ZIP)
for member in sorted(m for m in zf.namelist() if m.endswith(".csv") and "/" not in m and not m.startswith("__")):
    cls = to_class(member)
    with zf.open(member) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="")
        text.readline()
        for line in text:
            p = line.split(",", 4)
            if len(p) < 3:
                continue
            class_total[cls] += 1
            d = p[IDX_DST]
            if is_real_device(d):
                class_dst[cls][d] += 1
    print(f"  streamed {member}", flush=True)

classes = sorted(class_total)
attack_classes = [c for c in classes if c != "Benign"]
NC = len(classes)

# ── infrastructure filter: device meaningfully present (>=1% share) in >80% of classes ──
appear = collections.Counter()
for c in classes:
    tot = class_total[c]
    for m, k in class_dst[c].items():
        if tot and k / tot >= APPEAR_SHARE:
            appear[m] += 1
infra = {m for m, n in appear.items() if n / NC > INFRA_APPEAR_FRAC}
print(f"\nINFRASTRUCTURE excluded ({len(infra)} devices, appear in >{INFRA_APPEAR_FRAC:.0%} of classes):")
for m in sorted(infra, key=lambda m: -appear[m]):
    print(f"    {m}  meaningful in {appear[m]}/{NC} classes")

# ── victims = NON-infra destinations, per share threshold; sensitivity curve ──
def victims_at(c, t):
    tot = class_total[c]
    return [m for m, k in class_dst[c].items()
            if m not in infra and is_real_device(m) and tot and k / tot >= t]

curve = {}
for t in SHARE_CURVE:
    per_class = {c: victims_at(c, t) for c in classes}
    ok_attack = [c for c in attack_classes if len(per_class[c]) >= MIN_VICTIMS]
    ok_all = [c for c in classes if len(per_class[c]) >= MIN_VICTIMS]
    curve[t] = {
        "attack_classes_ge3_victims": len(ok_attack),
        "all_classes_ge3_victims": len(ok_all),
        "verdict_attack_bar": "VIABLE" if len(ok_attack) >= MIN_CLASSES else "NOT_VIABLE",
        "per_class_victim_counts": {c: len(per_class[c]) for c in classes},
    }

print(f"\n=== SENSITIVITY CURVE (victims = non-infra dst >= share; bar = >={MIN_VICTIMS} victims in >={MIN_CLASSES} ATTACK classes) ===")
print(f"{'share':>6s} {'attack_classes>=3v':>18s} {'verdict(attack bar)':>20s}")
for t in SHARE_CURVE:
    print(f"{t:6.0%} {curve[t]['attack_classes_ge3_victims']:18d} {curve[t]['verdict_attack_bar']:>20s}")

# ── per-class victim lists at 5% (non-infra) for the Mirai #52 test ──
victims_5 = {c: {m: class_dst[c][m] for m in victims_at(c, 0.05)} for c in classes}
mirai = ["Mirai-greeth_flood", "Mirai-greip_flood", "Mirai-udpplain"]
mirai_union = sorted({m for c in mirai for m in victims_5[c]})
print(f"\n=== Mirai flood victims (non-infra, >=5%) ===")
for c in mirai:
    print(f"  {c:22s} {sorted(victims_5[c])}")
print(f"  UNION across 3 Mirai classes: {len(mirai_union)} devices -> {mirai_union}")

out = {
    "correction": "infra filter restored (>%.0f%% meaningful-appearance excluded); sensitivity curve; attack-only bar" % (INFRA_APPEAR_FRAC*100),
    "n_classes": NC, "n_attack_classes": len(attack_classes),
    "infrastructure_excluded": sorted(infra),
    "infra_appearance": {m: appear[m] for m in sorted(infra)},
    "sensitivity_curve": {f"{int(t*100)}pct": curve[t] for t in SHARE_CURVE},
    "victims_5pct_by_class": {c: victims_5[c] for c in classes},
    "mirai_victim_union_5pct": mirai_union,
}
json.dump(out, open(os.path.join(HERE, "census_corrected.json"), "w"), indent=2)
print(f"\nWrote {os.path.join(HERE, 'census_corrected.json')}")
