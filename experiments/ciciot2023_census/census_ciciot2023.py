#!/usr/bin/env python3
"""
CICIoT2023 device-disjoint census — from the 'Packet From PCAP' CSVs (eth_src/eth_dst kept).
Streams archive.zip members (no 4.6GB extraction). Produces BOTH:
  A) the frozen src-anchored verdict (same procedure as CICIoMT2024's stage2_device_disjoint.py:
     role by cross-class frequency, victim if in <=40% of classes; need >=3 victims in >=10 classes)
  B) a topology-aware dst-concentration count (victim = destination absorbing a real share of a
     class's attack packets) reported at 1%/5%/10% thresholds so the verdict's sensitivity is visible.
Every number carries its basis. Broadcast/multicast excluded from victim candidates (hygiene, noted).
"""
import zipfile, io, collections, json, sys

ZIP = "./data/CICIOT23/archive.zip"
N_ROWS_HINT = None

# header: timestamp,eth_src,eth_dst,eth_type,ip_src,ip_dst,... -> eth_src=idx1, eth_dst=idx2
IDX_SRC, IDX_DST = 1, 2

def to_class(member):
    b = member[:-4] if member.endswith(".csv") else member
    return "Benign" if b.startswith("BenignTraffic") else b

def is_real_device(mac):
    if not mac or ":" not in mac:
        return False
    if mac == "ff:ff:ff:ff:ff:ff":       # broadcast
        return False
    first = mac.split(":")[0]
    try:
        b0 = int(first, 16)
    except ValueError:
        return False
    if b0 & 1:                            # multicast bit set (01:xx, 33:33, 03:.., etc.)
        return False
    return True

src_classes = collections.defaultdict(set)     # mac -> set(classes it SENT in)
dst_classes = collections.defaultdict(set)      # mac -> set(classes it RECEIVED in)
class_dst_counts = collections.defaultdict(collections.Counter)  # class -> Counter(dst -> pkts)
class_total = collections.Counter()

zf = zipfile.ZipFile(ZIP)
members = [m for m in zf.namelist() if m.endswith(".csv") and "/" not in m and not m.startswith("__")]
for member in sorted(members):
    cls = to_class(member)
    n = 0
    with zf.open(member) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="")
        text.readline()  # skip header
        for line in text:
            p = line.split(",", 4)
            if len(p) < 3:
                continue
            s, d = p[IDX_SRC], p[IDX_DST]
            src_classes[s].add(cls)
            dst_classes[d].add(cls)
            class_dst_counts[cls][d] += 1
            class_total[cls] += 1
            n += 1
    print(f"  {member:32s} -> class={cls:20s} rows={n:,}", file=sys.stderr)

classes = sorted(class_total)
NC = len(classes)
print(f"\nStreamed {sum(class_total.values()):,} packets across {NC} merged classes.", file=sys.stderr)

# ---- Check 1: is cross-class-frequency degenerate? histogram of len(src_classes[mac]) ----
src_hist = collections.Counter(len(v) for v in src_classes.values())
dst_hist = collections.Counter(len(v) for v in dst_classes.values())

# ---- (A) frozen src-anchored verdict (same procedure as CICIoMT2024) ----
INFRA, VICT = 0.80, 0.40
role = {}
for mac, cls_set in src_classes.items():
    frac = len(cls_set) / NC
    role[mac] = "infrastructure" if frac > INFRA else ("cross_class" if frac > VICT else "victim")
per_class_src_victims = collections.defaultdict(set)
for mac, cls_set in src_classes.items():
    if role[mac] == "victim":
        for c in cls_set:
            per_class_src_victims[c].add(mac)
src_ok = [c for c in classes if len(per_class_src_victims.get(c, ())) >= 3]
src_verdict = "VIABLE" if len(src_ok) >= 10 else "NOT_VIABLE"
role_counts = collections.Counter(role.values())

# ---- (B) topology dst-concentration victims, at 1%/5%/10% of a class's packets ----
dst_report = {}
for c in classes:
    tot = class_total[c]
    counts = class_dst_counts[c]
    row = {}
    for thr in (0.01, 0.05, 0.10):
        vict = [m for m, k in counts.items() if is_real_device(m) and tot and (k / tot) >= thr]
        row[f"victims_ge_{int(thr*100)}pct"] = len(vict)
    # top-5 real-device destinations for eyeballing
    top = [(m, k, round(k / tot, 3)) for m, k in counts.most_common(12) if is_real_device(m)][:5]
    row["top_real_dst"] = top
    dst_report[c] = row

def verdict_at(thrkey):
    ok = [c for c in classes if dst_report[c][thrkey] >= 3]
    return ("VIABLE" if len(ok) >= 10 else "NOT_VIABLE"), ok

out = {
    "dataset": "CICIoT2023 (Packet From PCAP CSVs, archive.zip)",
    "n_classes_merged": NC,
    "classes": classes,
    "total_packets": sum(class_total.values()),
    "per_class_packets": dict(class_total),
    "check1_src_class_count_histogram": dict(sorted(src_hist.items())),
    "check1_dst_class_count_histogram": dict(sorted(dst_hist.items())),
    "A_frozen_src_anchored": {
        "procedure": "SAME as CICIoMT2024 stage2: victim if src in <=40% of classes; >=3 victims in >=10 classes",
        "role_counts": dict(role_counts),
        "n_victim_macs": role_counts.get("victim", 0),
        "classes_with_ge3_victims": src_ok,
        "verdict": src_verdict,
        "per_class_src_victim_counts": {c: len(per_class_src_victims.get(c, ())) for c in classes},
    },
    "B_dst_concentration": {
        "procedure": "victim = destination MAC absorbing >= threshold of that class's packets (real devices only; bcast/mcast excluded)",
        "verdict_ge1pct":  {"verdict": verdict_at("victims_ge_1pct")[0],  "classes_ok": verdict_at("victims_ge_1pct")[1]},
        "verdict_ge5pct":  {"verdict": verdict_at("victims_ge_5pct")[0],  "classes_ok": verdict_at("victims_ge_5pct")[1]},
        "verdict_ge10pct": {"verdict": verdict_at("victims_ge_10pct")[0], "classes_ok": verdict_at("victims_ge_10pct")[1]},
        "per_class": dst_report,
    },
}
print(json.dumps(out, indent=2))
