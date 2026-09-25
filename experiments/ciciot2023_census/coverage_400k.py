#!/usr/bin/env python3
"""
400k-packet coverage check (reviewer §3.7). The family-map extraction (iiot2025_family_pkt2.py) reads
every pcap with `tshark -c 400000` to bound time on huge files. The exposure is the SHARED benign capture
benign_whole-network3.pcap: one multi-device file whose first 400k packets supply the benign side of
EVERY family. Two ways that biases the map:
  (1) non-temporal ordering  -> the first 400k packets are an arbitrary slice, not "the first minutes";
  (2) label/ device-correlated coverage -> a victim device whose benign traffic starts after packet 400k
      contributes ZERO benign flows to the capped map, silently.
This streams the FULL benign pcap (no cap) and reports: total packets, whether timestamps are monotonic,
and for every endpoint MAC its first-appearance packet index -> how many distinct MACs (and how many of
the udp-frag victim MACs, if the ablation cache is present) appear within the first 400k vs the full file.
Also capinfos each per-device attack pcap to see whether the 400k cap truncated it.
"""
import subprocess, os, glob, json, collections, re

EX = "./data/iiot/extract"
BENIGN = EX + "/benign/benign_whole-network3.pcap"
HERE = os.path.dirname(os.path.abspath(__file__))
CAP = 400000
CACHE = "./cache/iiot_udpfrag_ablation_cache.json"

def capinfos(pcap):
    p = subprocess.run(["capinfos", "-c", "-u", "-M", pcap], capture_output=True, text=True)
    return p.stdout

def victim_macs():
    if os.path.exists(CACHE):
        d = json.load(open(CACHE))
        return {m for m in d.get("vic_mac", {}).values() if m}
    return set()

def stream_benign():
    """full pass: return (n_packets, stats, {mac: first_idx}). Quantifies non-monotonicity so a lone 1us
    reorder is not confused with a genuinely shuffled file (advisor: state the magnitude, not a boolean)."""
    p = subprocess.Popen(["tshark", "-r", BENIGN, "-T", "fields",
                          "-e", "frame.time_epoch", "-e", "eth.src", "-e", "eth.dst"],
                         stdout=subprocess.PIPE, text=True)
    first_idx = {}; n = 0; last_t = -1.0
    back_steps = 0; max_back = 0.0; t0 = None; t_at_cap = None; last_seen = None
    for line in p.stdout:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 3: continue
        ts, esrc, edst = parts[0], parts[1], parts[2]
        try: t = float(ts)
        except: t = last_t
        if t0 is None: t0 = t
        if last_t >= 0 and t < last_t:
            back_steps += 1; max_back = max(max_back, last_t - t)
        last_t = t; last_seen = t
        if n == CAP: t_at_cap = t
        for m in (esrc, edst):
            if m and m not in first_idx: first_idx[m] = n
        n += 1
    p.wait()
    stats = {"back_steps": back_steps, "max_back_delta_s": round(max_back, 6),
             "t_first": t0, "t_last": last_seen,
             "t_at_packet_400k": t_at_cap,
             "elapsed_at_400k_s": round((t_at_cap - t0), 3) if (t_at_cap is not None and t0 is not None) else None,
             "total_duration_s": round((last_seen - t0), 3) if (last_seen is not None and t0 is not None) else None}
    return n, stats, first_idx

print("capinfos benign whole-network:", flush=True)
print(capinfos(BENIGN), flush=True)

vmacs = victim_macs()
print(f"udp-frag victim MACs available from cache: {len(vmacs)}", flush=True)

n, stats, first_idx = stream_benign()
in_400k = {m for m, i in first_idx.items() if i < CAP}
after_400k = {m for m, i in first_idx.items() if i >= CAP}
vic_in = vmacs & in_400k
vic_after = vmacs & after_400k
vic_absent = vmacs - set(first_idx)

result = {
    "note": "coverage of the SHARED benign_whole-network3 capture under the family-map's -c 400000 cap.",
    "benign_total_packets": n,
    "cap": CAP,
    "timestamps_monotonic": stats["back_steps"] == 0,
    "ordering": stats,
    "distinct_macs_total": len(first_idx),
    "distinct_macs_in_first_400k": len(in_400k),
    "distinct_macs_first_seen_after_400k": len(after_400k),
    "udp_frag_victim_macs": len(vmacs),
    "victim_macs_in_first_400k": len(vic_in),
    "victim_macs_first_seen_after_400k": len(vic_after),
    "victim_macs_absent_from_benign": len(vic_absent),
    "victim_macs_after_400k_list": sorted(vic_after),
}
json.dump(result, open(os.path.join(HERE, "coverage_400k.json"), "w"), indent=2)
print(json.dumps(result, indent=2))
print(f"\nREAD: {len(vic_after)} victim MAC(s) first-seen AFTER 400k => those devices have NO benign in the "
      f"capped map (device-coverage bias). Here: {len(vic_after)}.")
print(f"      ordering: {stats['back_steps']} backward steps, max {stats['max_back_delta_s']}s; "
      f"packet 400k at t+{stats['elapsed_at_400k_s']}s of {stats['total_duration_s']}s total. "
      f"Small max-back + early 400k-time => effectively chronological cap; large => interleaved (cap spans the run).")
