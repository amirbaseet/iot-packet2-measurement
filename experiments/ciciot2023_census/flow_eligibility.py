#!/usr/bin/env python3
"""
Flow-eligibility fraction (reviewer §3.6). The packet-2 method emits a feature row ONLY when a flow key
(ip.src, ip.dst, sport, dport, proto) sees a SECOND same-direction packet. Single-packet flows (a lone
SYN, a one-shot UDP probe, a scan touch) are silently dropped and never scored. If that drop rate
correlates with the label, the evaluated population is a biased subset. This reports, per family (and
benign), the fraction of distinct directional flow keys that reach a 2nd packet (= are eligible) and the
fraction of packets that live in eligible flows.

Measured on the SAME population the family map uses: `tshark -c 400000` per pcap (env TSHARK_CAP to change).
Read-only over the SSD; no model fitting.
"""
import subprocess, os, glob, json, collections

EX = "./data/iiot/extract"
HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.environ.get("TSHARK_CAP", "400000")
# family dir -> glob; benign handled separately (single whole-network file)
FAM_DIRS = {"ddos_udpfrag": "ddos/*udp-frag-flood*.pcap", "dos_syn80": "dos_syn80/*.pcap",
            "mitm": "mitm/*.pcap", "recon": "recon/*.pcap", "bruteforce": "bruteforce/*.pcap",
            "malware_mirai": "malware/*.pcap"}
BENIGN = "benign/benign_whole-network3.pcap"

def tshark(pcap):
    p = subprocess.run(["tshark", "-r", pcap, "-c", CAP, "-T", "fields",
                        "-e", "ip.src", "-e", "ip.dst", "-e", "ip.proto",
                        "-e", "tcp.srcport", "-e", "tcp.dstport", "-e", "udp.srcport", "-e", "udp.dstport"],
                       capture_output=True, text=True)
    for line in p.stdout.splitlines():
        yield line.split("\t")

def eligibility(pcaps):
    """count[key] = packets on that directional key; eligible = keys with >=2."""
    count = collections.Counter()
    for pcap in pcaps:
        for c in tshark(pcap):
            if len(c) < 7: continue
            ips, ipd, proto, tsp, tdp, usp, udp_ = c[:7]
            if not ips or not ipd: continue
            sport = tsp or usp; dport = tdp or udp_
            count[(ips, ipd, sport, dport, proto)] += 1
    total_keys = len(count)
    eligible_keys = sum(1 for v in count.values() if v >= 2)
    total_pkts = sum(count.values())
    pkts_in_eligible = sum(v for v in count.values() if v >= 2)
    return {"total_flow_keys": total_keys, "eligible_flow_keys": eligible_keys,
            "eligible_flow_frac": round(eligible_keys / total_keys, 4) if total_keys else None,
            "total_packets": total_pkts, "packets_in_eligible_flows": pkts_in_eligible,
            "eligible_packet_frac": round(pkts_in_eligible / total_pkts, 4) if total_pkts else None}

out = {"note": f"fraction of directional flow keys that reach a 2nd same-direction packet (= packet-2 "
               f"eligible). tshark -c {CAP} per pcap, same population as the family map.", "cap": CAP, "families": {}}
for fam, pat in FAM_DIRS.items():
    pcaps = sorted(p for p in glob.glob(f"{EX}/{pat}") if "/._" not in p)
    if not pcaps:
        print(f"{fam:16s} NO PCAPS ({pat})", flush=True); continue
    r = eligibility(pcaps); r["n_pcaps"] = len(pcaps); out["families"][fam] = r
    print(f"{fam:16s} eligible flows {r['eligible_flow_frac']}  ({r['eligible_flow_keys']}/{r['total_flow_keys']})  "
          f"eligible packets {r['eligible_packet_frac']}  [{len(pcaps)} pcaps]", flush=True)
r = eligibility([f"{EX}/{BENIGN}"]); r["n_pcaps"] = 1; out["families"]["benign"] = r
print(f"{'benign':16s} eligible flows {r['eligible_flow_frac']}  ({r['eligible_flow_keys']}/{r['total_flow_keys']})  "
      f"eligible packets {r['eligible_packet_frac']}", flush=True)

json.dump(out, open(os.path.join(HERE, "flow_eligibility.json"), "w"), indent=2)
print("\nWrote flow_eligibility.json")
print("READ: if eligible_flow_frac differs a lot across labels (e.g. flood low, benign high), the packet-2 "
      "population is a label-correlated subset and the reported AUC/FPR describe only that subset.")
