#!/usr/bin/env python3
"""
GATE (advisor): packet-2 = first-two-packets-per-flow. If Mirai floods use spoofed/randomized
source ports, most packets land in 1-packet 'flows' that never reach packet 2 -> packet-2 features
are mostly UNDEFINED for exactly the classes we want to test, and any result is about the flow
assembler, not #52. Count, before building anything, the fraction of packets in >=2-packet flows.
Flow key = (ip_src, ip_dst, src_port, dst_port, ip_p). Report overall AND victim-directed.
"""
import zipfile, io, collections, os

ZIP = "./data/CICIOT23/archive.zip"
# header: timestamp,eth_src,eth_dst,eth_type,ip_src,ip_dst,ip_p,ip_len,ip_ttl,transport_proto,src_port,dst_port,...
I_ETHDST, I_IPSRC, I_IPDST, I_IPP, I_SPORT, I_DPORT = 2, 4, 5, 6, 10, 11
MIRAI = ["Mirai-greeth_flood.csv", "Mirai-greip_flood.csv", "Mirai-udpplain.csv"]
VICTIMS = {"08:7c:39:ce:6e:2a","1c:12:b0:9b:0c:ec","1c:fe:2b:98:16:dd","9c:8e:cd:1d:ab:9f","cc:f4:11:9c:d0:00"}

zf = zipfile.ZipFile(ZIP)
print(f"{'class':22s} {'packets':>10s} {'flows':>10s} {'ge2_flows':>9s} {'pkts_in_ge2':>11s} {'%pkts_ge2':>9s} {'%vic_ge2':>8s}")
for member in MIRAI:
    flows = collections.Counter()          # flow key -> packet count
    vic_flows = collections.Counter()      # same, but only flows whose dst-device is a victim
    npkt = 0
    with zf.open(member) as fh:
        text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="")
        text.readline()
        for line in text:
            p = line.rstrip("\n").split(",")
            if len(p) <= I_DPORT:
                continue
            key = (p[I_IPSRC], p[I_IPDST], p[I_SPORT], p[I_DPORT], p[I_IPP])
            flows[key] += 1
            if p[I_ETHDST] in VICTIMS:
                vic_flows[key] += 1
            npkt += 1
    ge2 = {k: c for k, c in flows.items() if c >= 2}
    pkts_ge2 = sum(ge2.values())
    vic_pkts = sum(vic_flows.values())
    vic_pkts_ge2 = sum(c for k, c in vic_flows.items() if c >= 2)
    pct = 100.0 * pkts_ge2 / npkt if npkt else 0
    vpct = 100.0 * vic_pkts_ge2 / vic_pkts if vic_pkts else 0
    print(f"{member[:-4]:22s} {npkt:10,} {len(flows):10,} {len(ge2):9,} {pkts_ge2:11,} {pct:8.1f}% {vpct:7.1f}%")
