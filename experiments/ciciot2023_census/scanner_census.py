#!/usr/bin/env python3
"""
WS2 Step 0 — scanner-source census (the GATE for attacker-disjoint recon).
Pre-flight risk #1 (mirror of the gateway bug): a recon capture contains traffic from BOTH the scanner
AND every host that replies. Counting distinct eth_src would count responders as "scanners". A real
scanner is defined by FAN-OUT: one source that contacts MANY distinct destinations. So we rank sources
by distinct-dst count per recon class, and count how many genuine scanners exist.

Pre-flight risk #2: scans are often single-packet (lone SYN). We also report the >=2-packet flow fraction
per recon class, so we know whether packet-2 features are even defined for recon.

Gate: attacker-disjoint needs >=2 distinct scanner sources per class (1 train / 1 test); >=3 to be comfortable.
Reads the LOCAL zip copy. No training here — census only.
"""
import zipfile, io, collections, json, os

ZIP=os.environ.get("CICIOT_ZIP",
    "./data/CICIOT23/archive.zip")
HERE=os.path.dirname(os.path.abspath(__file__))
RECON={"Recon-OSScan.csv":"OSScan","Recon-PortScan.csv":"PortScan","Recon-PingSweep.csv":"PingSweep",
       "Recon-HostDiscovery.csv":"HostDiscovery","VulnerabilityScan.csv":"VulnerabilityScan"}
# infra (from the corrected victim census) — gateways, excluded as scanner candidates
INFRA={"3c:18:a0:41:c3:a0","44:bb:3b:00:39:07"}
IDX_SRC,IDX_DST=1,2; IDX_IPSRC,IDX_IPDST,IDX_SPORT,IDX_DPORT,IDX_IPP=4,5,10,11,6
FANOUT_MIN=10          # a "scanner" contacts >= this many distinct destinations (reported at several cuts)

def is_real(mac):
    if not mac or ":" not in mac or mac=="ff:ff:ff:ff:ff:ff": return False
    try: return not (int(mac.split(":")[0],16)&1)
    except ValueError: return False

report={}
for member,name in RECON.items():
    src_dsts=collections.defaultdict(set)     # src mac -> set of distinct dst devices it contacted
    src_pkts=collections.Counter()
    flows=collections.Counter()               # 5-tuple -> packet count (for the >=2-packet gate)
    npkt=0
    zf=zipfile.ZipFile(ZIP)
    with zf.open(member) as fh:
        text=io.TextIOWrapper(fh,encoding="utf-8",errors="replace",newline="")
        text.readline()
        for line in text:
            p=line.rstrip("\n").split(",")
            if len(p)<=IDX_DPORT: continue
            s,d=p[IDX_SRC],p[IDX_DST]
            if is_real(s):
                src_dsts[s].add(p[IDX_IPDST])     # fan-out measured on dst IP (targets probed)
                src_pkts[s]+=1
            flows[(p[IDX_IPSRC],p[IDX_IPDST],p[IDX_SPORT],p[IDX_DPORT],p[IDX_IPP])]+=1
            npkt+=1
    # scanners = non-infra real sources with high fan-out
    fan={s:len(dsts) for s,dsts in src_dsts.items() if s not in INFRA}
    scanners_10=sorted([s for s,fo in fan.items() if fo>=10],key=lambda s:-fan[s])
    scanners_50=[s for s,fo in fan.items() if fo>=50]
    scanners_100=[s for s,fo in fan.items() if fo>=100]
    ge2=sum(1 for c in flows.values() if c>=2); pkts_ge2=sum(c for c in flows.values() if c>=2)
    report[name]={
        "packets":npkt,"distinct_real_src":len(fan),
        "top_sources_by_fanout":[{"mac":s,"fanout_dsts":fan[s],"packets":src_pkts[s]} for s in scanners_10[:8]] or
                                 [{"mac":s,"fanout_dsts":fan[s],"packets":src_pkts[s]} for s in sorted(fan,key=lambda s:-fan[s])[:8]],
        "n_scanners_fanout>=10":len(scanners_10),
        "n_scanners_fanout>=50":len(scanners_50),
        "n_scanners_fanout>=100":len(scanners_100),
        "pct_packets_in_ge2_flows":round(100*pkts_ge2/npkt,1) if npkt else 0.0,
    }
    print(f"{name:20s} pkts={npkt:>9,} real_src={len(fan):>3} "
          f"scanners(fanout>=10/50/100)={len(scanners_10)}/{len(scanners_50)}/{len(scanners_100)} "
          f">=2pkt={report[name]['pct_packets_in_ge2_flows']}%")
    print(f"    top sources by fan-out: "+", ".join(f"{d['mac']}({d['fanout_dsts']}dst)" for d in report[name]['top_sources_by_fanout'][:5]))

# gate verdict at fan-out>=10
n_ok_2=sum(1 for r in report.values() if r["n_scanners_fanout>=10"]>=2)
n_ok_3=sum(1 for r in report.values() if r["n_scanners_fanout>=10"]>=3)
verdict="VIABLE" if n_ok_2>=2 else "NOT_VIABLE"   # need >=2 recon classes each with >=2 scanners to even attempt
print(f"\nGATE: {sum(1 for r in report.values() if r['n_scanners_fanout>=10']>=2)}/{len(report)} recon classes have >=2 scanners (fanout>=10); "
      f"{n_ok_3}/{len(report)} have >=3.")
print(f"Attacker-disjoint is ATTEMPTABLE where a class has >=2 scanners AND >=2-packet flows exist.")
json.dump({"fanout_min_for_scanner":FANOUT_MIN,"infra_excluded":sorted(INFRA),
           "classes":report,"classes_with_ge2_scanners":n_ok_2,"classes_with_ge3_scanners":n_ok_3},
          open(os.path.join(HERE,"scanner_census.json"),"w"),indent=2)
print("Wrote scanner_census.json")
