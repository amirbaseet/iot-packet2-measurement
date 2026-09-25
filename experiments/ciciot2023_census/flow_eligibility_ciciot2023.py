#!/usr/bin/env python3
"""
Flow-eligibility on CICIoT2023 (reviewer round-2 §1: is eligibility a confound for the dividing line?).
On CIC IIoT 2025 the failing families had LOW eligibility (MITM 50%, Mirai 9%) and the generalizing ones HIGH
(flood 99%) -- monotone with the verdict, which is the referee's worry. The decisive test is a family that
BREAKS that monotonicity. CICIoT2023 MITM, DNS spoofing, and Mirai all FAIL device-disjoint here; if they fail
at HIGH eligibility, eligibility does not predict the verdict and the confound is empirically dead.

Eligibility = fraction of directional flow keys (p[4],p[5],p[10],p[11],p[6]) that reach a 2nd packet -- the exact
key feature_ablation.py uses. Reads archive.zip (packet-level CSVs). Env CICIOT_ZIP to override.
"""
import zipfile, io, os, json, collections

ZIP = os.environ.get("CICIOT_ZIP", "./data/CICIOT23/archive.zip")
HERE = os.path.dirname(os.path.abspath(__file__))
FAMILIES = {
    "MITM-ArpSpoofing": ["MITM-ArpSpoofing.csv"],
    "DNS_Spoofing":     ["DNS_Spoofing.csv"],
    "Mirai_flood":      ["Mirai-greeth_flood.csv", "Mirai-greip_flood.csv", "Mirai-udpplain.csv"],
    "Recon":            ["Recon-OSScan.csv", "Recon-PortScan.csv", "Recon-HostDiscovery.csv"],
    "Benign":           ["BenignTraffic.csv", "BenignTraffic1.csv", "BenignTraffic2.csv", "BenignTraffic3.csv"],
}
# device-disjoint verdict from the paper (for the readout)
VERDICT = {"MITM-ArpSpoofing": "FAILS (AUC 0.44)", "DNS_Spoofing": "FAILS (AUC 0.44-0.80)",
           "Mirai_flood": "FAILS (AUC 0.66)", "Recon": "attacker-axis (0.66-0.82)", "Benign": "-"}

def eligibility(members):
    count = collections.Counter(); zf = zipfile.ZipFile(ZIP)
    for m in members:
        with zf.open(m) as fh:
            t = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline=""); t.readline()
            for line in t:
                p = line.rstrip("\n").split(",")
                if len(p) <= 11: continue
                count[(p[4], p[5], p[10], p[11], p[6])] += 1
    tot = len(count); elig = sum(1 for v in count.values() if v >= 2)
    tp = sum(count.values()); ep = sum(v for v in count.values() if v >= 2)
    return {"total_flow_keys": tot, "eligible_flow_keys": elig,
            "eligible_flow_frac": round(elig / tot, 4) if tot else None,
            "eligible_packet_frac": round(ep / tp, 4) if tp else None}

out = {"note": "CICIoT2023 flow-eligibility per family (2nd-same-direction-packet fraction), to test whether "
               "eligibility predicts the device-disjoint verdict. If failing families are HIGH-eligible here, "
               "eligibility is not the confound.", "families": {}}
for fam, members in FAMILIES.items():
    r = eligibility(members); r["disjoint_verdict"] = VERDICT[fam]; out["families"][fam] = r
    print(f"{fam:18s} eligible {r['eligible_flow_frac']}  ({r['eligible_flow_keys']}/{r['total_flow_keys']})  "
          f"verdict: {r['disjoint_verdict']}", flush=True)
json.dump(out, open(os.path.join(HERE, "flow_eligibility_ciciot2023.json"), "w"), indent=2)
print("\nWrote flow_eligibility_ciciot2023.json")
print("READ: if MITM/DNS/Mirai are HIGH-eligible yet FAIL, eligibility does not predict the verdict -> "
      "the dividing line is the mechanism (2 packets; length is not a feature), not eligibility.")
