#!/usr/bin/env python3
"""
CIC IIoT 2025 gate — the check that decides whether pooling with CICIoT2023 adds NEW devices or the SAME ones.
Compares CIC IIoT 2025's device MACs/OUIs against the CICIoT2023 reference (ciciot2023_macs.json).

Usage:
  python mac_intersection_gate.py <path>
    <path> = a CIC IIoT 2025 .pcap/.pcapng  (MACs read via tshark)
           | a .csv that has eth_src/eth_dst (or Src MAC/Dst MAC) columns
           | a .txt list of MACs (one per line)

READ the verdict:
  - HIGH MAC overlap  -> same physical devices -> pooling adds ~0 victims; CIC IIoT 2025 only CONFIRMS the map
                         on a second capture (still useful, but does NOT escape the victim-diversity wall).
  - LOW MAC overlap, HIGH OUI overlap -> same device KITS, different units -> some new devices, but a shared
                         device-model fingerprint remains a cross-dataset confound to control.
  - LOW MAC and LOW OUI overlap -> genuinely new devices -> pooling can escape the wall (best case).
No detection, no download — a set comparison.
"""
import json, os, sys, subprocess, re, collections
HERE=os.path.dirname(os.path.abspath(__file__))
REF=json.load(open(os.path.join(HERE,"ciciot2023_macs.json")))
ref_macs=set(REF["macs"]); ref_ouis=set(REF["ouis"])
MAC_RE=re.compile(r"\b([0-9a-fA-F]{2}(:[0-9a-fA-F]{2}){5})\b")

def is_real(m):
    if not m or ":" not in m or m.lower()=="ff:ff:ff:ff:ff:ff": return False
    try: return not (int(m.split(":")[0],16)&1)
    except ValueError: return False

def macs_from(path):
    macs=set(); low=path.lower()
    if low.endswith((".pcap",".pcapng")):
        out=subprocess.run(["tshark","-r",path,"-T","fields","-e","eth.src","-e","eth.dst"],
                           capture_output=True,text=True).stdout
        for tok in MAC_RE.finditer(out): macs.add(tok.group(1).lower())
    elif low.endswith(".csv"):
        with open(path,encoding="utf-8",errors="replace") as fh:
            header=fh.readline().strip().split(","); cols={c.strip().lower():i for i,c in enumerate(header)}
            si=next((cols[k] for k in ("eth_src","src mac","srcmac","mac_src","source") if k in cols),None)
            di=next((cols[k] for k in ("eth_dst","dst mac","dstmac","mac_dst","destination") if k in cols),None)
            if si is None and di is None:   # fall back: regex any MAC in the line
                for line in fh:
                    for tok in MAC_RE.finditer(line): macs.add(tok.group(1).lower())
            else:
                for line in fh:
                    p=line.rstrip("\n").split(",")
                    for i in (si,di):
                        if i is not None and i<len(p): macs.add(p[i].strip().lower())
    else:
        for line in open(path,encoding="utf-8",errors="replace"):
            m=line.strip().lower()
            if MAC_RE.fullmatch(m): macs.add(m)
    return {m for m in macs if is_real(m)}

if len(sys.argv)<2:
    print(__doc__); print("REFERENCE READY: CICIoT2023 =",len(ref_macs),"MACs,",len(ref_ouis),"OUIs. Provide a CIC IIoT 2025 path."); sys.exit(0)

new=macs_from(sys.argv[1]); new_ouis={m[:8] for m in new}
mac_shared=new & ref_macs; oui_shared=new_ouis & ref_ouis
print(f"CIC IIoT 2025 slice: {len(new)} MACs, {len(new_ouis)} OUIs")
print(f"MAC overlap with CICIoT2023: {len(mac_shared)}/{len(new)} ({100*len(mac_shared)/max(len(new),1):.0f}%)")
print(f"OUI overlap: {len(oui_shared)}/{len(new_ouis)} ({100*len(oui_shared)/max(len(new_ouis),1):.0f}%)")
print(f"NEW MACs (not in CICIoT2023): {len(new-ref_macs)}   NEW OUIs: {len(new_ouis-ref_ouis)}")
frac=len(mac_shared)/max(len(new),1)
verdict=("SAME DEVICES — pooling adds ~0 victims (only confirms the map)" if frac>0.5 else
         "MIXED — some new devices; watch the shared-kit confound" if oui_shared else
         "NEW DEVICES — pooling can escape the victim-diversity wall (best case)")
print(f"\nGATE VERDICT: {verdict}")
json.dump({"slice":sys.argv[1],"n_new_macs":len(new),"mac_overlap":len(mac_shared),"oui_overlap":len(oui_shared),
           "new_macs":len(new-ref_macs),"new_ouis":len(new_ouis-ref_ouis),"verdict":verdict},
          open(os.path.join(HERE,"iiot2025_gate.json"),"w"),indent=2)
