#!/usr/bin/env python3
"""
Prep for the CIC IIoT 2025 gate: build the COMPLETE CICIoT2023 device-MAC set (and OUI/vendor-prefix set)
from the local zip, so the MAC-intersection check runs instantly once a CIC IIoT 2025 slice lands.
OUI matters: same lab may reuse the same device KITS (same vendor prefixes) even with different units.
"""
import zipfile, io, json, os, collections
ZIP=os.environ.get("CICIOT_ZIP",
    "./data/CICIOT23/archive.zip")
HERE=os.path.dirname(os.path.abspath(__file__))
def is_real(m):
    if not m or ":" not in m or m=="ff:ff:ff:ff:ff:ff": return False
    try: return not (int(m.split(":")[0],16)&1)
    except ValueError: return False
macs=set(); zf=zipfile.ZipFile(ZIP)
for member in sorted(m for m in zf.namelist() if m.endswith(".csv") and "/" not in m and not m.startswith("__")):
    with zf.open(member) as fh:
        text=io.TextIOWrapper(fh,encoding="utf-8",errors="replace",newline=""); text.readline()
        for line in text:
            p=line.split(",",3)
            if len(p)<3: continue
            for m in (p[1],p[2]):
                if is_real(m): macs.add(m)
ouis=collections.Counter(m[:8] for m in macs)
out={"n_macs":len(macs),"macs":sorted(macs),"n_ouis":len(ouis),"ouis":dict(sorted(ouis.items(),key=lambda kv:-kv[1]))}
json.dump(out,open(os.path.join(HERE,"ciciot2023_macs.json"),"w"),indent=2)
print(f"CICIoT2023: {len(macs)} distinct real device MACs across {len(ouis)} OUIs (vendor prefixes)")
print("top OUIs:", ", ".join(f"{o}({n})" for o,n in list(out['ouis'].items())[:8]))
print("Wrote ciciot2023_macs.json")
