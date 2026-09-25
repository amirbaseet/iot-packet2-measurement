#!/usr/bin/env python3
"""
CIC IIoT 2025 gate + census in one streaming pass over combined_dataset.csv (has device_mac + labels).
Answers the two questions the supervisor's dataset raises:
  (A) INTERSECTION: does CIC IIoT 2025 share CICIoT2023's device MACs/OUIs? (same devices -> pooling adds
      ~0; new devices -> can escape the victim-diversity wall)
  (B) CENSUS: how many distinct devices per attack class? (does IT alone escape the wall CICIoT2023 hit?)
Reads the LOCAL zip copy. Set comparison + counts, no training.
"""
import zipfile, io, csv, json, os, collections
ZIP=os.environ.get("IIOT2025_ZIP",
    "./data/iiot/iiot2025.zip")
HERE=os.path.dirname(os.path.abspath(__file__))
REF=json.load(open(os.path.join(HERE,"ciciot2023_macs.json")))
ref_macs=set(m.lower() for m in REF["macs"]); ref_ouis=set(REF["ouis"])
def is_real(m):
    if not m or ":" not in m or m.lower()=="ff:ff:ff:ff:ff:ff": return False
    try: return not (int(m.split(":")[0],16)&1)
    except ValueError: return False

zf=zipfile.ZipFile(ZIP); member=[m for m in zf.namelist() if m.endswith(".csv")][0]
dev_macs=set()
per_l1=collections.defaultdict(set); per_full=collections.defaultdict(set)
nrows=0
with zf.open(member) as fh:
    text=io.TextIOWrapper(fh,encoding="utf-8",errors="replace",newline="")
    r=csv.reader(text); header=next(r)
    col={c.strip():i for i,c in enumerate(header)}
    im=col.get("device_mac"); i1=col.get("label1"); iF=col.get("label_full")
    for row in r:
        nrows+=1
        if im is not None and im<len(row):
            m=row[im].strip().lower()
            if is_real(m):
                dev_macs.add(m)
                if i1 is not None and i1<len(row): per_l1[row[i1].strip()].add(m)
                if iF is not None and iF<len(row): per_full[row[iF].strip()].add(m)
print(f"rows={nrows:,}  distinct device_mac={len(dev_macs)}")

# (A) intersection
new_ouis={m[:8] for m in dev_macs}
mac_shared=dev_macs & ref_macs; oui_shared=new_ouis & ref_ouis
frac=len(mac_shared)/max(len(dev_macs),1)
print(f"\n[A] INTERSECTION with CICIoT2023 ({len(ref_macs)} MACs / {len(ref_ouis)} OUIs):")
print(f"    MAC overlap {len(mac_shared)}/{len(dev_macs)} ({100*frac:.0f}%)  |  OUI overlap {len(oui_shared)}/{len(new_ouis)}")
print(f"    NEW MACs {len(dev_macs-ref_macs)}  NEW OUIs {len(new_ouis-ref_ouis)}")
verdict=("SAME DEVICES — pooling adds ~0; CIC IIoT 2025 would only CONFIRM the map on a 2nd dataset" if frac>0.5 else
         "MIXED — some shared device kits (OUI); new units present; watch the shared-kit confound" if oui_shared else
         "NEW DEVICES — genuinely different fleet; pooling can escape the victim-diversity wall")
print(f"    VERDICT: {verdict}")

# (B) census — distinct devices per class
def summ(d,name,bar=3,minc=10):
    ok=[c for c,s in d.items() if len(s)>=bar]
    print(f"\n[B] {name}: {len(d)} classes; {len(ok)} have >={bar} distinct devices "
          f"-> {'VIABLE' if len(ok)>=minc else 'NOT_VIABLE'} (bar: >={bar} in >={minc})")
    for c in sorted(d,key=lambda c:-len(d[c])):
        print(f"    {c:34s} {len(d[c]):3d} devices")
    return {c:len(s) for c,s in d.items()}, ok
l1_counts,l1_ok=summ(per_l1,"by label1 (broad category)")
full_counts,full_ok=summ(per_full,"by label_full (specific attack)")

json.dump({"n_rows":nrows,"n_device_macs":len(dev_macs),
           "intersection":{"mac_overlap":len(mac_shared),"oui_overlap":len(oui_shared),
                           "new_macs":len(dev_macs-ref_macs),"new_ouis":len(new_ouis-ref_ouis),"verdict":verdict},
           "devices_per_label1":l1_counts,"devices_per_label_full":full_counts,
           "label1_classes_ge3":l1_ok,"label_full_classes_ge3":full_ok},
          open(os.path.join(HERE,"iiot2025_gate.json"),"w"),indent=2)
print("\nWrote iiot2025_gate.json")
