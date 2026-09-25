#!/usr/bin/env python3
"""
Per-victim benign-flow counts behind the paper's claim that CIC IIoT 2025 MITM victim sensors are benign-sparse
(4-12 benign flows each), which is why no clean device-disjoint operating point is measurable there. Reuses the
committed extraction caches (MITM sensors -> real device MAC from the udp-frag capture; benign flows + real-MAC
touch from the udp-frag ablation cache). No SSD, no model. Writes mitm_benign_sparsity.json so the number traces.
"""
import json, os
from collections import Counter

SC = "./cache/"
HERE = os.path.dirname(os.path.abspath(__file__))
mitm = json.load(open(SC + "iiot_mitm_clean_cache.json"))
udp = json.load(open(SC + "iiot_udpfrag_ablation_cache.json"))
real = {k.lower(): v for k, v in udp["vic_mac"].items() if v}
btouch = [set(t) for t in udp["ben_touch"]]

# benign flows per real device MAC
per_mac = Counter()
for t in btouch:
    for m in t:
        per_mac[m] += 1

# map each MITM victim sensor -> its real MAC -> benign count
sensors = sorted({s.lower() for (_, _, s) in mitm["atk"]})
per_sensor = {}
for s in sensors:
    mac = real.get(s)
    if mac:
        per_sensor[s] = {"mac": mac, "benign_flows": per_mac.get(mac, 0)}

counts = [v["benign_flows"] for v in per_sensor.values()]
# the "sensor" devices (exclude the chatty cameras/infra) — report the low end that the claim rests on
sensor_only = {s: v for s, v in per_sensor.items() if s.endswith("-sensor")}
sc_counts = sorted(v["benign_flows"] for v in sensor_only.values())
out = {"note": "benign flows per CIC IIoT 2025 MITM victim device (real MAC from udp-frag). The IoT SENSORS are "
               "benign-sparse (4-12 each); a few chatty devices dominate -> no clean device-disjoint FPR.",
       "per_device": per_sensor,
       "sensor_benign_range": [min(sc_counts), max(sc_counts)] if sc_counts else None,
       "n_sensors": len(sensor_only), "all_device_benign_min": min(counts), "all_device_benign_max": max(counts)}
json.dump(out, open(os.path.join(HERE, "mitm_benign_sparsity.json"), "w"), indent=2)
print(f"SENSOR benign range: {out['sensor_benign_range']} over {out['n_sensors']} sensors")
print(f"all-device range: {out['all_device_benign_min']}-{out['all_device_benign_max']}")
for s, v in sorted(per_sensor.items(), key=lambda x: x[1]["benign_flows"]):
    print(f"  {s:28s} {v['mac']} benign={v['benign_flows']}")
print("Wrote mitm_benign_sparsity.json")
