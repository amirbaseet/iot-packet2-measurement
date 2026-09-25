#!/usr/bin/env python3
"""
Rigor pass: pull EVERY number the workshop paper will cite from the committed result JSONs into one
source-of-truth file (paper_numbers.json), with seed-spread where per-partition/per-seed arrays exist.
The paper cites THIS file; no number is retyped from prose. Read-only on the result JSONs.
"""
import json, os, statistics as stt
HERE=os.path.dirname(os.path.abspath(__file__))
def L(n): return json.load(open(os.path.join(HERE,n)))
def spread(vals):
    vals=[v for v in vals if v is not None]
    if not vals: return None
    return {"n":len(vals),"min":round(min(vals),4),"max":round(max(vals),4),
            "mean":round(stt.mean(vals),4),"median":round(stt.median(vals),4),
            "std":round(stt.pstdev(vals),4) if len(vals)>1 else 0.0}

out={}

# --- census ---
cc=L("census_corrected.json")
out["census"]={
  "CICIoMT2024":"NOT_VIABLE: 19 classes, 5 victim devices total, ~1 per class (source: feasibility.json, prior)",
  "CICIoT2023_infra_excluded":cc["infrastructure_excluded"],
  "CICIoT2023_sensitivity_attack_classes_ge3victims":{k:v["attack_classes_ge3_victims"] for k,v in cc["sensitivity_curve"].items()},
  "CICIoT2023_verdict":"NOT_VIABLE at meaningful (>=5%) thresholds",
  "CIC_IIoT2025_gate":{k:L("iiot2025_gate.json")[k] for k in ("n_device_macs","intersection") if k in L("iiot2025_gate.json")},
}

# --- Mirai packet-2 device-disjoint on CICIoT2023 (per-partition) ---
md=L("mirai_disjoint_result.json"); parts=md["disjoint_partitions"]
out["ciciot2023_mirai_pkt2"]={
  "random":md["random"],
  "disjoint_auc":spread([p["auc"] for p in parts]),
  "disjoint_fpr_at_thr":spread([p["fpr_at_thr"] for p in parts]),
  "n_partitions":len(parts)}

# --- earliness (pkt-2/20/full) ---
er=L("earliness_result.json")
out["earliness"]={row["tag"]:{"random_auc":row["random_auc"],"random_fpr":row["random_fpr"],
                              "disjoint_auc_min_med_max":row["disjoint_auc"],"disjoint_fpr_min_med_max":row["disjoint_fpr"],
                              "at_chance":row["at_chance"]} for row in er}

# --- recon attacker-disjoint (full feature set) ---
rr=L("recon_result.json")["full"]
out["recon_attacker_disjoint"]={k:{"random_auc":v["random"]["auc"],"random_fpr":v["random"]["fpr"],
    "disjoint_auc_median":v["attacker_disjoint"]["auc_median"],"disjoint_auc_range":v["attacker_disjoint"]["auc_range"],
    "disjoint_fpr_median":v["attacker_disjoint"]["fpr_median"]} for k,v in rr.items() if isinstance(v,dict) and "attacker_disjoint" in v}

# --- CIC IIoT 2025 windowed device-disjoint (aggregate + per-family) ---
wd=L("iiot2025_disjoint.json")
out["iiot2025_windowed_aggregate"]={"n_devices":wd["n_devices"],"random":wd["random"],
    "disjoint_auc_min_med_max":wd["disjoint_auc"],"disjoint_fpr_min_med_max":wd["disjoint_fpr"]}
out["iiot2025_windowed_per_family"]=L("iiot2025_per_family.json")

# --- CIC IIoT 2025 packet-2 (udp-frag-flood 26 victims) ---
dp=L("iiot2025_ddos_pkt2.json")
out["iiot2025_packet2_udpfrag"]={"n_victim_devices":dp["n_victim_devices"],"random":dp["random"],
    "victim_disjoint_auc":dp["victim_disjoint"]["auc"],"victim_disjoint_fpr":dp["victim_disjoint"]["fpr"]}

# --- CIC IIoT 2025 packet-2 Mirai (8 victims) ---
mp=L("iiot2025_mirai_pkt2.json")
out["iiot2025_packet2_mirai"]={"n_victim_devices":mp["n_victim_devices"],"random":mp["random"],
    "victim_disjoint_auc":mp["victim_disjoint"]["auc"],"victim_disjoint_fpr":mp["victim_disjoint"]["fpr"]}

# --- family sweep (dos/mitm/recon/web/bruteforce packet-2) ---
out["iiot2025_packet2_family_sweep"]={r["family"]:{"n_victim_devices":r["n_victim_devices"],
    "random_auc":r["random_auc"],"random_fpr":r["random_fpr"],
    "disjoint_auc_min_med_max":r.get("disjoint_auc"),"disjoint_fpr_min_med_max":r.get("disjoint_fpr")}
    for r in L("iiot2025_family_summary.json")}

# --- CICIoT2023 MITM-ArpSpoofing + DNS_Spoofing packet-2 leave-one-victim-out ---
mdd=L("mitm_dns_disjoint_result.json")["results"]
def _loo(cls):
    r=mdd[cls]; folds=r["leave_one_victim_out"]; cal=[f for f in folds if f.get("calibrated")]
    return {"random_auc":r["random"]["auc"],"random_fpr":r["random"]["fpr_at_thr"],
            "n_calibrated_folds":len(cal),
            "disjoint_auc_per_fold":[f["auc"] for f in cal],
            "disjoint_fpr_per_fold":[f["fpr_at_thr"] for f in cal],
            "disjoint_auc":spread([f["auc"] for f in cal]),
            "disjoint_fpr":spread([f["fpr_at_thr"] for f in cal])}
out["ciciot2023_mitm_dns_pkt2"]={
  "MITM_ArpSpoofing":_loo("MITM-ArpSpoofing"),
  "DNS_Spoofing":_loo("DNS_Spoofing"),
  "note":"leave-one-victim-out over 3 census 5%-victims/class; folds reported individually (n=2 MITM, "
         "3 DNS); MITM.csv holds 0 ARP frames -> measures redirected IP traffic, not the ARP spoof itself"}

# --- controlled within-dataset device-count sweep (udp-frag, fixed held-out victims) ---
sw=L("iiot2025_devcount_sweep.json")
out["iiot2025_devcount_sweep"]={
  "held_test_victims":sw["n_held_test"],"training_pool_available":sw["training_pool_available"],
  "auc_by_train_victims":{r["train_victims"]:r["auc_median"] for r in sw["sweep"]},
  "fpr_by_train_victims":{r["train_victims"]:r["fpr_median"] for r in sw["sweep"]},
  "note":"WITHIN CIC IIoT 2025 udp-frag; only training-victim count varies; ranking flat -> device count is "
         "NOT the ranking driver; the cross-dataset 0.66-vs-0.997 gap is a dataset/attack effect"}

# --- pooling confound ---
pv=L("pooled_mirai_v2.json")
out["pooling_confound"]={"origin_auc_final":pv["origin_auc_final"],"surviving_features":pv["surviving_features"],
    "dropped_features":pv["dropped_features"],"pooled_victims":pv["pooled_victims"],
    "note":"origin control (predict lab from benign) stays ~1.0 even after dropping 9/12 features -> pooling confounded"}

# --- reviewer revision experiments (2026-09-09) ---
fa=L("feature_ablation_ciciot2023.json"); fi=L("feature_ablation_iiot2025.json")
out["feature_ablation"]={
  "ciciot2023":{fam:{m:{"disjoint_auc":d[m]["disjoint_auc_med"],"random_auc":d[m]["random_auc"]}
                     for m in d} for fam,d in fa["families"].items()},
  "iiot2025_udpfrag":{m:{"disjoint_auc":fi["results"][m]["disjoint_auc_med"],
                         "disjoint_fpr":fi["results"][m]["disjoint_fpr_med"],
                         "random_auc":fi["results"][m]["random_auc"]} for m in fi["results"]},
  "note":"masks drop identity fields (ttl/win=stack, sport=allocation) and, for the flood, no_size drops the "
         "attack channel. Disjoint AUC barely moves under any mask -> the signal is redundantly encoded, NOT "
         "localizable to a few fields (null-localization); the isolation experiment the reviewer asked for "
         "returned null. Flood no_size: AUC holds 0.99 but FPR doubles 0.23->0.53."}

wc=L("wilson_ci.json")
out["wilson_ci_disjoint_fpr"]={fam:[{"fpr":f["fpr"],"wilson95":f["wilson95"],"benign_n":f["benign_n"]}
                                    for f in folds] for fam,folds in wc["families"].items()}
out["wilson_ci_disjoint_fpr"]["note"]=("Wilson 95% score interval on device-disjoint FPR; tight intervals (<2pp "
    "for MITM/DNS at 98-100%) -> the operating-point failure is not sampling noise; Mirai spans 10-99% across folds")

cv=L("coverage_400k.json")
out["coverage_400k"]={"benign_total_packets":cv["benign_total_packets"],"cap":cv["cap"],
  "back_steps":cv["ordering"]["back_steps"],"max_back_delta_s":cv["ordering"]["max_back_delta_s"],
  "elapsed_at_400k_s":cv["ordering"]["elapsed_at_400k_s"],"total_duration_s":cv["ordering"]["total_duration_s"],
  "victim_macs_in_first_400k":cv["victim_macs_in_first_400k"],"udp_frag_victim_macs":cv["udp_frag_victim_macs"],
  "note":"the family-map -c 400000 cap is effectively chronological (max reorder 2us) and takes the first ~1.6h "
         "of a 12h capture; all 26 victim MACs appear in it -> full device coverage, benign side not biased"}

fe=L("flow_eligibility.json")
out["flow_eligibility"]={fam:{"eligible_flow_frac":r["eligible_flow_frac"],"eligible_packet_frac":r["eligible_packet_frac"]}
                         for fam,r in fe["families"].items()}
out["flow_eligibility"]["note"]=("fraction of directional flow keys reaching a 2nd same-direction packet (packet-2 "
    "eligible); swings 99%(flood)->1.2%(recon), 9%(Mirai) -> per-family AUC/FPR describe a label-correlated subset")

mc=L("iiot2025_mitm_clean.json")
out["iiot2025_mitm_clean"]={"n_groups":mc["n_groups"],"random":mc["random"],"clean_disjoint":mc["clean_disjoint"],
  "sweep":{r["train_groups"]:r["auc_median"] for r in mc["sweep"]},
  "note":"CLEAN CIC IIoT 2025 MITM device-disjoint (24 victim sensors, real-MAC symmetric holdout). Neither this "
         "(AUC 0.80/FPR 0.69, benign-sparse: sensors have 4-12 benign flows) NOR the earlier 0.53/91% (defective "
         "split) is a trustworthy number. MITM collapse rests on CICIoT2023 (AUC 0.44, FPR 98%). SWEEP survives: "
         "AUC does not climb with training victims (0.39->0.72, no trend), FPR ~90% -> no recovery from device count"}

wm=L("iiot2025_windowed_matched.json")
out["windowed_matched_udpfrag"]={"disjoint":wm["disjoint"],"random":wm["random"],
  "matched_victims":wm["matched_victims"],"n_feature_cols":wm["n_feature_cols"],
  "packet2_reference":wm["packet2_udpfrag_reference"],
  "note":"MATCHED windowed vs packet-2 on the SAME udp-frag victims: windowed disjoint AUC 0.973 / FPR median "
         "0.035 vs packet-2 0.997 / 0.226 -> windowed keeps ranking, cuts the operating point 22.6%->3.5% "
         "(range 1.4-37.5%, not perfectly stable). Cost: per-device windows, not packet-2 inline."}

bs=L("mitm_benign_sparsity.json")
out["mitm_benign_sparsity"]={"sensor_benign_range":bs["sensor_benign_range"],"n_sensors":bs["n_sensors"],
  "all_device_benign_min":bs["all_device_benign_min"],"all_device_benign_max":bs["all_device_benign_max"],
  "note":"CIC IIoT 2025 MITM victim SENSORS carry 4-12 benign flows each (14 sensors); chatty devices reach 1507 "
         "-> device-disjoint operating point unmeasurable, dominated by which device lands in the fold"}

ec=L("flow_eligibility_ciciot2023.json")
out["flow_eligibility_ciciot2023"]={fam:{"eligible_flow_frac":r["eligible_flow_frac"],"verdict":r["disjoint_verdict"]}
                                     for fam,r in ec["families"].items()}
out["flow_eligibility_ciciot2023"]["note"]=("round-2 confound check: CICIoT2023 failing families MITM 40%, DNS "
    "37%, Mirai 52% eligible (Benign 55%). Mirai FAILS at 52% here vs 9% on CIC IIoT 2025 -> same family+verdict "
    "across a 6x eligibility gap, so eligibility does not set the verdict; length is not a packet-2 feature")

fp=L("fingerprint_positive.json"); fc=L("fingerprint_positive_ciciot2023.json")
out["fingerprint_positive"]={
  "CIC_IIoT2025":{"n_devices":fp["n_devices"],"accuracy":fp["accuracy"],"majority_baseline_acc":fp["majority_baseline_acc"],
    "macro_ovr_auc":fp["macro_ovr_auc"],"per_device_ovr_auc":fp["per_device_ovr_auc"],
    "top_features":[t[0] for t in fp["top_features"]],"gateway_excluded":fp["gateway_excluded"]},
  "CICIoT2023":{"n_devices":fc["n_devices"],"accuracy":fc["accuracy"],"majority_baseline_acc":fc["majority_baseline_acc"],
    "macro_ovr_auc":fc["macro_ovr_auc"],"per_device_ovr_auc":fc["per_device_ovr_auc"],
    "top_features":[t[0] for t in fc["top_features"]],"gateway_excluded":fc["gateway_excluded"]},
  "note":("positive fingerprint test: predict DEVICE from BENIGN packet-2 features (NOT device-disjoint; device is "
          "the label; infrastructure MACs excluded). TWO datasets: CIC IIoT 2025 macro-AUC 0.999 / acc 0.986 vs "
          "0.612 over 8 devices; CICIoT2023 macro-AUC 0.998 / acc 0.925 vs 0.058 over 43 devices. Both fingerprints "
          "are carried by TCP window + TTL + IP length. Demonstrates the channel positively where the ablation null "
          "could only infer it")}

json.dump(out,open(os.path.join(HERE,"paper_numbers.json"),"w"),indent=2)
print("Wrote paper_numbers.json. Key headline numbers:")
print(f"  CICIoT2023 Mirai pkt-2 disjoint AUC: {out['ciciot2023_mirai_pkt2']['disjoint_auc']}")
print(f"  CIC IIoT 2025 pkt-2 udp-frag (26 dev) disjoint AUC: {out['iiot2025_packet2_udpfrag']['victim_disjoint_auc']} FPR {out['iiot2025_packet2_udpfrag']['victim_disjoint_fpr']}")
print(f"  CIC IIoT 2025 pkt-2 Mirai (8 dev) disjoint AUC: {out['iiot2025_packet2_mirai']['victim_disjoint_auc']} FPR {out['iiot2025_packet2_mirai']['victim_disjoint_fpr']}")
print(f"  family sweep: "+", ".join(f"{k} auc_med={v['disjoint_auc_min_med_max'][1] if v['disjoint_auc_min_med_max'] else 'thin'}" for k,v in out['iiot2025_packet2_family_sweep'].items()))
print(f"  pooling origin AUC (want ~0.5): {out['pooling_confound']['origin_auc_final']}")
mm=out['ciciot2023_mitm_dns_pkt2']
print(f"  CICIoT2023 MITM pkt-2: random AUC {mm['MITM_ArpSpoofing']['random_auc']} -> disjoint {mm['MITM_ArpSpoofing']['disjoint_auc_per_fold']} (FPR {mm['MITM_ArpSpoofing']['disjoint_fpr_per_fold']})")
print(f"  CICIoT2023 DNS  pkt-2: random AUC {mm['DNS_Spoofing']['random_auc']} -> disjoint {mm['DNS_Spoofing']['disjoint_auc_per_fold']} (FPR {mm['DNS_Spoofing']['disjoint_fpr_per_fold']})")
