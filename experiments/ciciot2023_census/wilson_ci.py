#!/usr/bin/env python3
"""
Wilson 95% score intervals on the device-disjoint FPR (reviewer §3.4). Computed from the COMMITTED
result JSONs — no SSD, no re-run. Each disjoint fold stores n_test, test_attack_frac, fpr_at_thr; the
benign denominator is benign_N = round(n_test*(1-test_attack_frac)) and FP = round(fpr*benign_N). We
report the Wilson interval (bounded to [0,1], correct near the p->1 boundary where these FPRs sit,
unlike the Wald approximation) so the operating-point failure is shown to be far outside sampling noise.
"""
import json, os, math

HERE = os.path.dirname(os.path.abspath(__file__))
Z = 1.959963984540054  # 95%

def wilson(k, n, z=Z):
    if n == 0: return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = (z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def fold_ci(n_test, attack_frac, fpr):
    benign_n = round(n_test * (1 - attack_frac))
    fp = round(fpr * benign_n)
    lo, hi = wilson(fp, benign_n)
    return {"benign_n": benign_n, "fp": fp, "fpr": round(fpr, 4),
            "wilson95": [round(lo, 4), round(hi, 4)]}

out = {"note": "Wilson 95% score interval on device-disjoint FPR; benign_n = n_test*(1-test_attack_frac), "
               "FP = fpr*benign_n; from committed JSONs, no re-run.", "families": {}}

# MITM + DNS (leave-one-victim-out)
md = json.load(open(os.path.join(HERE, "mitm_dns_disjoint_result.json")))
for fam, blob in md["results"].items():
    folds = []
    for f in blob["leave_one_victim_out"]:
        ci = fold_ci(f["n_test"], f["test_attack_frac"], f["fpr_at_thr"])
        ci["held_out_victim"] = f["held_out_victim"]; ci["auc"] = f["auc"]
        folds.append(ci)
    out["families"][fam] = folds

# Mirai (2-victim holdout partitions)
mi = json.load(open(os.path.join(HERE, "mirai_disjoint_result.json")))
folds = []
for f in mi["disjoint_partitions"]:
    ci = fold_ci(f["n_test"], f["test_attack_frac"], f["fpr_at_thr"])
    ci["test_victims"] = f.get("test_victims"); ci["auc"] = f["auc"]
    folds.append(ci)
out["families"]["Mirai_flood"] = folds

json.dump(out, open(os.path.join(HERE, "wilson_ci.json"), "w"), indent=2)
for fam, folds in out["families"].items():
    print(f"\n{fam}:")
    for f in folds:
        who = f.get("held_out_victim") or f.get("test_victims")
        print(f"  FPR {f['fpr']:.3f}  Wilson95 [{f['wilson95'][0]:.3f}, {f['wilson95'][1]:.3f}]  "
              f"(FP {f['fp']}/{f['benign_n']} benign)  AUC {f['auc']}  {who}")
print("\nWrote wilson_ci.json")
