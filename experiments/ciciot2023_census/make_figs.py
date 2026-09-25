#!/usr/bin/env python3
"""
Generate the two paper figures from the committed source-of-truth (paper_numbers.json).
Every plotted point is a number in that file — no value is retyped here beyond axis labels.
Outputs vector PDFs sized for an IEEE conference column (~3.4 in wide) into docs/paper/figs/.

Fig 1  census_sensitivity.pdf  — CICIoT2023: # attack classes with >=3 victims vs the victim-
       concentration threshold; the >=10-class viability bar. Shows viable only at <=2%.
Fig 2  devcount_scatter.pdf     — device-disjoint median AUC vs victim/device count, by verdict.
       Floods climb with device count; benign-opening families stay near chance regardless.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "..", "paper", "figs"))
os.makedirs(OUT, exist_ok=True)
D = json.load(open(os.path.join(HERE, "paper_numbers.json")))
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "figure.dpi": 200})

# ---------- Figure 1: census sensitivity ----------
cur = D["census"]["CICIoT2023_sensitivity_attack_classes_ge3victims"]
xs = [int(k.replace("pct", "")) for k in cur]
ys = [cur[k] for k in cur]
fig, ax = plt.subplots(figsize=(3.4, 1.55))
ax.plot(xs, ys, "o-", color="#1f4e79", lw=1.3, ms=4)
ax.axhline(10, color="#c00000", ls="--", lw=0.9)
ax.text(11.5, 10.4, "viability bar (>=10 classes)", color="#c00000", fontsize=6.5, va="bottom")
for x, y in zip(xs, ys):
    ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 4), fontsize=6.5, ha="center")
ax.set_xlabel("victim-concentration threshold (%)")
ax.set_ylabel("attack classes\nwith >=3 victims")
ax.set_xticks(xs); ax.set_ylim(-1, 19)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "census_sensitivity.pdf")); plt.close(fig)

# ---------- Figure 2: controlled within-dataset device-count sweep ----------
sw = D["iiot2025_devcount_sweep"]
ks = sorted(int(k) for k in sw["auc_by_train_victims"])
auc = [sw["auc_by_train_victims"][str(k)] for k in ks]
fpr = [sw["fpr_by_train_victims"][str(k)] for k in ks]
ciciot_mirai5 = D["ciciot2023_mirai_pkt2"]["disjoint_auc"]["median"]   # 5 victims total (3 train, 2 held out), OTHER dataset

fig, ax = plt.subplots(figsize=(3.4, 1.9))
ax.axhline(0.5, color="#888", ls=":", lw=0.8)
ax.text(ks[0] - 2.5, 0.515, "chance", color="#888", fontsize=6.5, va="bottom", ha="left")
ax.plot(ks, auc, "o-", color="#1f7a1f", lw=1.3, ms=4, label="AUC (ranking), same dataset")
ax.plot(ks, fpr, "s--", color="#c00000", lw=1.0, ms=4, mfc="none", label="FPR at seen threshold")
# cross-dataset contrast: CICIoT2023 Mirai flood at 5 victims
ax.plot([3], [ciciot_mirai5], "*", color="#e07b00", ms=12, zorder=5)
ax.annotate("CICIoT2023 Mirai,\n3 training victims", (3, ciciot_mirai5),
            textcoords="offset points", xytext=(14, -2), fontsize=6.0, va="center", color="#a55500")
ax.set_xlabel("training victim count (fixed held-out test victims)")
ax.set_ylabel("device-disjoint (median)")
ax.set_ylim(0.0, 1.05); ax.set_xlim(2, ks[-1] + 1)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(fontsize=6.2, loc="lower left", bbox_to_anchor=(0.0, 0.06), frameon=False, handletextpad=0.3, borderaxespad=0.2)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "devcount_scatter.pdf")); plt.close(fig)

print("Wrote:")
for f in ("census_sensitivity.pdf", "devcount_scatter.pdf"):
    p = os.path.join(OUT, f); print(f"  {p}  ({os.path.getsize(p)} bytes)")
print("Fig2 sweep AUC:", list(zip(ks, auc)))
print("Fig2 sweep FPR:", list(zip(ks, fpr)))
print("Fig2 CICIoT2023 Mirai contrast (5 victims):", round(ciciot_mirai5, 3))
