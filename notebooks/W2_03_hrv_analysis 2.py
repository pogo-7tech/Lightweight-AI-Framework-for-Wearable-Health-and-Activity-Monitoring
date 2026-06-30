"""
Week 2 | Step 3 of 4  — REAL WESAD DATA
=========================================
HRV Feature Extraction & Analysis

Extracts real HRV features from actual WESAD BVP signals:
  - Heart Rate (bpm)
  - RMSSD (ms) — parasympathetic index
  - SDNN (ms)  — overall HRV
  - pNN50 (%)  — stress marker
  - Mean RR interval (ms)

Compares Baseline vs Stress vs Amusement across 15 real subjects.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy.signal import butter, filtfilt, find_peaks
import warnings
warnings.filterwarnings("ignore")

OUT   = Path("notebooks/week2_figures")
OUT.mkdir(parents=True, exist_ok=True)
BG    = "#0d1117"
WESAD = Path("data/WESAD/WESAD")
FS    = 64
LABEL_MAP = {1: "Baseline", 2: "Stress", 3: "Amusement"}
CLRS      = {"Baseline": "#58a6ff", "Stress": "#f85149", "Amusement": "#3fb950"}

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def load_subject(sid):
    with open(WESAD / sid / f"{sid}.pkl", "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    bvp    = raw["signal"]["wrist"]["BVP"].flatten()
    labels = raw["label"]
    ratio  = len(labels) / len(bvp)
    return bvp, labels[::int(ratio)][:len(bvp)]

def clean_bvp(sig, fs=64):
    nyq  = fs / 2
    b, a = butter(4, [0.5/nyq, 8.0/nyq], btype="band")
    filt = filtfilt(b, a, sig)
    x    = np.arange(len(filt))
    return filt - np.polyval(np.polyfit(x, filt, 3), x)

def extract_hrv(seg, fs=64):
    """Extract HRV features from a clean BVP segment."""
    if len(seg) < fs * 10:
        return None
    peaks, _ = find_peaks(seg, distance=int(fs * 0.4), height=np.std(seg) * 0.3)
    if len(peaks) < 5:
        return None
    rr = np.diff(peaks) / fs * 1000    # ms
    rr = rr[(rr > 300) & (rr < 1500)] # physiological range filter
    if len(rr) < 4:
        return None
    rr_diff = np.diff(rr)
    return {
        "HR":    60000 / np.mean(rr),
        "RMSSD": np.sqrt(np.mean(rr_diff**2)),
        "SDNN":  np.std(rr),
        "pNN50": 100 * np.sum(np.abs(rr_diff) > 50) / len(rr_diff),
        "MeanRR": np.mean(rr),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Extract HRV for all subjects × conditions
# ─────────────────────────────────────────────────────────────────────────────
SUBJECTS = sorted([d.name for d in WESAD.iterdir()
                   if d.is_dir() and d.name.startswith("S")])

results = {cond: {feat: [] for feat in ["HR","RMSSD","SDNN","pNN50","MeanRR"]}
           for cond in LABEL_MAP.values()}

valid_subjects = []
print("="*65)
print(f"  {'Subject':<8} {'HR_base':>8} {'HR_stress':>10} {'RMSSD_base':>11} {'RMSSD_stress':>13}")
print("  " + "-"*63)

for sid in SUBJECTS:
    bvp, labs = load_subject(sid)
    clean     = clean_bvp(bvp)
    sub_ok    = True
    row = {}
    for lbl, name in LABEL_MAP.items():
        mask = (labs == lbl)
        if mask.sum() < FS * 30:
            sub_ok = False; break
        feats = extract_hrv(clean[mask])
        if feats is None:
            sub_ok = False; break
        row[name] = feats

    if sub_ok:
        valid_subjects.append(sid)
        for name, feats in row.items():
            for k, v in feats.items():
                results[name][k].append(v)
        print(f"  {sid:<8} {row['Baseline']['HR']:>8.1f} {row['Stress']['HR']:>10.1f} "
              f"{row['Baseline']['RMSSD']:>11.1f} {row['Stress']['RMSSD']:>13.1f}")

print("="*65)
print(f"\n  ✅ Valid subjects: {len(valid_subjects)} — {valid_subjects}")
print("\n  ── Group Means ──")
for cond in LABEL_MAP.values():
    hr = np.mean(results[cond]["HR"])
    rm = np.mean(results[cond]["RMSSD"])
    sd = np.mean(results[cond]["SDNN"])
    pn = np.mean(results[cond]["pNN50"])
    print(f"  {cond:>10}: HR={hr:.1f} bpm | RMSSD={rm:.1f} ms | SDNN={sd:.1f} ms | pNN50={pn:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# Figure A — HRV boxplot (Baseline vs Stress vs Amusement)
# ─────────────────────────────────────────────────────────────────────────────
features = ["HR", "RMSSD", "SDNN", "pNN50"]
flabels  = ["Heart Rate (bpm)", "RMSSD (ms)", "SDNN (ms)", "pNN50 (%)"]
conds    = list(LABEL_MAP.values())

fig, axes = plt.subplots(1, 4, figsize=(18, 6), facecolor=BG)
fig.suptitle(f"Week 2 | Real WESAD HRV Analysis — {len(valid_subjects)} Subjects",
             color="white", fontsize=12, fontweight="bold")

for ax, feat, fl in zip(axes, features, flabels):
    ax.set_facecolor(BG); ax.tick_params(colors="#8b949e", labelsize=9)
    [s.set_edgecolor("#21262d") for s in ax.spines.values()]
    ax.grid(color="#21262d", linewidth=0.5, axis="y")

    data   = [results[c][feat] for c in conds]
    colors = [CLRS[c] for c in conds]
    bp = ax.boxplot(data, patch_artist=True,
                    medianprops={"color":"white","linewidth":2},
                    whiskerprops={"color":"#8b949e"},
                    capprops={"color":"#8b949e"},
                    flierprops={"markerfacecolor":"#8b949e","marker":"o","markersize":4})
    for patch, clr in zip(bp["boxes"], colors):
        patch.set_facecolor(clr); patch.set_alpha(0.75)
    ax.set_xticklabels(conds, color="#8b949e", rotation=15)
    ax.set_title(fl, color="#c9d1d9", fontsize=9)

    # Annotate medians
    for i, d in enumerate(data):
        ax.text(i+1, np.median(d), f" {np.median(d):.1f}",
                color="white", fontsize=8, va="center")

plt.tight_layout()
plt.savefig(OUT / "W2_03a_hrv_boxplot.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\n✅ Saved: W2_03a_hrv_boxplot.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure B — Mean HRV bar chart per condition
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle("Week 2 | HRV Mean Comparison — Baseline vs Stress vs Amusement",
             color="white", fontsize=12, fontweight="bold")

# HR bar
ax = axes[0]
ax.set_facecolor(BG); ax.tick_params(colors="#8b949e")
[s.set_edgecolor("#21262d") for s in ax.spines.values()]
ax.grid(color="#21262d", linewidth=0.5, axis="y")
hr_means = [np.mean(results[c]["HR"]) for c in conds]
hr_stds  = [np.std(results[c]["HR"])  for c in conds]
bars = ax.bar(conds, hr_means, color=[CLRS[c] for c in conds],
              alpha=0.85, yerr=hr_stds, capsize=5,
              error_kw={"color":"white","linewidth":1.5})
for bar, val in zip(bars, hr_means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f"{val:.1f}", ha="center", color="white", fontsize=11, fontweight="bold")
ax.set_ylabel("Heart Rate (bpm)", color="#8b949e")
ax.set_title("Mean Heart Rate ± SD", color="#c9d1d9", fontsize=10)

# RMSSD bar
ax = axes[1]
ax.set_facecolor(BG); ax.tick_params(colors="#8b949e")
[s.set_edgecolor("#21262d") for s in ax.spines.values()]
ax.grid(color="#21262d", linewidth=0.5, axis="y")
rm_means = [np.mean(results[c]["RMSSD"]) for c in conds]
rm_stds  = [np.std(results[c]["RMSSD"])  for c in conds]
bars = ax.bar(conds, rm_means, color=[CLRS[c] for c in conds],
              alpha=0.85, yerr=rm_stds, capsize=5,
              error_kw={"color":"white","linewidth":1.5})
for bar, val in zip(bars, rm_means):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            f"{val:.1f}", ha="center", color="white", fontsize=11, fontweight="bold")
ax.set_ylabel("RMSSD (ms)", color="#8b949e")
ax.set_title("Mean RMSSD ± SD  (lower = more stressed)", color="#c9d1d9", fontsize=10)

plt.tight_layout()
plt.savefig(OUT / "W2_03b_hrv_bar.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅ Saved: W2_03b_hrv_bar.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure C — RMSSD per subject scatter
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG)
ax.set_facecolor(BG); ax.tick_params(colors="#8b949e", labelsize=8)
[s.set_edgecolor("#21262d") for s in ax.spines.values()]
ax.grid(color="#21262d", linewidth=0.5, axis="y")
x = np.arange(len(valid_subjects))
for i, (cond, clr) in enumerate(CLRS.items()):
    vals = results[cond]["RMSSD"]
    ax.scatter(x + (i-1)*0.25, vals, color=clr, s=80,
               label=cond, zorder=5, alpha=0.9)
    ax.plot(x + (i-1)*0.25, vals, color=clr, linewidth=0.8, alpha=0.4)

ax.set_xticks(x); ax.set_xticklabels(valid_subjects, color="#8b949e")
ax.set_ylabel("RMSSD (ms)", color="#8b949e")
ax.set_title("RMSSD per Subject — Real WESAD Data (lower RMSSD = more stressed)",
             color="#c9d1d9", fontsize=10)
ax.legend(framealpha=0.2, fontsize=9)
fig.suptitle("Week 2 | Per-Subject HRV: Baseline vs Stress vs Amusement",
             color="white", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT / "W2_03c_rmssd_per_subject.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅ Saved: W2_03c_rmssd_per_subject.png")
print(f"\n✅ HRV analysis complete on {len(valid_subjects)} real WESAD subjects!")
