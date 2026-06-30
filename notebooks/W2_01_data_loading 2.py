"""
Week 2 | Step 1 of 4  — UPDATED WITH REAL WESAD DATA
=======================================================
Data Loading & Structure

Loads REAL WESAD dataset (15 subjects, 3 conditions):
  - BVP (Blood Volume Pulse = PPG) from wrist Empatica E4 at 64 Hz
  - Labels: 0=transient, 1=baseline, 2=stress, 3=amusement
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUT = Path("notebooks/week2_figures")
OUT.mkdir(parents=True, exist_ok=True)
BG = "#0d1117"

WESAD_DIR = Path("data/WESAD/WESAD")
FS_BVP    = 64          # Hz — Empatica E4 BVP sample rate
LABEL_MAP = {1: "Baseline", 2: "Stress", 3: "Amusement"}

# ─────────────────────────────────────────────────────────────────────────────
# Load real WESAD subjects
# ─────────────────────────────────────────────────────────────────────────────
SUBJECTS = sorted([d.name for d in WESAD_DIR.iterdir()
                   if d.is_dir() and d.name.startswith("S")])

print("="*60)
print("  REAL WESAD Dataset — Loading Summary")
print("="*60)
print(f"  Dataset path : {WESAD_DIR}")
print(f"  Subjects found: {len(SUBJECTS)} — {SUBJECTS}")
print(f"  BVP sampling rate: {FS_BVP} Hz")
print(f"  Conditions: Baseline(1), Stress(2), Amusement(3)")
print("="*60)

dataset = {}
for sid in SUBJECTS:
    pkl_path = WESAD_DIR / sid / f"{sid}.pkl"
    if not pkl_path.exists():
        print(f"  ⚠️ {sid}: .pkl not found, skipping")
        continue
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    bvp    = raw["signal"]["wrist"]["BVP"].flatten()  # (N,)
    labels = raw["label"]                              # sampled at 700 Hz
    # Downsample labels to BVP rate (64 Hz): label array is at 700 Hz
    label_ratio = len(labels) / len(bvp)
    bvp_labels  = labels[::int(label_ratio)][:len(bvp)]
    dataset[sid] = {"bvp": bvp, "labels": bvp_labels, "fs": FS_BVP}
    dur_min = len(bvp) / FS_BVP / 60
    print(f"  {sid}: {len(bvp):>7,} BVP samples  ({dur_min:.1f} min)")

print("="*60)
print(f"  ✅ Loaded {len(dataset)} subjects successfully")

# ─────────────────────────────────────────────────────────────────────────────
# Per-condition segment statistics (subject S2 example)
# ─────────────────────────────────────────────────────────────────────────────
s2     = dataset["S2"]
bvp    = s2["bvp"]
labels = s2["labels"]

print("\n  Subject S2 — Segment lengths per condition:")
for lbl, name in LABEL_MAP.items():
    mask = labels == lbl
    secs = mask.sum() / FS_BVP
    print(f"    {name:>10}: {secs:.0f} sec  ({mask.sum():,} samples)")

# ─────────────────────────────────────────────────────────────────────────────
# Figure: Dataset overview — all subjects, BVP duration + condition breakdown
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle("Week 2 | Real WESAD Dataset — Data Loading Overview",
             color="white", fontsize=12, fontweight="bold")

clrs = {"Baseline": "#58a6ff", "Stress": "#f85149", "Amusement": "#3fb950"}

# Left — total BVP duration per subject
ax = axes[0]
ax.set_facecolor(BG)
[s.set_edgecolor("#21262d") for s in ax.spines.values()]
ax.grid(color="#21262d", linewidth=0.5, axis="y")
durations = [len(dataset[s]["bvp"]) / FS_BVP / 60 for s in dataset]
bars = ax.bar(list(dataset.keys()), durations, color="#58a6ff", alpha=0.85)
ax.set_xlabel("Subject", color="#8b949e")
ax.set_ylabel("Duration (minutes)", color="#8b949e")
ax.set_title("BVP Recording Duration per Subject", color="#c9d1d9", fontsize=10)
ax.tick_params(colors="#8b949e", labelsize=8)
for bar, val in zip(bars, durations):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            f"{val:.0f}m", ha="center", color="white", fontsize=7)

# Right — condition breakdown for S2
ax = axes[1]
ax.set_facecolor(BG)
[s.set_edgecolor("#21262d") for s in ax.spines.values()]
cond_secs = {name: (s2["labels"] == lbl).sum() / FS_BVP
             for lbl, name in LABEL_MAP.items()}
ax.barh(list(cond_secs.keys()), list(cond_secs.values()),
        color=list(clrs.values()), alpha=0.85)
ax.set_xlabel("Duration (seconds)", color="#8b949e")
ax.set_title("Subject S2 — Condition Duration Breakdown", color="#c9d1d9", fontsize=10)
ax.tick_params(colors="#8b949e")
ax.grid(color="#21262d", linewidth=0.5, axis="x")

plt.tight_layout()
plt.savefig(OUT / "W2_01_data_loading.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\n✅ Saved: W2_01_data_loading.png")
print(f"✅ WESAD real data loaded: {len(dataset)} subjects, BVP @ {FS_BVP} Hz")
