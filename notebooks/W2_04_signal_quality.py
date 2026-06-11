"""
Week 2 | Step 4 of 4  — REAL WESAD DATA
=========================================
Signal Quality & Missing Data Report

Analyses real WESAD BVP for dropout events, signal quality
scores, and per-subject reconstruction quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings("ignore")

OUT   = Path("notebooks/week2_figures")
OUT.mkdir(parents=True, exist_ok=True)
BG    = "#0d1117"
WESAD = Path("data/WESAD/WESAD")
FS    = 64
DROPOUT_THRESH = 1e-4   # std below this = flat-line dropout
DROPOUT_WIN    = FS      # 1-second rolling window

def load_bvp(sid):
    with open(WESAD / sid / f"{sid}.pkl", "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    return raw["signal"]["wrist"]["BVP"].flatten()

def detect_dropouts(sig, win=FS, thresh=DROPOUT_THRESH):
    """Returns boolean mask where signal has dropped out."""
    mask = np.zeros(len(sig), dtype=bool)
    for i in range(0, len(sig) - win, win // 2):
        seg = sig[i:i+win]
        if np.std(seg) < thresh:
            mask[i:i+win] = True
    return mask

def cubic_spline_reconstruct(sig, mask):
    """Reconstruct dropout gaps using cubic spline interpolation."""
    from scipy.interpolate import CubicSpline
    idx   = np.arange(len(sig))
    good  = ~mask
    if good.sum() < 10:
        return sig.copy()
    cs = CubicSpline(idx[good], sig[good])
    out = sig.copy()
    out[mask] = cs(idx[mask])
    return out

SUBJECTS = sorted([d.name for d in WESAD.iterdir()
                   if d.is_dir() and d.name.startswith("S")])

# ─────────────────────────────────────────────────────────────────────────────
# Per-subject dropout analysis
# ─────────────────────────────────────────────────────────────────────────────
print("="*65)
print(f"  {'Subject':<8} {'Samples':>9} {'Dropouts':>10} {'Dropout%':>9} {'Quality':>9}")
print("  " + "-"*63)

quality_scores = []
dropout_pcts   = []
for sid in SUBJECTS:
    bvp  = load_bvp(sid)
    mask = detect_dropouts(bvp)
    pct  = 100 * mask.sum() / len(bvp)
    qual = "GOOD" if pct < 1 else ("FAIR" if pct < 5 else "POOR")
    quality_scores.append(qual)
    dropout_pcts.append(pct)
    print(f"  {sid:<8} {len(bvp):>9,} {mask.sum():>10,} {pct:>8.2f}% {qual:>9}")

print("="*65)
good_n = quality_scores.count("GOOD")
fair_n = quality_scores.count("FAIR")
poor_n = quality_scores.count("POOR")
print(f"\n  GOOD: {good_n}  |  FAIR: {fair_n}  |  POOR: {poor_n}")

# ─────────────────────────────────────────────────────────────────────────────
# Figure A — Dropout rate per subject + quality flag
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5), facecolor=BG)
fig.suptitle("Week 2 | Real WESAD Signal Quality Report",
             color="white", fontsize=12, fontweight="bold")

q_clrs = {"GOOD": "#3fb950", "FAIR": "#f59e0b", "POOR": "#f85149"}
bar_colors = [q_clrs[q] for q in quality_scores]

ax = axes[0]
ax.set_facecolor(BG); ax.tick_params(colors="#8b949e", labelsize=8)
[s.set_edgecolor("#21262d") for s in ax.spines.values()]
ax.grid(color="#21262d", linewidth=0.5, axis="y")
bars = ax.bar(SUBJECTS, dropout_pcts, color=bar_colors, alpha=0.85)
ax.axhline(1.0, color="#f59e0b", linestyle="--", linewidth=1.5, label="Fair threshold (1%)")
ax.axhline(5.0, color="#f85149", linestyle="--", linewidth=1.5, label="Poor threshold (5%)")
ax.set_ylabel("Dropout Rate (%)", color="#8b949e")
ax.set_title("Signal Dropout Rate per Subject", color="#c9d1d9", fontsize=10)
ax.tick_params(axis="x", rotation=45)
ax.legend(framealpha=0.2, fontsize=8)
# Quality labels
for bar, q in zip(bars, quality_scores):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
            q[0], ha="center", color="white", fontsize=8, fontweight="bold")

# Figure B — S9 reconstruction example (worst subject)
ax = axes[1]
ax.set_facecolor(BG); ax.tick_params(colors="#8b949e", labelsize=8)
[s.set_edgecolor("#21262d") for s in ax.spines.values()]
ax.grid(color="#21262d", linewidth=0.5)

# Use subject with highest dropout for demo
worst_idx = int(np.argmax(dropout_pcts))
worst_sid = SUBJECTS[worst_idx]
bvp_w = load_bvp(worst_sid)[:FS*20]   # first 20 seconds

# Inject artificial dropout for demonstration (real dropouts may be subtle)
mask_w = np.zeros(len(bvp_w), dtype=bool)
mask_w[FS*6:FS*9]  = True    # inject 3-sec gap
mask_w[FS*14:FS*16] = True   # inject 2-sec gap
bvp_drop  = bvp_w.copy()
bvp_drop[mask_w] = np.mean(bvp_w)   # simulate flat-line

recon     = cubic_spline_reconstruct(bvp_drop, mask_w)
t20       = np.arange(len(bvp_w)) / FS

ax.plot(t20, bvp_w,    color="#3fb950",  linewidth=1.0, label="Original",    alpha=0.6)
ax.plot(t20, bvp_drop, color="#f85149",  linewidth=0.9, label="With dropout", alpha=0.7)
ax.plot(t20, recon,    color="#f59e0b",  linewidth=1.2, label="Reconstructed",linestyle="--")
ax.axvspan(6, 9,   color="#f85149", alpha=0.08)
ax.axvspan(14, 16, color="#f85149", alpha=0.08)
ax.set_xlabel("Time (seconds)", color="#8b949e")
ax.set_ylabel("BVP Amplitude",  color="#8b949e")
ax.set_title(f"Dropout Detection + Spline Reconstruction ({worst_sid})",
             color="#c9d1d9", fontsize=10)
ax.legend(framealpha=0.2, fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "W2_04_signal_quality.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\n✅ Saved: W2_04_signal_quality.png")
print("✅ Signal quality report complete on all 15 real WESAD subjects!")
