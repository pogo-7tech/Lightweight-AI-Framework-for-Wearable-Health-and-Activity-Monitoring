"""
Week 2 | Step 2 of 4  — REAL WESAD DATA
=========================================
PPG Exploratory Data Analysis (EDA)

Loads real BVP from WESAD .pkl files for all 15 subjects,
cleans each signal, and compares raw vs. cleaned quality.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings("ignore")

OUT      = Path("notebooks/week2_figures")
OUT.mkdir(parents=True, exist_ok=True)
BG       = "#0d1117"
WESAD    = Path("data/WESAD/WESAD")
FS       = 64
LABEL_MAP = {1: "Baseline", 2: "Stress", 3: "Amusement"}
CLRS     = {"Baseline": "#58a6ff", "Stress": "#f85149", "Amusement": "#3fb950"}

def load_subject(sid):
    with open(WESAD / sid / f"{sid}.pkl", "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    bvp    = raw["signal"]["wrist"]["BVP"].flatten()
    labels = raw["label"]
    ratio  = len(labels) / len(bvp)
    bvp_labels = labels[::int(ratio)][:len(bvp)]
    return bvp, bvp_labels

def bandpass(sig, lo=0.5, hi=8.0, fs=64, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo/nyq, hi/nyq], btype="band")
    return filtfilt(b, a, sig)

def detrend_poly(sig, deg=3):
    x = np.arange(len(sig))
    return sig - np.polyval(np.polyfit(x, sig, deg), x)

def snr(clean, noisy):
    noise = noisy - clean
    ps    = np.mean(clean**2)
    pn    = np.mean(noise**2) + 1e-12
    return 10 * np.log10(ps / pn)

SUBJECTS = sorted([d.name for d in WESAD.iterdir()
                   if d.is_dir() and d.name.startswith("S")])

# ─────────────────────────────────────────────────────────────────────────────
# Compute amplitude + noise stats for all subjects × 3 conditions
# ─────────────────────────────────────────────────────────────────────────────
print("Computing per-subject EDA metrics on real WESAD BVP...")
stats = {}
for sid in SUBJECTS:
    bvp, labs = load_subject(sid)
    filt  = bandpass(bvp)
    clean = detrend_poly(filt)
    sub   = {}
    for lbl, name in LABEL_MAP.items():
        mask = (labs == lbl)
        if mask.sum() < FS * 10:
            sub[name] = {"amp": 0, "snr": 0}
            continue
        seg_raw   = bvp[mask]
        seg_clean = clean[mask]
        sub[name] = {
            "amp": float(np.std(seg_clean)),
            "snr": float(snr(seg_clean, seg_raw))
        }
    stats[sid] = sub
    print(f"  {sid}: Baseline amp={sub['Baseline']['amp']:.4f}  "
          f"Stress amp={sub['Stress']['amp']:.4f}  "
          f"SNR gain={sub['Stress']['snr']:.1f}dB")

# ─────────────────────────────────────────────────────────────────────────────
# Figure A — BVP amplitude per subject per condition
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=BG)
fig.suptitle("Week 2 | Real WESAD EDA — BVP Signal Quality across 15 Subjects",
             color="white", fontsize=12, fontweight="bold")

ax = axes[0]
ax.set_facecolor(BG); ax.tick_params(colors="#8b949e", labelsize=8)
[s.set_edgecolor("#21262d") for s in ax.spines.values()]
ax.grid(color="#21262d", linewidth=0.5, axis="y")

x = np.arange(len(SUBJECTS))
w = 0.28
for i, (cond, clr) in enumerate(CLRS.items()):
    amps = [stats[s][cond]["amp"] for s in SUBJECTS]
    ax.bar(x + (i-1)*w, amps, w, label=cond, color=clr, alpha=0.85)

ax.set_xticks(x); ax.set_xticklabels(SUBJECTS, rotation=45, color="#8b949e")
ax.set_ylabel("BVP Std (amplitude)", color="#8b949e")
ax.set_title("BVP Amplitude per Subject × Condition", color="#c9d1d9", fontsize=10)
ax.legend(framealpha=0.2, fontsize=9)

# Figure B — Raw vs Cleaned (S2, Stress segment, first 10 seconds)
ax = axes[1]
ax.set_facecolor(BG); ax.tick_params(colors="#8b949e", labelsize=8)
[s.set_edgecolor("#21262d") for s in ax.spines.values()]
ax.grid(color="#21262d", linewidth=0.5)

bvp_s2, labs_s2 = load_subject("S2")
stress_mask = labs_s2 == 2
seg_raw   = bvp_s2[stress_mask][:FS*10]
seg_filt  = bandpass(seg_raw)
seg_clean = detrend_poly(seg_filt)
t10 = np.linspace(0, 10, len(seg_raw))
ax.plot(t10, seg_raw,   color="#f85149", linewidth=0.8, alpha=0.7, label="Raw BVP")
ax.plot(t10, seg_clean, color="#3fb950", linewidth=1.2, label="Cleaned BVP")
ax.set_xlabel("Time (seconds)", color="#8b949e")
ax.set_ylabel("BVP Amplitude", color="#8b949e")
ax.set_title("S2 — Stress Segment: Raw vs. Cleaned (10 sec)", color="#c9d1d9", fontsize=10)
ax.legend(framealpha=0.2, fontsize=9)

plt.tight_layout()
plt.savefig(OUT / "W2_02a_eda_amplitude.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\n✅ Saved: W2_02a_eda_amplitude.png")
