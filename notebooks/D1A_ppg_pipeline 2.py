"""
Week 1 | Deliverable 1A
========================
PPG / Respiration Preprocessing Pipeline

Steps demonstrated:
  1. Raw PPG signal (64 Hz) with noise + baseline wander
  2. Bandpass filter (0.5 – 8 Hz)
  3. Polynomial detrending (baseline removal)
  4. Peak detection + HRV feature extraction
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# ── Config ────────────────────────────────────────────────────────────────────
FS   = 64       # Sampling rate (Hz) — WESAD E4 wristband
BPM  = 72       # Heart rate to simulate
DUR  = 30       # Duration (seconds)
SEED = 42
OUT  = "notebooks/week1_figures/figA_ppg_pipeline.png"

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_ppg(fs, bpm, dur, noise_std=0.06, seed=42):
    np.random.seed(seed)
    t = np.linspace(0, dur, int(fs * dur))
    f = bpm / 60
    sig = (0.8 * np.sin(2 * np.pi * f * t)
           + 0.25 * np.sin(4 * np.pi * f * t)
           + 0.05 * np.sin(2 * np.pi * 0.15 * t)   # baseline wander
           + noise_std * np.random.randn(len(t)))
    return t, sig

def bandpass(sig, lo, hi, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)

def detrend_poly(sig, order=3):
    x = np.arange(len(sig))
    return sig - np.polyval(np.polyfit(x, sig, order), x)

def style(ax):
    BG = "#0d1117"
    ax.set_facecolor(BG)
    ax.tick_params(colors="#8b949e")
    [s.set_edgecolor("#21262d") for s in ax.spines.values()]
    ax.grid(color="#21262d", linewidth=0.5)

# ── Pipeline ──────────────────────────────────────────────────────────────────
t, raw   = make_ppg(FS, BPM, DUR)
filtered = bandpass(raw, 0.5, 8.0, FS)
clean    = detrend_poly(filtered.copy())
peaks, _ = find_peaks(clean, distance=int(FS * 0.5), height=0.05)

# HRV features
rr      = np.diff(peaks) / FS * 1000          # RR intervals in ms
hr_bpm  = 60000 / np.mean(rr)
rmssd   = np.sqrt(np.mean(np.diff(rr) ** 2))
sdnn    = np.std(rr)
pnn50   = 100 * np.sum(np.abs(np.diff(rr)) > 50) / len(rr)

print(f"\n HRV Features:")
print(f"  Heart Rate : {hr_bpm:.1f} bpm")
print(f"  RMSSD      : {rmssd:.1f} ms")
print(f"  SDNN       : {sdnn:.1f} ms")
print(f"  pNN50      : {pnn50:.1f} %")

# ── Plot ──────────────────────────────────────────────────────────────────────
BG = "#0d1117"
fig, axes = plt.subplots(4, 1, figsize=(14, 10), facecolor=BG)
fig.suptitle("Deliverable 1A — PPG Preprocessing Pipeline",
             color="white", fontsize=14, fontweight="bold")

# Stage 1 — Raw
style(axes[0])
axes[0].plot(t, raw, color="#58a6ff", linewidth=0.9)
axes[0].set_title("① Raw PPG (64 Hz) — noise + baseline wander visible",
                  color="#c9d1d9", fontsize=10)
axes[0].set_ylabel("Amplitude", color="#8b949e")

# Stage 2 — Bandpass
style(axes[1])
axes[1].plot(t, filtered, color="#f59e0b", linewidth=0.9)
axes[1].set_title("② After Bandpass Filter (0.5–8 Hz) — HF noise removed",
                  color="#c9d1d9", fontsize=10)
axes[1].set_ylabel("Amplitude", color="#8b949e")

# Stage 3 — Detrend
style(axes[2])
axes[2].plot(t, clean, color="#3fb950", linewidth=0.9)
axes[2].set_title("③ After Polynomial Detrend — flat baseline",
                  color="#c9d1d9", fontsize=10)
axes[2].set_ylabel("Amplitude", color="#8b949e")

# Stage 4 — Peaks + HRV
style(axes[3])
axes[3].plot(t, clean, color="#3fb950", linewidth=1.0, label="Clean PPG")
axes[3].scatter(t[peaks], clean[peaks], color="#f85149", s=50, zorder=5,
                label=f"Peaks  |  HR = {hr_bpm:.0f} bpm")
axes[3].set_title(
    f"④ Peak Detection + HRV: HR={hr_bpm:.0f} bpm  RMSSD={rmssd:.1f} ms  "
    f"SDNN={sdnn:.1f} ms  pNN50={pnn50:.1f}%",
    color="#c9d1d9", fontsize=10)
axes[3].set_ylabel("Amplitude", color="#8b949e")
axes[3].set_xlabel("Time (s)", color="#8b949e")
axes[3].legend(framealpha=0.2, fontsize=9)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
# plt.show()  # uncomment to display interactively
print(f"\n✅ Figure saved → {OUT}")
