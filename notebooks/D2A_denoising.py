"""
Week 1 | Deliverable 2A
========================
Signal Denoising

Demonstrates removal of 3 common noise types in wearable PPG signals:
  1. Gaussian white noise
  2. Baseline wander (slow drift ~0.12 Hz)
  3. Powerline interference (50 Hz)

Metric: SNR improvement (dB) before vs after filtering.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ── Config ────────────────────────────────────────────────────────────────────
FS   = 64
OUT  = "notebooks/week1_figures/figC_denoising.png"

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_clean_ppg(fs=64, dur=5, bpm=72, seed=1):
    np.random.seed(seed)
    t = np.linspace(0, dur, int(fs * dur))
    f = bpm / 60
    sig = (0.8 * np.sin(2 * np.pi * f * t)
           + 0.25 * np.sin(4 * np.pi * f * t))
    # Detrend
    sig -= np.polyval(np.polyfit(np.arange(len(sig)), sig, 3), np.arange(len(sig)))
    return t, sig

def bandpass(sig, lo, hi, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)

def snr_db(clean, noisy):
    noise = noisy - clean
    return 10 * np.log10(np.mean(clean ** 2) / (np.mean(noise ** 2) + 1e-12))

def style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    [s.set_edgecolor("#21262d") for s in ax.spines.values()]
    ax.grid(color="#21262d", linewidth=0.5)

# ── Generate signals ──────────────────────────────────────────────────────────
np.random.seed(1)
t, clean = make_clean_ppg(FS)

# Noise types
gaussian  = clean + 0.4  * np.random.randn(len(clean))
wander    = clean + 0.3  * np.sin(2 * np.pi * 0.12 * t)
powerline = clean + 0.08 * np.sin(2 * np.pi * 50.0 * t)

# Denoised (bandpass filter removes all three)
den_gaussian  = bandpass(gaussian,  0.5, 8.0, FS)
den_wander    = bandpass(wander,    0.5, 8.0, FS)
den_powerline = bandpass(powerline, 0.5, 8.0, FS)

# ── Results ───────────────────────────────────────────────────────────────────
noise_types = [
    ("Gaussian White Noise",     "#f85149",  gaussian,  den_gaussian),
    ("Baseline Wander (0.12 Hz)","#f59e0b",  wander,    den_wander),
    ("Powerline Interference (50 Hz)", "#a371f7", powerline, den_powerline),
]

print("\n Denoising Results:")
print(f"  {'Type':<30} {'SNR Before':>12} {'SNR After':>10} {'Improvement':>12}")
print("  " + "-" * 66)
for name, _, noisy, denoised in noise_types:
    before = snr_db(clean, noisy)
    after  = snr_db(clean, denoised)
    print(f"  {name:<30} {before:>10.1f}dB {after:>9.1f}dB {after-before:>+11.1f}dB")

# ── Plot ──────────────────────────────────────────────────────────────────────
BG  = "#0d1117"
fig, axes = plt.subplots(3, 2, figsize=(14, 9), facecolor=BG)
fig.suptitle("Deliverable 2A — Signal Denoising (3 Noise Types)",
             color="white", fontsize=14, fontweight="bold")

for row, (name, clr, noisy, denoised) in enumerate(noise_types):
    before = snr_db(clean, noisy)
    after  = snr_db(clean, denoised)

    style(axes[row, 0])
    axes[row, 0].plot(t, noisy,  color=clr,     linewidth=0.9, label="Noisy")
    axes[row, 0].plot(t, clean,  color="white", linewidth=0.7, alpha=0.35,
                      linestyle="--", label="Ground truth")
    axes[row, 0].set_title(f"{name} — Before  (SNR = {before:.1f} dB)",
                           color=clr, fontsize=10)
    axes[row, 0].set_ylabel("Amplitude", color="#8b949e", fontsize=8)
    axes[row, 0].legend(fontsize=8, framealpha=0.2)

    style(axes[row, 1])
    axes[row, 1].plot(t, denoised, color="#3fb950", linewidth=1.0, label="Denoised")
    axes[row, 1].plot(t, clean,    color="white",   linewidth=0.7, alpha=0.35,
                      linestyle="--", label="Ground truth")
    axes[row, 1].set_title(
        f"After Denoising  (SNR = {after:.1f} dB  |  +{after-before:.1f} dB)",
        color="#3fb950", fontsize=10)
    axes[row, 1].legend(fontsize=8, framealpha=0.2)

    if row == 2:
        axes[row, 0].set_xlabel("Time (s)", color="#8b949e")
        axes[row, 1].set_xlabel("Time (s)", color="#8b949e")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
# plt.show()  # uncomment to display interactively
print(f"\n✅ Figure saved → {OUT}")
