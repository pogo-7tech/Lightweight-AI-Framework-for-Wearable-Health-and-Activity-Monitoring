"""
Week 1 | Deliverable 2B
========================
Missing Signal Reconstruction + Anomaly Detection

Part 1 — Missing Signal Reconstruction
  - Detect flat-line dropouts (rolling std < threshold)
  - Reconstruct gaps using cubic spline interpolation
  - Evaluate with RMSE vs ground truth

Part 2 — Anomaly Detection (3 methods)
  - Rolling Z-score (lightweight, edge-friendly)
  - Reconstruction error (LSTM-autoencoder concept)
  - Amplitude threshold (Isolation Forest concept)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d

# ── Config ────────────────────────────────────────────────────────────────────
FS  = 64
OUT_RECON   = "notebooks/week1_figures/figD_missing_reconstruction.png"
OUT_ANOMALY = "notebooks/week1_figures/figE_anomaly_detection.png"

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_ppg(fs=64, dur=30, bpm=72, noise=0.04, seed=5):
    np.random.seed(seed)
    t = np.linspace(0, dur, int(fs * dur))
    f = bpm / 60
    sig = (0.8 * np.sin(2 * np.pi * f * t)
           + 0.25 * np.sin(4 * np.pi * f * t)
           + noise * np.random.randn(len(t)))
    # detrend + bandpass
    b, a = butter(4, [0.5 / (fs/2), 8.0 / (fs/2)], btype="band")
    sig  = filtfilt(b, a, sig)
    sig -= np.polyval(np.polyfit(np.arange(len(sig)), sig, 3), np.arange(len(sig)))
    return t, sig

def style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    [s.set_edgecolor("#21262d") for s in ax.spines.values()]
    ax.grid(color="#21262d", linewidth=0.5)

# ═══════════════════════════════════════════════════════════════════════
# PART 1 — MISSING SIGNAL RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════
t, clean = make_ppg()

# Inject flat-line dropouts
corrupted = clean.copy()
GAPS = [(int(5*FS), int(7*FS)), (int(15*FS), int(16.5*FS)), (int(22*FS), int(24*FS))]
for s, e in GAPS:
    corrupted[s:e] = 0.0

# Detect dropouts (rolling std below threshold)
WIN = int(0.5 * FS)
bad = np.zeros(len(corrupted), dtype=bool)
for i in range(WIN, len(corrupted) - WIN):
    if np.std(corrupted[i - WIN:i + WIN]) < 0.02:
        bad[i - WIN:i + WIN] = True

# Cubic spline reconstruction
good_idx  = np.where(~bad)[0]
good_val  = corrupted[good_idx]
interp_fn = interp1d(good_idx, good_val, kind="cubic",
                     bounds_error=False, fill_value=(good_val[0], good_val[-1]))
recon     = corrupted.copy()
recon[bad] = interp_fn(np.where(bad)[0])

# RMSE per gap
rmse_list = [np.sqrt(np.mean((recon[s:e] - clean[s:e]) ** 2)) for s, e in GAPS]
print("\n Missing Signal Reconstruction:")
for i, (s, e) in enumerate(GAPS):
    dur_s = (e - s) / FS
    print(f"  Gap {i+1}: {dur_s:.1f}s  →  RMSE = {rmse_list[i]:.4f}")

# Plot Part 1
BG  = "#0d1117"
fig, axes = plt.subplots(3, 1, figsize=(14, 8), facecolor=BG)
fig.suptitle("Deliverable 2B (Part 1) — Missing Signal Reconstruction",
             color="white", fontsize=14, fontweight="bold")

style(axes[0])
axes[0].plot(t, clean, color="#3fb950", linewidth=1.0)
axes[0].set_title("① Original clean signal", color="#c9d1d9", fontsize=10)
axes[0].set_ylabel("Amplitude", color="#8b949e", fontsize=8)

style(axes[1])
axes[1].plot(t, corrupted, color="#58a6ff", linewidth=0.9)
for i, (s, e) in enumerate(GAPS):
    axes[1].axvspan(t[s], t[e], color="#f85149", alpha=0.3,
                    label="Flat-line dropout" if i == 0 else "")
axes[1].set_title("② Signal with flat-line dropouts (3 gaps: 2s, 1.5s, 2s)",
                  color="#c9d1d9", fontsize=10)
axes[1].set_ylabel("Amplitude", color="#8b949e", fontsize=8)
axes[1].legend(framealpha=0.2)

style(axes[2])
axes[2].plot(t, recon,  color="#3fb950", linewidth=1.0, label="Reconstructed")
axes[2].plot(t, clean,  color="white",  linewidth=0.7, alpha=0.3,
             linestyle="--", label="Ground truth")
for s, e in GAPS:
    axes[2].axvspan(t[s], t[e], color="#3fb950", alpha=0.1)
axes[2].set_title("③ After cubic spline reconstruction — gaps filled",
                  color="#c9d1d9", fontsize=10)
axes[2].set_ylabel("Amplitude", color="#8b949e", fontsize=8)
axes[2].set_xlabel("Time (s)", color="#8b949e")
axes[2].legend(framealpha=0.2)
axes[2].text(0.01, 0.05, f"Mean RMSE in gaps: {np.mean(rmse_list):.4f}",
             transform=axes[2].transAxes, color="#3fb950", fontsize=9)

plt.tight_layout()
plt.savefig(OUT_RECON, dpi=150, bbox_inches="tight", facecolor=BG)
# plt.show()  # uncomment to display interactively
print(f"✅ Figure saved → {OUT_RECON}")

# ═══════════════════════════════════════════════════════════════════════
# PART 2 — ANOMALY DETECTION
# ═══════════════════════════════════════════════════════════════════════
np.random.seed(3)
t2, sig = make_ppg(noise=0.04, seed=3)
noisy   = sig.copy()
ANOMS   = [(int(8*FS), int(10*FS)), (int(20*FS), int(22*FS))]
for s, e in ANOMS:
    noisy[s:e] += 1.5 * np.random.randn(e - s)

# Method 1: Rolling Z-score
WIN_Z = 100
zscore = np.zeros(len(noisy))
for i in range(WIN_Z, len(noisy)):
    w = noisy[i - WIN_Z:i]
    zscore[i] = abs((noisy[i] - w.mean()) / (w.std() + 1e-8))

# Method 2: Reconstruction error (sliding window vs clean)
WIN_R = 64
recon_err = np.zeros(len(noisy))
for i in range(WIN_R, len(noisy)):
    recon_err[i] = np.mean((noisy[i-WIN_R:i] - sig[i-WIN_R:i]) ** 2)
recon_err /= (recon_err.max() + 1e-8)

# Method 3: Amplitude envelope threshold
envelope = np.abs(noisy)
thr_env  = np.mean(envelope) + 2 * np.std(envelope)

print("\n Anomaly Detection Results:")
print(f"  Method 1 — Z-score    : {(zscore > 3.0).sum()} samples flagged")
print(f"  Method 2 — Recon err  : {(recon_err > 0.5).sum()} samples flagged")
print(f"  Method 3 — Envelope   : {(envelope > thr_env).sum()} samples flagged")

# Plot Part 2
fig2, axes2 = plt.subplots(4, 1, figsize=(14, 10), facecolor=BG)
fig2.suptitle("Deliverable 2B (Part 2) — Anomaly Detection (3 Methods)",
              color="white", fontsize=14, fontweight="bold")

style(axes2[0])
axes2[0].plot(t2, noisy, color="#58a6ff", linewidth=0.9)
for i, (s, e) in enumerate(ANOMS):
    axes2[0].axvspan(t2[s], t2[e], color="#f85149", alpha=0.25,
                     label="Injected anomaly" if i == 0 else "")
axes2[0].set_title("Input signal with 2 injected anomaly bursts",
                   color="#c9d1d9", fontsize=10)
axes2[0].set_ylabel("Amplitude", color="#8b949e", fontsize=8)
axes2[0].legend(framealpha=0.2)

style(axes2[1])
axes2[1].plot(t2, zscore, color="#f59e0b", linewidth=0.9, label="Z-score")
axes2[1].axhline(3.0, color="#f85149", linestyle="--", linewidth=1.2, label="Threshold z=3")
axes2[1].fill_between(t2, zscore, 3.0, where=zscore > 3.0, color="#f85149", alpha=0.4)
axes2[1].set_title("① Rolling Z-Score  (window = 100 samples)",
                   color="#c9d1d9", fontsize=10)
axes2[1].set_ylabel("Z-score", color="#8b949e", fontsize=8)
axes2[1].legend(framealpha=0.2, fontsize=8)

style(axes2[2])
axes2[2].plot(t2, recon_err, color="#a371f7", linewidth=0.9, label="Recon error")
axes2[2].axhline(0.5, color="#f85149", linestyle="--", linewidth=1.2, label="Threshold")
axes2[2].fill_between(t2, recon_err, 0.5, where=recon_err > 0.5,
                       color="#f85149", alpha=0.4)
axes2[2].set_title("② Reconstruction Error  (LSTM-Autoencoder concept)",
                   color="#c9d1d9", fontsize=10)
axes2[2].set_ylabel("Norm. Error", color="#8b949e", fontsize=8)
axes2[2].legend(framealpha=0.2, fontsize=8)

style(axes2[3])
axes2[3].plot(t2, envelope, color="#3fb950", linewidth=0.9, label="Amplitude envelope")
axes2[3].axhline(thr_env, color="#f85149", linestyle="--", linewidth=1.2,
                 label=f"Threshold = {thr_env:.2f}")
axes2[3].fill_between(t2, envelope, thr_env, where=envelope > thr_env,
                       color="#f85149", alpha=0.4, label="Flagged")
axes2[3].set_title("③ Amplitude Threshold  (Isolation Forest concept)",
                   color="#c9d1d9", fontsize=10)
axes2[3].set_ylabel("Amplitude", color="#8b949e", fontsize=8)
axes2[3].set_xlabel("Time (s)", color="#8b949e")
axes2[3].legend(framealpha=0.2, fontsize=8)

plt.tight_layout()
plt.savefig(OUT_ANOMALY, dpi=150, bbox_inches="tight", facecolor=BG)
# plt.show()  # uncomment to display interactively
print(f"✅ Figure saved → {OUT_ANOMALY}")
