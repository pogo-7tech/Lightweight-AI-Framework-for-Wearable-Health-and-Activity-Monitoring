"""
Week 1 | Deliverable 1B
========================
Motion / IMU Preprocessing Pipeline

Steps demonstrated:
  1. Raw tri-axis accelerometer + gyroscope (100 Hz)
  2. Gravity component separation (low-pass 0.3 Hz)
  3. Body acceleration extraction
  4. Signal comparison across 3 activity levels
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ── Config ────────────────────────────────────────────────────────────────────
FS  = 100       # Sampling rate (Hz) — PAMAP2 IMU
DUR = 10        # Seconds to display
OUT = "notebooks/week1_figures/figB_motion_pipeline.png"

# ── Helpers ───────────────────────────────────────────────────────────────────
def make_imu(activity, fs=100, dur=30, seed=7):
    np.random.seed(seed)
    t    = np.linspace(0, dur, int(fs * dur))
    cfg  = {"sitting": (0.0, 0.01), "walking": (1.8, 0.5), "running": (2.5, 1.2)}
    freq, amp = cfg.get(activity, (1.8, 0.5))
    acc  = np.column_stack([
        amp * np.sin(2 * np.pi * freq * t) + 0.02 * np.random.randn(len(t)),
        amp * np.cos(2 * np.pi * freq * t) + 0.02 * np.random.randn(len(t)),
        9.81 + 0.1 * amp * np.sin(4 * np.pi * freq * t) + 0.01 * np.random.randn(len(t)),
    ])
    gyro = np.column_stack([
        0.3 * amp * np.sin(2 * np.pi * freq * t + 0.5) + 0.01 * np.random.randn(len(t)),
        0.3 * amp * np.cos(2 * np.pi * freq * t + 0.5) + 0.01 * np.random.randn(len(t)),
        0.05 * np.random.randn(len(t)),
    ])
    return t, acc, gyro

def separate_gravity(acc, fs, cutoff=0.3, order=4):
    """Low-pass filter to extract gravity; subtract for body acceleration."""
    nyq = fs / 2
    b, a = butter(order, cutoff / nyq, btype="low")
    gravity  = np.column_stack([filtfilt(b, a, acc[:, i]) for i in range(3)])
    body_acc = acc - gravity
    return body_acc, gravity

def magnitude(triaxis):
    return np.sqrt(np.sum(triaxis ** 2, axis=1))

def style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e")
    [s.set_edgecolor("#21262d") for s in ax.spines.values()]
    ax.grid(color="#21262d", linewidth=0.5)

# ── Pipeline ──────────────────────────────────────────────────────────────────
activities  = ["sitting", "walking", "running"]
labels      = ["Sitting (REST)", "Walking (LIGHT)", "Running (INTENSE)"]
colors      = ["#58a6ff", "#3fb950", "#f85149"]
N           = int(DUR * FS)   # samples to plot

BG  = "#0d1117"
fig, axes = plt.subplots(3, 3, figsize=(15, 8), facecolor=BG)
fig.suptitle("Deliverable 1B — Motion / IMU Preprocessing Pipeline",
             color="white", fontsize=14, fontweight="bold")

for col, (act, lbl, clr) in enumerate(zip(activities, labels, colors)):
    t, acc, gyro = make_imu(act)
    body_acc, _  = separate_gravity(acc, FS)
    mag          = magnitude(body_acc)

    for ax in axes[:, col]:
        style(ax)

    # Row 0: Raw accelerometer
    for i, ax_lbl in enumerate(["X", "Y", "Z"]):
        axes[0, col].plot(t[:N], acc[:N, i], linewidth=0.8, alpha=0.8, label=ax_lbl)
    axes[0, col].set_title(f"{lbl}\nRaw Accelerometer", color=clr, fontsize=9)
    axes[0, col].legend(fontsize=7, framealpha=0.2)
    axes[0, col].set_ylabel("m/s²", color="#8b949e", fontsize=8)

    # Row 1: Body acceleration (gravity removed)
    for i, ax_lbl in enumerate(["X", "Y", "Z"]):
        axes[1, col].plot(t[:N], body_acc[:N, i], linewidth=0.8, alpha=0.8, label=ax_lbl)
    axes[1, col].set_title("Body Acc (gravity removed)", color=clr, fontsize=9)
    axes[1, col].legend(fontsize=7, framealpha=0.2)
    axes[1, col].set_ylabel("m/s²", color="#8b949e", fontsize=8)

    # Row 2: Magnitude signal
    axes[2, col].plot(t[:N], mag[:N], color=clr, linewidth=1.2)
    axes[2, col].set_title("Magnitude |body_acc|", color=clr, fontsize=9)
    axes[2, col].set_ylabel("m/s²", color="#8b949e", fontsize=8)
    axes[2, col].set_xlabel("Time (s)", color="#8b949e", fontsize=8)

    # Print stats
    print(f"\n{lbl}")
    print(f"  Mean magnitude : {mag.mean():.3f} m/s²")
    print(f"  Std magnitude  : {mag.std():.3f} m/s²")

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
# plt.show()  # uncomment to display interactively
print(f"\n✅ Figure saved → {OUT}")
