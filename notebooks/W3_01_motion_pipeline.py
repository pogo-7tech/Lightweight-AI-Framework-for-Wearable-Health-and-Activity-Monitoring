"""
Week 3 | Step 1 of 2  — REAL PAMAP2 DATA
==========================================
Motion / IMU Pipeline on Real PAMAP2 Dataset

Dataset: PAMAP2 Physical Activity Monitoring
  - 9 subjects, 100 Hz IMU (wrist + chest + ankle)
  - 54 columns: timestamp, activityID, HR, 3×IMU data
  - Activities used: lying, sitting, standing, walking, running, cycling

Pipeline:
  1. Load real .dat files (subject101–109)
  2. Gravity separation (low-pass filter)
  3. 18-feature extraction per window
  4. Visualize activities + feature discrimination
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings("ignore")

OUT   = Path("notebooks/week3_figures")
OUT.mkdir(parents=True, exist_ok=True)
BG    = "#0d1117"
FS    = 100      # Hz — PAMAP2 IMU sample rate
WIN   = 200      # 2-second window at 100 Hz
STEP  = 100      # 50% overlap

# Activity IDs we use (subset of 18 total)
TARGET_ACTS = {1: "Lying", 2: "Sitting", 3: "Standing",
               4: "Walking", 5: "Running", 6: "Cycling"}
ACT_COLORS  = {1: "#58a6ff", 2: "#a371f7", 3: "#3fb950",
               4: "#f59e0b", 5: "#f85149", 6: "#ff7b72"}

PAMAP2_DIR = Path("data/PAMAP2/PAMAP2_Dataset/Protocol")

# Column indices for wrist IMU acc (columns 4-6, 0-indexed = 3,4,5 acc16g)
# Per dataset doc: col 0=timestamp, 1=actID, 2=HR, 3=temp_hand,
# 4-6=acc16g_hand, 7-9=acc6g_hand, 10-12=gyro_hand, 13-15=mag_hand
ACC_COLS = [4, 5, 6]   # wrist 16g accelerometer x,y,z (best for activity)

def load_subject(path):
    """Load a single subject .dat file into DataFrame."""
    df = pd.read_csv(path, sep=" ", header=None, na_values="NaN")
    return df

def lowpass(sig, cut=0.3, fs=100, order=4):
    nyq = fs / 2
    b, a = butter(order, cut / nyq, btype="low")
    return filtfilt(b, a, sig)

def extract_features(window):
    """18 statistical features from a 3-axis window (N×3)."""
    feats = []
    for ax in range(window.shape[1]):
        col = window[:, ax]
        feats += [
            np.mean(col),              # mean
            np.std(col),               # std
            np.max(col),               # max
            np.sqrt(np.mean(col**2)),  # RMS
            np.sum(col != 0) / len(col),  # zero-crossing rate approx
            np.sum(np.abs(np.diff(col)))  # signal magnitude area
        ]
    return np.array(feats)

# ─────────────────────────────────────────────────────────────────────────────
# Load real PAMAP2 data for all 9 subjects
# ─────────────────────────────────────────────────────────────────────────────
SUBJECTS = sorted(PAMAP2_DIR.glob("subject*.dat"))

print("="*65)
print("  REAL PAMAP2 Dataset — Loading Summary")
print("="*65)

all_feats, all_labels = [], []
subject_counts = {}

for subj_path in SUBJECTS:
    sid = subj_path.stem
    df  = load_subject(subj_path)

    # Filter to target activities only, drop NaN rows
    df  = df[df.iloc[:, 1].isin(TARGET_ACTS.keys())].dropna(subset=[3,4,5,6])
    if len(df) == 0:
        continue

    times  = df.iloc[:, 0].values
    acts   = df.iloc[:, 1].values.astype(int)
    acc    = df.iloc[:, ACC_COLS].values.astype(float)

    # Gravity separation
    grav   = np.column_stack([lowpass(acc[:, i]) for i in range(3)])
    body   = acc - grav

    # Sliding window feature extraction
    for start in range(0, len(body) - WIN, STEP):
        seg  = body[start:start+WIN]
        lbl  = acts[start + WIN//2]
        if lbl not in TARGET_ACTS:
            continue
        all_feats.append(extract_features(seg))
        all_labels.append(lbl)

    # Count per activity
    for act_id in TARGET_ACTS:
        n = int((acts == act_id).sum())
        subject_counts[f"{sid}_{act_id}"] = n

    act_summary = {TARGET_ACTS[k]: int((acts==k).sum()) for k in TARGET_ACTS if (acts==k).sum()>0}
    print(f"  {sid}: {len(df):>8,} rows | Acts: {act_summary}")

X = np.array(all_feats)
y = np.array(all_labels)

print("="*65)
print(f"  ✅ Total windows: {len(X)} | Feature dim: {X.shape[1]}")
for act_id, name in TARGET_ACTS.items():
    n = int((y == act_id).sum())
    print(f"     {name:<12}: {n:>5} windows")

# ─────────────────────────────────────────────────────────────────────────────
# Figure A — Raw IMU + gravity separation for 3 activities (subject101)
# ─────────────────────────────────────────────────────────────────────────────
df101 = load_subject(PAMAP2_DIR / "subject101.dat").dropna(subset=[3,4,5,6])
acts101 = df101.iloc[:, 1].values.astype(float)
acc101  = df101.iloc[:, ACC_COLS].values.astype(float)
grav101 = np.column_stack([lowpass(acc101[:, i]) for i in range(3)])
body101 = acc101 - grav101

fig, axes = plt.subplots(3, 2, figsize=(16, 10), facecolor=BG)
fig.suptitle("Week 3 | Real PAMAP2 — IMU Pipeline: Gravity Separation per Activity",
             color="white", fontsize=12, fontweight="bold")

for row_idx, (act_id, act_name) in enumerate([(1,"Lying"),(4,"Walking"),(5,"Running")]):
    mask  = (acts101 == act_id)
    idx   = np.where(mask)[0]
    if len(idx) < FS*5:
        continue
    start = idx[len(idx)//2]
    seg_raw  = acc101[start:start+FS*5]
    seg_body = body101[start:start+FS*5]
    t5 = np.arange(len(seg_raw)) / FS
    clr = ACT_COLORS[act_id]

    ax = axes[row_idx, 0]
    ax.set_facecolor(BG); ax.tick_params(colors="#8b949e", labelsize=8)
    [s.set_edgecolor("#21262d") for s in ax.spines.values()]
    ax.grid(color="#21262d", linewidth=0.5)
    for i, lbl in enumerate(["X","Y","Z"]):
        ax.plot(t5, seg_raw[:, i], linewidth=0.8, alpha=0.8, label=f"Acc {lbl}")
    ax.set_title(f"{act_name} — Raw IMU (wrist)", color="#c9d1d9", fontsize=9)
    ax.set_ylabel("Acc (m/s²)", color="#8b949e")
    ax.legend(framealpha=0.2, fontsize=8, loc="upper right")

    ax = axes[row_idx, 1]
    ax.set_facecolor(BG); ax.tick_params(colors="#8b949e", labelsize=8)
    [s.set_edgecolor("#21262d") for s in ax.spines.values()]
    ax.grid(color="#21262d", linewidth=0.5)
    for i, lbl in enumerate(["X","Y","Z"]):
        ax.plot(t5, seg_body[:, i], color=clr, linewidth=0.9, alpha=0.8, label=f"Body {lbl}")
    ax.set_title(f"{act_name} — Body Acceleration (gravity removed)", color="#c9d1d9", fontsize=9)
    ax.set_ylabel("Body Acc (m/s²)", color="#8b949e")
    ax.legend(framealpha=0.2, fontsize=8, loc="upper right")

for ax in axes[2]:
    ax.set_xlabel("Time (seconds)", color="#8b949e")

plt.tight_layout()
plt.savefig(OUT / "W3_01a_motion_pipeline.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("\n✅ Saved: W3_01a_motion_pipeline.png")

# ─────────────────────────────────────────────────────────────────────────────
# Figure B — Feature discrimination: RMS X vs Std Y per activity
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG)
fig.suptitle("Week 3 | Feature Space — Real PAMAP2 Activity Discrimination",
             color="white", fontsize=12, fontweight="bold")

ax = axes[0]
ax.set_facecolor(BG); ax.tick_params(colors="#8b949e", labelsize=8)
[s.set_edgecolor("#21262d") for s in ax.spines.values()]
ax.grid(color="#21262d", linewidth=0.5)
for act_id, name in TARGET_ACTS.items():
    mask = (y == act_id)
    if mask.sum() == 0: continue
    ax.scatter(X[mask, 3], X[mask, 1],
               color=ACT_COLORS[act_id], s=8, alpha=0.5, label=name)
ax.set_xlabel("RMS X-axis", color="#8b949e")
ax.set_ylabel("Std X-axis", color="#8b949e")
ax.set_title("Feature Space: RMS vs Std (X-axis)", color="#c9d1d9", fontsize=10)
ax.legend(framealpha=0.2, fontsize=8, markerscale=3)

ax = axes[1]
ax.set_facecolor(BG); ax.tick_params(colors="#8b949e", labelsize=8)
[s.set_edgecolor("#21262d") for s in ax.spines.values()]
ax.grid(color="#21262d", linewidth=0.5)
feat_names = ["Mean","Std","Max","RMS","ZCR","SMA"]
means_by_act = {}
for act_id, name in TARGET_ACTS.items():
    mask = (y == act_id)
    if mask.sum() == 0: continue
    means_by_act[name] = X[mask, :6].mean(axis=0)

x_pos = np.arange(6)
w = 0.13
for i, (name, vals) in enumerate(means_by_act.items()):
    act_id = [k for k,v in TARGET_ACTS.items() if v==name][0]
    normed = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    ax.bar(x_pos + (i-2.5)*w, normed, w,
           color=ACT_COLORS[act_id], alpha=0.8, label=name)
ax.set_xticks(x_pos); ax.set_xticklabels(feat_names, color="#8b949e")
ax.set_ylabel("Normalised Feature Value", color="#8b949e")
ax.set_title("Feature Profile per Activity (X-axis features)", color="#c9d1d9", fontsize=10)
ax.legend(framealpha=0.2, fontsize=8)

plt.tight_layout()
plt.savefig(OUT / "W3_01b_feature_discrimination.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("✅ Saved: W3_01b_feature_discrimination.png")
print(f"\n✅ Real PAMAP2 pipeline complete — {len(X)} windows extracted from 9 subjects!")
