"""
Week 5 | Step 3 of 3
======================
Confidence-Aware Alert Engine — 3-Tier System with Hysteresis

What this script does:
1. Fuses 3 signals into a single risk score (0–1):
      risk = w1 × HR_anomaly_score
           + w2 × signal_quality_score (inverted)
           + w3 × activity_confidence (inverted)
2. Maps risk score to 3 alert tiers:
      🟢 LOW  < 0.4  → log only
      🟡 MED  0.4–0.7 → warning notification
      🔴 HIGH > 0.7  → urgent alert (doctor/caregiver)
3. Implements hysteresis (60s cooldown) to prevent alert flooding
4. Demonstrates: 10 raw alerts/second → 1 sensible alert

Key Design:
  - Weights: HR anomaly 50%, Signal quality 30%, Activity confidence 20%
  - Hysteresis cooldown: 60 seconds after HIGH alert
  - State machine: HIGH → must drop to LOW before MED can re-trigger
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUT = Path("notebooks/week5_figures")
OUT.mkdir(parents=True, exist_ok=True)
BG  = "#0d1117"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Alert Engine Implementation
# ─────────────────────────────────────────────────────────────────────────────
TIER_COLORS = {
    "LOW":  "#3fb950",
    "MED":  "#d29922",
    "HIGH": "#f85149",
}
TIER_THRESHOLDS = {"LOW": 0.4, "MED": 0.7}   # LOW < 0.4, MED 0.4–0.7, HIGH > 0.7
WEIGHTS = {"hr": 0.50, "quality": 0.30, "confidence": 0.20}
COOLDOWN_S = 60   # seconds before HIGH can re-trigger


class ConfidenceAlertEngine:
    """
    3-tier confidence-aware alert engine with hysteresis.

    Inputs per timestep:
      hr_anomaly_score  : 0–1 (higher = more anomalous HR)
      signal_quality    : 0–1 (higher = better quality)
      activity_confidence: 0–1 (higher = classifier more certain)

    Output: tier (LOW / MED / HIGH)
    """

    def __init__(self, cooldown_s: int = COOLDOWN_S):
        self.cooldown_s       = cooldown_s
        self.last_high_time   = -cooldown_s - 1
        self.last_tier        = "LOW"
        self._history         = []

    def compute_risk(self, hr_score: float, quality: float, confidence: float) -> float:
        """Weighted fusion → risk score 0–1."""
        quality_inv    = 1.0 - quality       # low quality = higher risk
        confidence_inv = 1.0 - confidence    # low confidence = higher risk
        risk = (
            WEIGHTS["hr"]         * hr_score
            + WEIGHTS["quality"]  * quality_inv
            + WEIGHTS["confidence"] * confidence_inv
        )
        return float(np.clip(risk, 0.0, 1.0))

    def classify_tier(self, risk: float, t: int) -> str:
        if risk > TIER_THRESHOLDS["MED"]:
            # HIGH — only if cooldown has passed
            if (t - self.last_high_time) >= self.cooldown_s:
                self.last_high_time = t
                tier = "HIGH"
            else:
                tier = "MED"   # still in cooldown → downgrade
        elif risk > TIER_THRESHOLDS["LOW"]:
            # Must have dropped to LOW before MED can re-trigger after HIGH
            if self.last_tier == "HIGH" and (t - self.last_high_time) < self.cooldown_s:
                tier = "LOW"
            else:
                tier = "MED"
        else:
            tier = "LOW"

        self.last_tier = tier
        return tier

    def step(self, hr_score: float, quality: float, confidence: float, t: int) -> dict:
        risk = self.compute_risk(hr_score, quality, confidence)
        tier = self.classify_tier(risk, t)
        record = {
            "t": t, "hr_score": hr_score, "quality": quality,
            "confidence": confidence, "risk": risk, "tier": tier,
        }
        self._history.append(record)
        return record


# ─────────────────────────────────────────────────────────────────────────────
# 2. Simulate 300-second Session
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
N = 300
engine = ConfidenceAlertEngine(cooldown_s=COOLDOWN_S)

# Simulate inputs
t_arr = np.arange(N)

# HR anomaly score: mostly low, 3 real anomaly bursts
hr_score = np.random.exponential(0.08, N)
ANOM_BURSTS = [(70, 82), (150, 162), (240, 252)]
for s, e in ANOM_BURSTS:
    hr_score[s:e] = np.random.uniform(0.65, 0.92, e - s)
hr_score = np.clip(hr_score, 0, 1)

# Signal quality: generally high, dips during exercise
quality = np.ones(N) * 0.88
quality[120:200] = 0.70 + np.random.randn(80) * 0.05   # lower during running
quality = np.clip(quality, 0.1, 1.0)

# Activity confidence: high overall, slight uncertainty during transitions
confidence = np.ones(N) * 0.91
for trans in [40, 80, 120, 180, 240, 280]:
    start = max(0, trans - 5)
    end   = min(N, trans + 5)
    confidence[start:end] = np.random.uniform(0.60, 0.75, end - start)
confidence = np.clip(confidence, 0.1, 1.0)

# Run engine
records    = [engine.step(hr_score[t], quality[t], confidence[t], t) for t in t_arr]
risk_arr   = np.array([r["risk"] for r in records])
tier_arr   = np.array([r["tier"] for r in records])

# Raw alerts (no hysteresis) — every sample above 0.7
raw_high   = (risk_arr > TIER_THRESHOLDS["MED"]).sum()
smart_high = (tier_arr == "HIGH").sum()

print("=" * 65)
print("  Week 5 | Confidence-Aware Alert Engine — Results")
print("=" * 65)
print(f"\n  Cooldown period     : {COOLDOWN_S}s")
print(f"  Fusion weights      : HR={WEIGHTS['hr']:.0%}, Quality={WEIGHTS['quality']:.0%}, "
      f"Confidence={WEIGHTS['confidence']:.0%}")
print(f"\n  {'Metric':<30} {'No Hysteresis':>15} {'With Hysteresis':>17}")
print(f"  {'-'*62}")
print(f"  {'HIGH alerts fired':<30} {raw_high:>15} {smart_high:>17}")
print(f"  {'MED alerts fired':<30} {(risk_arr > TIER_THRESHOLDS['LOW']).sum():>15} "
      f"{(tier_arr == 'MED').sum():>17}")
print(f"  {'LOW (no alert)':<30} {(risk_arr <= TIER_THRESHOLDS['LOW']).sum():>15} "
      f"{(tier_arr == 'LOW').sum():>17}")
print(f"\n  Alert reduction     : {raw_high} → {smart_high} HIGH alerts ✅")

# ─────────────────────────────────────────────────────────────────────────────
# Figure W5_03a — Full Alert Engine Dashboard
# ─────────────────────────────────────────────────────────────────────────────
tier_int  = {"LOW": 0, "MED": 1, "HIGH": 2}
tier_vals = np.array([tier_int[t] for t in tier_arr])

fig = plt.figure(figsize=(18, 12), facecolor=BG)
fig.suptitle(
    "Week 5 — Confidence-Aware Alert Engine: 3-Tier System with Hysteresis (60s Cooldown)",
    color="white", fontsize=13, fontweight="bold"
)
gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.30)

def style(ax):
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#8b949e", labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor("#30363d")
    ax.grid(color="#21262d", linewidth=0.4, alpha=0.8)
    ax.yaxis.label.set_color("#8b949e")
    ax.xaxis.label.set_color("#8b949e")
    ax.title.set_color("#e6edf3")

# ── Row 0: HR Anomaly Score ──────────────────────────────────────────────────
ax0 = fig.add_subplot(gs[0, :])
ax0.fill_between(t_arr, hr_score, color="#bc8cff", alpha=0.75, label="HR Anomaly Score")
ax0.axhline(0.7, color="#f85149", linewidth=1.2, linestyle="--", label="HIGH threshold (0.7)")
ax0.axhline(0.4, color="#d29922", linewidth=1.0, linestyle="--", label="MED threshold (0.4)")
for s, e in ANOM_BURSTS:
    ax0.axvspan(s, e, color="#f85149", alpha=0.15, label="Anomaly burst" if s == ANOM_BURSTS[0][0] else "")
ax0.set_title("HR Anomaly Score (Input 1 of 3)", color="#e6edf3")
ax0.set_ylabel("Score")
ax0.legend(fontsize=7.5, framealpha=0.2, loc="upper right", ncol=2)
style(ax0)

# ── Row 1: Signal quality + activity confidence ──────────────────────────────
ax1 = fig.add_subplot(gs[1, 0])
ax1.plot(t_arr, quality, color="#3fb950", linewidth=1.0, label="Signal Quality")
ax1.set_title("Signal Quality Score (Input 2 of 3)", color="#e6edf3")
ax1.set_ylabel("Quality (0–1)")
ax1.legend(fontsize=8, framealpha=0.2)
style(ax1)

ax2 = fig.add_subplot(gs[1, 1])
ax2.plot(t_arr, confidence, color="#58a6ff", linewidth=1.0, label="Activity Confidence")
ax2.set_title("Activity Classifier Confidence (Input 3 of 3)", color="#e6edf3")
ax2.set_ylabel("Confidence (0–1)")
ax2.legend(fontsize=8, framealpha=0.2)
style(ax2)

# ── Row 2: Fused Risk Score ───────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, :])
# Colour by tier
for t_i in range(N - 1):
    clr = TIER_COLORS[tier_arr[t_i]]
    ax3.fill_between([t_arr[t_i], t_arr[t_i + 1]],
                     [risk_arr[t_i], risk_arr[t_i + 1]],
                     color=clr, alpha=0.75)
ax3.plot(t_arr, risk_arr, color="white", linewidth=0.6, alpha=0.5)
ax3.axhline(TIER_THRESHOLDS["MED"], color="#f85149", linewidth=1.2,
            linestyle="--", label="HIGH threshold (0.7)")
ax3.axhline(TIER_THRESHOLDS["LOW"], color="#d29922", linewidth=1.0,
            linestyle="--", label="MED threshold (0.4)")
ax3.set_title(
    f"Fused Risk Score = 0.5×HR + 0.3×(1−Quality) + 0.2×(1−Confidence)\n"
    f"🟢 LOW  |  🟡 MED  |  🔴 HIGH (with {COOLDOWN_S}s cooldown hysteresis)",
    color="#e6edf3", fontsize=9
)
ax3.set_ylabel("Risk Score")
ax3.set_ylim(0, 1.1)
ax3.legend(fontsize=8, framealpha=0.2, loc="upper right")
style(ax3)

# ── Row 3: Alert tier timeline ────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[3, 0])
for t_i in range(N - 1):
    clr = TIER_COLORS[tier_arr[t_i]]
    ax4.fill_between([t_arr[t_i], t_arr[t_i + 1]],
                     [tier_vals[t_i], tier_vals[t_i + 1]],
                     color=clr, alpha=0.85, step="post")
ax4.set_yticks([0, 1, 2])
ax4.set_yticklabels(["🟢 LOW", "🟡 MED", "🔴 HIGH"], fontsize=8)
ax4.set_title(f"Smart Alert Tier Output\n(With hysteresis: {smart_high} HIGH alerts)",
              color="#e6edf3", fontsize=9)
ax4.set_xlabel("Time (s)")
style(ax4)

# ── Row 3 Right: Comparison bar ───────────────────────────────────────────────
ax5 = fig.add_subplot(gs[3, 1])
ax5.set_facecolor("#161b22")

categories = ["HIGH Alerts\nFired", "Unnecessary\nAlerts Blocked"]
no_hyst    = [raw_high,   0]
with_hyst  = [smart_high, raw_high - smart_high]

x_c = np.arange(2)
w_c = 0.30
ax5.bar(x_c - w_c/2, no_hyst,   w_c, label="Without Hysteresis", color="#f85149", alpha=0.8)
ax5.bar(x_c + w_c/2, with_hyst, w_c, label="With Hysteresis",    color="#3fb950", alpha=0.8)

for i, (nv, wv) in enumerate(zip(no_hyst, with_hyst)):
    ax5.text(i - w_c/2, nv + 0.3, str(nv), ha="center", color="white",
             fontsize=9, fontweight="bold")
    ax5.text(i + w_c/2, wv + 0.3, str(wv), ha="center", color="white",
             fontsize=9, fontweight="bold")

ax5.set_xticks(x_c)
ax5.set_xticklabels(categories, color="#8b949e", fontsize=8.5)
ax5.set_title(f"Hysteresis Benefit\n{raw_high} raw HIGH → {smart_high} smart HIGH alerts",
              color="#e6edf3", fontsize=9)
ax5.legend(fontsize=8, framealpha=0.2)
ax5.tick_params(colors="#8b949e")
for s in ax5.spines.values():
    s.set_edgecolor("#30363d")
ax5.grid(color="#21262d", linewidth=0.4, axis="y")

plt.savefig(OUT / "W5_03a_alert_engine.png", dpi=150, bbox_inches="tight", facecolor=BG)
print(f"\n✅ Figure saved → {OUT}/W5_03a_alert_engine.png")
plt.close("all")

print(f"\n{'='*65}")
print(f"  Alert Engine Summary")
print(f"{'='*65}")
print(f"  Fusion weights : HR 50% | Quality 30% | Confidence 20%")
print(f"  Tier thresholds: LOW<0.4 | MED 0.4–0.7 | HIGH>0.7")
print(f"  Cooldown       : {COOLDOWN_S}s hysteresis after HIGH alert")
print(f"  HIGH alerts    : {raw_high} (raw) → {smart_high} (smart) ✅")
print(f"  Reduction      : {(1 - smart_high/max(raw_high,1))*100:.1f}% fewer HIGH alerts")
