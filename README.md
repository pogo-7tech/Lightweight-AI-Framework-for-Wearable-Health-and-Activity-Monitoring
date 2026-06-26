# Edge-Aware Health and Activity Monitoring using Wearable Signal Intelligence

> **Internship Project** | 2-Month Research & Development Program  
> **Paper**: "Lightweight AI Framework for Wearable Health and Activity Monitoring" (IEEE Format)

---

## Project Overview

A full-stack edge-AI pipeline for real-time physiological and motion monitoring:

```
Wearable Sensors (PPG + IMU)
        │
        ▼
┌───────────────────────┐
│  Preprocessing Layer  │  ← bandpass filter, denoising, HRV features
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Edge Inference Layer │  ← 1D-CNN (TFLite INT8), LSTM-AE, Isolation Forest
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Context Alert Engine │  ← activity-aware thresholds, 3-tier alerts
└───────────────────────┘
        │
   MQTT / InfluxDB
        │
        ▼
┌───────────────────────┐
│  Streamlit Dashboard  │  ← real-time waveforms, anomaly scores, activity
└───────────────────────┘
```

---

## Weekly Progress

| Week | Deliverables | Key Result |
|------|-------------|------------|
| **Week 1** | PPG preprocessing pipeline, IMU gravity separation, baseline wander removal | PPG SNR: 18.4 dB |
| **Week 2** | HRV feature extraction (RMSSD, SDNN, pNN50), 15-subject WESAD EDA, signal quality scoring | R² HRV Fit: 0.87 |
| **Week 3** | 1D-CNN denoiser training on PAMAP2 (8 subjects), motion artifact removal | MSE = 0.016 |
| **Week 4** | LSTM-AE anomaly detector (AUC=0.962), Isolation Forest (AUC=0.895), TFLite INT8 edge benchmark | 74.2% model compression |
| **Week 5** | MQTT + InfluxDB cloud pipeline, Streamlit real-time dashboard, end-to-end integration validation | 92.5% bandwidth reduction, E2E latency 10.7 ms |

---

## Project Structure

```
Internship/
├── data/
│   ├── raw/              # WESAD, PAMAP2 raw files
│   ├── processed/        # cleaned windows, feature tensors
│   └── synthetic/        # generated test signals
├── src/
│   ├── preprocessing/
│   │   ├── ppg_pipeline.py       # PPG/respiration pipeline
│   │   ├── motion_pipeline.py    # IMU tri-axis pipeline
│   │   └── missing_data.py       # dropout detection & reconstruction
│   ├── models/
│   │   ├── denoiser.py           # 1D-CNN autoencoder denoiser
│   │   ├── anomaly_detector.py   # LSTM-AE + Isolation Forest + Z-score
│   │   ├── activity_classifier.py # compact 1D-CNN, 6 classes
│   │   ├── context_monitor.py    # activity-aware physiological ranges
│   │   └── alert_engine.py       # confidence-aware 3-tier alerts
│   ├── edge/
│   │   └── export_tflite.py      # INT8 TFLite + ONNX export + benchmark
│   ├── cloud/                    # ← NEW in Week 5
│   │   ├── mqtt_publisher.py     # simulated edge MQTT publisher
│   │   ├── influxdb_writer.py    # MQTT→InfluxDB subscriber
│   │   └── docker-compose.yml    # Mosquitto + InfluxDB + Grafana stack
│   └── dashboard/                # ← NEW in Week 5
│       └── app.py                # Streamlit real-time monitoring dashboard
├── notebooks/            # EDA and training notebooks
│   ├── W2_01_data_loading.py
│   ├── W2_02_ppg_eda.py
│   ├── W2_03_hrv_analysis.py
│   ├── W2_04_signal_quality.py
│   ├── W3_01_motion_pipeline.py
│   ├── W3_02_denoiser_training.py
│   ├── W4_01_anomaly_detection.py
│   ├── W4_02_edge_benchmark.py
│   ├── W5_01_dashboard_demo.py       # ← NEW
│   ├── W5_02_integration_validation.py  # ← NEW
│   └── W5_03_final_summary.py           # ← NEW
├── tests/                # pytest unit tests
├── benchmarks/
│   └── latency_benchmark.py      # cloud vs edge latency comparison
├── configs/
│   ├── pipeline_config.yaml
│   └── cloud_config.yaml
└── paper/
    └── main.tex          # IEEE conference paper (IEEEtran)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the dashboard (demo mode — no data needed)

```bash
streamlit run src/dashboard/app.py
```

### 3. Start the cloud stack (requires Docker)

```bash
cd src/cloud
docker compose up -d
```

### 4. Run Week 5 notebooks

```bash
# Telemetry & dashboard demo
python notebooks/W5_01_dashboard_demo.py

# End-to-end integration validation
python notebooks/W5_02_integration_validation.py

# 5-week project summary
python notebooks/W5_03_final_summary.py
```

### 5. Run tests

```bash
pytest tests/ -v --tb=short
```

### 6. Run latency benchmark

```bash
python benchmarks/latency_benchmark.py
```

---

## Key Results

| Metric | Value |
|--------|-------|
| LSTM-AE Anomaly AUC-ROC | **0.962** |
| Isolation Forest AUC-ROC | 0.895 |
| Denoiser MSE | 0.016 |
| TFLite INT8 model size | 108 KB (↓74.2% from 420 KB) |
| Edge inference latency | 2.1 ms (↓83% from 12.4 ms FP32) |
| Bandwidth reduction | **92.5%** (raw → MQTT telemetry) |
| End-to-end pipeline latency | **10.7 ms** |
| Alert engine test pass rate | 8/8 ✅ |

---

## Datasets

| Dataset | Signal | Hz | Classes | Use |
|---|---|---|---|---|
| [WESAD](https://uni-siegen.de/wesad) | PPG, EDA, Temp | 64 | Stress/Baseline | Denoising, anomaly detection |
| [PAMAP2](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring) | IMU (wrist+ankle+chest) | 100 | 18 activities | Activity classification |

Download instructions in `data/README.md`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Signal processing | NumPy, SciPy, NeuroKit2, BioSPPy |
| Deep learning | TensorFlow/Keras |
| Edge export | TensorFlow Lite (INT8), ONNX |
| Cloud pipeline | MQTT (Mosquitto), InfluxDB 2.x, Grafana |
| Dashboard | Streamlit, Plotly |
| Paper | LaTeX (IEEEtran), Overleaf |

---

## Paper

**Title**: Lightweight AI Framework for Wearable Health and Activity Monitoring  
**Format**: IEEE Conference (IEEEtran)  
**Source**: `paper/main.tex` → upload to Overleaf

---

## License

For academic/internship use only.
