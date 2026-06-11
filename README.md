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
│   ├── cloud/
│   │   ├── mqtt_publisher.py     # simulated edge MQTT publisher
│   │   ├── influxdb_writer.py    # MQTT→InfluxDB subscriber
│   │   └── docker-compose.yml    # Mosquitto + InfluxDB + Grafana stack
│   └── dashboard/
│       └── app.py                # Streamlit monitoring dashboard
├── notebooks/            # EDA and training notebooks
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

### 4. Run tests

```bash
pytest tests/ -v --tb=short
```

### 5. Run latency benchmark

```bash
python benchmarks/latency_benchmark.py
```

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
