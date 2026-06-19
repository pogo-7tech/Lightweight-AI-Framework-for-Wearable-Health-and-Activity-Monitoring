"""
Anomaly Detector
=================
Two complementary approaches:

1. LSTM Autoencoder (unsupervised, sequence-level)
   - Learns normal physiological signal patterns
   - Flags windows where reconstruction error > learned threshold

2. Isolation Forest (unsupervised, feature-level)
   - Fast, lightweight, ideal for edge deployment
   - Operates on HRV / IMU feature vectors

3. Rolling Z-score baseline
   - Ultra-lightweight — no model required
   - Suitable for real-time streaming on edge hardware
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import yaml
from loguru import logger
from pathlib import Path
from typing import Optional
import joblib


def _load_config(config_path: str = "configs/pipeline_config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── LSTM Autoencoder ──────────────────────────────────────────────────────────

def build_lstm_autoencoder(window_size: int, n_features: int, latent_dim: int = 64) -> keras.Model:
    """Build LSTM autoencoder for temporal anomaly detection."""
    inp = keras.Input(shape=(window_size, n_features), name="seq_input")

    # Encoder
    encoded = layers.LSTM(latent_dim, activation="tanh", return_sequences=False)(inp)

    # Repeat for decoder
    repeated = layers.RepeatVector(window_size)(encoded)

    # Decoder
    decoded = layers.LSTM(latent_dim, activation="tanh", return_sequences=True)(repeated)
    out     = layers.TimeDistributed(layers.Dense(n_features), name="reconstruction")(decoded)

    model = keras.Model(inp, out, name="LSTM_Autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


class LSTMAnomalyDetector:
    """LSTM autoencoder for sequence-level anomaly detection."""

    def __init__(
        self,
        window_size: int = 50,
        n_features:  int = 1,
        latent_dim:  int = 64,
        model_path:  Optional[str] = None,
    ):
        self.window_size = window_size
        self.n_features  = n_features
        self.threshold_  = None

        if model_path and Path(model_path).exists():
            self.model = keras.models.load_model(model_path)
            logger.info(f"LSTM anomaly model loaded from {model_path}")
        else:
            self.model = build_lstm_autoencoder(window_size, n_features, latent_dim)
            logger.info(f"LSTM autoencoder built | window={window_size} features={n_features}")

    def _to_sequences(self, data: np.ndarray) -> np.ndarray:
        """Shape data into (N, window_size, n_features)."""
        if data.ndim == 1:
            data = data[:, np.newaxis]
        seqs = []
        for i in range(0, len(data) - self.window_size + 1, self.window_size):
            seqs.append(data[i: i + self.window_size])
        return np.array(seqs, dtype=np.float32)

    def fit(
        self,
        normal_data: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        val_split: float = 0.1,
        threshold_percentile: float = 95.0,
        save_path: Optional[str] = None,
    ) -> None:
        seqs = self._to_sequences(normal_data)
        logger.info(f"LSTM training | sequences={len(seqs)} epochs={epochs}")

        callbacks = [
            keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
        ]
        self.model.fit(
            seqs, seqs,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=val_split,
            callbacks=callbacks,
            verbose=1,
        )

        # Set threshold from reconstruction error on training data
        recon_errors = self._reconstruction_errors(seqs)
        self.threshold_ = float(np.percentile(recon_errors, threshold_percentile))
        logger.info(f"Anomaly threshold (p{threshold_percentile:.0f}) = {self.threshold_:.6f}")

        if save_path:
            self.model.save(save_path)
            logger.info(f"LSTM model saved → {save_path}")

    def _reconstruction_errors(self, seqs: np.ndarray) -> np.ndarray:
        preds = self.model.predict(seqs, verbose=0)
        return np.mean(np.mean((seqs - preds) ** 2, axis=2), axis=1)

    def predict(self, data: np.ndarray) -> dict:
        """
        Returns anomaly scores and binary labels.

        Returns
        -------
        dict with 'scores', 'labels' (1=anomaly, 0=normal), 'threshold'
        """
        seqs   = self._to_sequences(data)
        scores = self._reconstruction_errors(seqs)
        labels = (scores > self.threshold_).astype(int) if self.threshold_ else np.zeros(len(scores))
        return {"scores": scores, "labels": labels, "threshold": self.threshold_}


# ── Isolation Forest ──────────────────────────────────────────────────────────

class IsolationForestDetector:
    """Feature-level anomaly detection using Isolation Forest."""

    def __init__(self, contamination: float = 0.05, model_path: Optional[str] = None):
        self.contamination = contamination
        self.scaler_       = StandardScaler()

        if model_path and Path(model_path).exists():
            bundle = joblib.load(model_path)
            self.model_  = bundle["model"]
            self.scaler_ = bundle["scaler"]
            logger.info(f"IsolationForest loaded from {model_path}")
        else:
            self.model_ = IsolationForest(
                contamination=contamination,
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
            )
            logger.info("IsolationForest initialized")

    def fit(self, features: pd.DataFrame, save_path: Optional[str] = None) -> None:
        X = self.scaler_.fit_transform(features.select_dtypes(include=[np.number]).fillna(0))
        self.model_.fit(X)
        logger.info(f"IsolationForest fitted | samples={len(X)}")
        if save_path:
            joblib.dump({"model": self.model_, "scaler": self.scaler_}, save_path)
            logger.info(f"IsolationForest saved → {save_path}")

    def predict(self, features: pd.DataFrame) -> dict:
        """Returns anomaly scores (negative = more anomalous) and labels (1=anomaly)."""
        X      = self.scaler_.transform(features.select_dtypes(include=[np.number]).fillna(0))
        scores = -self.model_.score_samples(X)   # flip: higher = more anomalous
        labels = (self.model_.predict(X) == -1).astype(int)
        return {"scores": scores, "labels": labels}


# ── Rolling Z-score (edge-friendly) ──────────────────────────────────────────

class RollingZScoreDetector:
    """
    Ultra-lightweight online anomaly detector using a rolling z-score.
    No model training required — ideal for real-time edge deployment.
    """

    def __init__(self, window_size: int = 100, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold   = threshold
        self._buffer: list[float] = []

    def update(self, value: float) -> dict:
        """
        Process one new sample and return anomaly decision.

        Returns
        -------
        dict with 'zscore', 'is_anomaly', 'mean', 'std'
        """
        self._buffer.append(value)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

        arr    = np.array(self._buffer)
        mean   = float(np.mean(arr))
        std    = float(np.std(arr))
        zscore = abs(value - mean) / (std + 1e-8)

        return {
            "zscore":     zscore,
            "is_anomaly": bool(zscore > self.threshold),
            "mean":       mean,
            "std":        std,
        }

    def process_batch(self, signal: np.ndarray) -> pd.DataFrame:
        """Process a full signal array and return a DataFrame of results."""
        results = [self.update(float(v)) for v in signal]
        return pd.DataFrame(results)
