"""
Edge Model Export — TFLite INT8 + ONNX
========================================
Converts trained Keras models to:
  1. TensorFlow Lite (INT8 post-training quantization) — for microcontrollers / RPi
  2. ONNX — for CPU/GPU edge runtimes (ONNX Runtime)

Also benchmarks model size and accuracy trade-off.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
import pathlib
from loguru import logger
from typing import Optional, Callable
import json
import time


# ── TFLite Export ────────────────────────────────────────────────────────────

def export_tflite_float32(model: keras.Model, output_path: str) -> int:
    """Export model as float32 TFLite model. Returns file size in bytes."""
    converter      = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model   = converter.convert()
    pathlib.Path(output_path).write_bytes(tflite_model)
    size = len(tflite_model)
    logger.info(f"TFLite float32 saved → {output_path} ({size / 1024:.1f} KB)")
    return size


def export_tflite_int8(
    model: keras.Model,
    output_path: str,
    representative_dataset: Callable,
) -> int:
    """
    Export model with INT8 post-training quantization.

    Parameters
    ----------
    model                 : trained Keras model
    output_path           : .tflite output file path
    representative_dataset: callable yielding batches of representative input data

    Returns
    -------
    int : file size in bytes
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()
    pathlib.Path(output_path).write_bytes(tflite_model)
    size = len(tflite_model)
    logger.info(f"TFLite INT8 saved → {output_path} ({size / 1024:.1f} KB)")
    return size


# ── ONNX Export ───────────────────────────────────────────────────────────────

def export_onnx(model: keras.Model, output_path: str, input_shape: tuple) -> int:
    """
    Export Keras model to ONNX format via tf2onnx.

    Requires: pip install tf2onnx
    """
    try:
        import tf2onnx
        import onnx

        spec = (tf.TensorSpec(input_shape, tf.float32, name="input"),)
        _, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)

        size = pathlib.Path(output_path).stat().st_size
        logger.info(f"ONNX model saved → {output_path} ({size / 1024:.1f} KB)")
        return size
    except ImportError:
        logger.error("tf2onnx not installed. Run: pip install tf2onnx")
        return 0


# ── TFLite Inference Benchmark ────────────────────────────────────────────────

def benchmark_tflite(tflite_path: str, test_inputs: np.ndarray, n_runs: int = 100) -> dict:
    """
    Measure TFLite inference latency on test inputs.

    Returns
    -------
    dict with p50, p95, p99 latency (ms), throughput (samples/sec)
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    latencies = []
    for i in range(min(n_runs, len(test_inputs))):
        inp = test_inputs[i: i + 1]

        # Handle INT8 quantization scaling
        if input_details[0]["dtype"] == np.int8:
            scale, zero_point = input_details[0]["quantization"]
            inp = (inp / scale + zero_point).astype(np.int8)

        interpreter.set_tensor(input_details[0]["index"], inp)

        t0 = time.perf_counter()
        interpreter.invoke()
        latencies.append((time.perf_counter() - t0) * 1000.0)   # ms

    lat = np.array(latencies)
    result = {
        "n_runs":          len(lat),
        "mean_ms":         float(np.mean(lat)),
        "p50_ms":          float(np.percentile(lat, 50)),
        "p95_ms":          float(np.percentile(lat, 95)),
        "p99_ms":          float(np.percentile(lat, 99)),
        "throughput_sps":  float(1000.0 / np.mean(lat)),
        "model_size_kb":   pathlib.Path(tflite_path).stat().st_size / 1024,
    }
    logger.info(f"Benchmark [{tflite_path}]: p50={result['p50_ms']:.2f}ms | p99={result['p99_ms']:.2f}ms")
    return result


# ── Full export pipeline ──────────────────────────────────────────────────────

def export_all(
    model: keras.Model,
    output_dir: str,
    representative_data: np.ndarray,
    model_name: str = "model",
) -> dict:
    """
    Export model to float32 TFLite + INT8 TFLite and benchmark both.

    Parameters
    ----------
    model              : trained Keras model
    output_dir         : directory for exported models
    representative_data: sample data for INT8 calibration (N, *input_shape)
    model_name         : base filename

    Returns
    -------
    dict with paths and benchmark results
    """
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    f32_path = str(out / f"{model_name}_float32.tflite")
    i8_path  = str(out / f"{model_name}_int8.tflite")

    # Float32 export
    size_f32 = export_tflite_float32(model, f32_path)

    # INT8 export
    def rep_dataset():
        for i in range(min(100, len(representative_data))):
            yield [representative_data[i: i + 1].astype(np.float32)]

    size_i8 = export_tflite_int8(model, i8_path, rep_dataset)

    # Benchmarks
    bench_f32 = benchmark_tflite(f32_path, representative_data)
    bench_i8  = benchmark_tflite(i8_path,  representative_data)

    summary = {
        "float32": {"path": f32_path, "size_kb": size_f32 / 1024, **bench_f32},
        "int8":    {"path": i8_path,  "size_kb": size_i8  / 1024, **bench_i8},
        "compression_ratio": size_f32 / (size_i8 + 1e-6),
        "speedup":           bench_f32["mean_ms"] / (bench_i8["mean_ms"] + 1e-6),
    }

    report_path = out / f"{model_name}_export_report.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Export report saved → {report_path}")

    return summary
