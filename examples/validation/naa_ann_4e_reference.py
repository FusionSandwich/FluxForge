#!/usr/bin/env python3
"""
Reference runner for NAA-ANN-1 (4e) using the original notebook workflow.

This script executes the same preprocessing + ANN training logic as the
NAA-ANN-1 notebook and writes a reference results CSV for parity checks.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.models import load_model
except ImportError as exc:
    raise SystemExit("TensorFlow is required to run the NAA-ANN reference workflow.") from exc


LABEL_COLS = ["Se", "UNC", "LOD"]
EXP_COLS = ["SAMPLE MASS", "FILL", "CMP"]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _load_csv_tables(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open("4e_in_data.csv") as f_data:
            real = pd.read_csv(f_data, sep=";")
        with zf.open("4e_in_synthetic.csv") as f_synth:
            synth = pd.read_csv(f_synth, sep=";")

    exp = pd.concat([real[LABEL_COLS + EXP_COLS], synth[LABEL_COLS + EXP_COLS]], axis=0)
    exp = exp.reset_index(drop=True)
    exp[LABEL_COLS] = exp[LABEL_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    exp[EXP_COLS] = exp[EXP_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return exp


def _load_counts(zip_path: Path, max_files: int | None = None) -> tuple[np.ndarray, list[str]]:
    with zipfile.ZipFile(zip_path) as zf:
        data_files = [
            name for name in zf.namelist()
            if name.endswith(".dat") and (name.startswith("4e_data_") or name.startswith("4e_synth_"))
        ]
        data_files = sorted(data_files)
        if max_files is not None:
            data_files = data_files[:max_files]

        spectra = []
        for name in data_files:
            with zf.open(name) as f:
                spectra.append(np.loadtxt(f, usecols=[-1]))

    sample_ids = [Path(name).stem for name in data_files]
    return np.vstack(spectra), sample_ids


def _preprocess_features(features: np.ndarray, peak_cap: float = 10000.0) -> np.ndarray:
    capped = features.astype(float, copy=True)
    over = capped > peak_cap
    if np.any(over):
        capped[over] = peak_cap + np.power(capped[over], 0.2)
    col_max = np.max(capped, axis=0)
    return capped / (0.0001 + col_max)


def _normalize_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label_min = labels.min(axis=0)
    label_max = labels.max(axis=0)
    denom = np.where(label_max - label_min > 0, label_max - label_min, 1.0)
    return (labels - label_min) / denom, label_min, label_max


def _split_indices(n_total: int, n_real: int = 216, n_train_real: int = 150) -> tuple[list[int], list[int]]:
    train_idx = list(range(0, n_train_real)) + list(range(n_real, n_total))
    test_idx = list(range(n_train_real + 1, n_real))
    return train_idx, test_idx


def _build_reference_model(input_dim: int, activation: str = "gelu", dropout_rate: float = 0.01) -> Model:
    x_in = layers.Input(shape=(input_dim,), name="naa_ann_ref_input")
    head = x_in[:, :min(100, input_dim)]
    x = layers.Dense(20, activation=activation)(head)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(5, activation=activation)(x)
    outputs = [layers.Dense(1, name=f"target_{i}")(x) for i in range(3)]
    return Model(x_in, outputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NAA-ANN-1 4e reference workflow.")
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path(
            "testing/NAA-ANN-1/data augmentation code/versions/2022-04-27/"
            "NAA2 data augmentation output 4e.zip"
        ),
        help="Path to the NAA-ANN-1 4e zip bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("FluxForge/artifacts/validation/naa_ann_4e"),
        help="Output directory for reference artifacts.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional sample cap.")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs.")
    parser.add_argument("--patience", type=int, default=120, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument(
        "--no-swap-lod-unc",
        action="store_true",
        help="Disable the notebook's LOD/UNC prediction ordering swap.",
    )
    args = parser.parse_args()

    _set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    exp = _load_csv_tables(args.zip)
    spectra, sample_ids = _load_counts(args.zip, max_files=args.max_files)
    if args.max_files is not None:
        exp = exp.iloc[:spectra.shape[0]]

    labels = exp[LABEL_COLS].to_numpy(dtype=float)
    features = np.column_stack([exp[EXP_COLS].to_numpy(dtype=float), spectra])

    features = _preprocess_features(features)
    labels_norm, label_min, label_max = _normalize_labels(labels)

    train_idx, test_idx = _split_indices(features.shape[0])
    X_train = features[train_idx]
    y_train = labels_norm[train_idx]
    X_test = features[test_idx]
    y_test = labels_norm[test_idx]

    model = _build_reference_model(features.shape[1])
    model.compile(loss=["mse", "mse", "mse"], optimizer="adam", loss_weights=[1.0, 0.02, 0.1])

    checkpoint_path = args.output_dir / "reference_model.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=args.patience, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", mode="min", save_best_only=True),
    ]

    model.fit(
        X_train,
        [y_train[:, 0], y_train[:, 1], y_train[:, 2]],
        validation_data=(X_test, [y_test[:, 0], y_test[:, 1], y_test[:, 2]]),
        epochs=args.epochs,
        batch_size=10,
        verbose=0,
        callbacks=callbacks,
    )

    if checkpoint_path.exists():
        model = load_model(checkpoint_path)

    preds_norm = model.predict(features, batch_size=100, verbose=0)
    preds_norm = np.column_stack([p.reshape(-1) for p in preds_norm])
    denom = np.where(label_max - label_min > 0, label_max - label_min, 1.0)
    preds = preds_norm * denom + label_min

    results = pd.DataFrame({"spectra": sample_ids})
    results["Se_real"] = labels[:, 0]
    results["Se_predicted"] = preds[:, 0]
    results["LOD_real"] = labels[:, 2]
    results["UNC_real"] = labels[:, 1]

    if not args.no_swap_lod_unc:
        results["LOD_predicted"] = preds[:, 1]
        results["UNC_predicted"] = preds[:, 2]
    else:
        results["LOD_predicted"] = preds[:, 2]
        results["UNC_predicted"] = preds[:, 1]

    results["mask"] = "train"
    results.loc[test_idx, "mask"] = "test"
    results["Se_ratio"] = results["Se_predicted"] / results["Se_real"].replace(0.0, np.nan)

    output_path = args.output_dir / "reference_results.csv"
    results.to_csv(output_path, index=False)
    print(f"Wrote reference results to {output_path}")


if __name__ == "__main__":
    main()
