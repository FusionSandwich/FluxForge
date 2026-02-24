#!/usr/bin/env python3
"""
NAA-ANN-1 (4e) parity run.

Trains a FluxForge ANN on the 4e dataset and compares predictions against
the published NAA-ANN-1 results CSV.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from fluxforge.analysis.naa_ann import (
    HAS_TENSORFLOW,
    build_naa_ann4e_model,
    load_naa_ann4e_dataset,
    prepare_naa_ann4e_dataset,
    split_naa_ann4e_dataset,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def _stack_outputs(predictions: list[np.ndarray]) -> np.ndarray:
    return np.column_stack([p.reshape(-1) for p in predictions])


def _detect_reference_swap(reference: pd.DataFrame) -> bool:
    def _corr(a: pd.Series, b: pd.Series) -> float:
        return a.corr(b)

    corr_unc = _corr(reference["UNC_predicted"], reference["UNC_real"])
    corr_lod = _corr(reference["LOD_predicted"], reference["LOD_real"])
    corr_unc_swap = _corr(reference["UNC_predicted"], reference["LOD_real"])
    corr_lod_swap = _corr(reference["LOD_predicted"], reference["UNC_real"])

    return (corr_unc_swap > corr_unc) and (corr_lod_swap > corr_lod)


def main() -> None:
    parser = argparse.ArgumentParser(description="NAA-ANN-1 4e parity runner.")
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("testing/NAA-ANN-1/data augmentation code/versions/2022-04-27/NAA2 data augmentation output 4e.zip"),
        help="Path to the NAA-ANN-1 4e zip bundle.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("testing/NAA-ANN-1/ANN code/code and results versions/2022-05-05 4e/NAA1 2022-05-09 4e results.csv"),
        help="Reference results CSV from NAA-ANN-1.",
    )
    parser.add_argument(
        "--reference-results",
        type=Path,
        default=Path("FluxForge/artifacts/validation/naa_ann_4e/reference_results.csv"),
        help="Reference results CSV generated from the notebook workflow.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("FluxForge/artifacts/validation/naa_ann_4e"),
        help="Output directory for parity artifacts.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional sample cap.")
    parser.add_argument("--epochs", type=int, default=2000, help="Training epochs.")
    parser.add_argument("--patience", type=int, default=120, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    args = parser.parse_args()

    if not HAS_TENSORFLOW:
        raise RuntimeError("TensorFlow is required to run the parity training.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    _set_seed(args.seed)

    dataset = load_naa_ann4e_dataset(args.zip, max_files=args.max_files)
    features, labels = prepare_naa_ann4e_dataset(dataset)
    X_train, y_train, X_test, y_test = split_naa_ann4e_dataset(features, labels)

    model = build_naa_ann4e_model(features.shape[1])
    model.compile(
        loss=["mse", "mse", "mse"],
        optimizer="adam",
        loss_weights=[1.0, 0.02, 0.1],
    )

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.models import load_model

    checkpoint_path = args.output_dir / "best_model.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=args.patience, verbose=1),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", mode="min", save_best_only=True),
    ]

    history = model.fit(
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

    preds_norm = _stack_outputs(model.predict(features, batch_size=100, verbose=0))
    denom = np.where(dataset.label_max - dataset.label_min > 0, dataset.label_max - dataset.label_min, 1.0)
    preds = preds_norm * denom + dataset.label_min

    reference_path = args.reference_results if args.reference_results.exists() else args.results
    reference = pd.read_csv(reference_path)
    predicted = pd.DataFrame(
        {
            "spectra": dataset.sample_ids,
            "Se_pred_fluxforge": preds[:, 0],
            "UNC_pred_fluxforge": preds[:, 1],
            "LOD_pred_fluxforge": preds[:, 2],
        }
    )

    swap_reference = _detect_reference_swap(reference)
    merged = reference.merge(predicted, on="spectra", how="inner")
    merged["reference_source"] = reference_path.name
    merged["reference_swap_lod_unc"] = bool(swap_reference)

    if swap_reference:
        merged["UNC_pred_reference"] = merged["LOD_predicted"]
        merged["LOD_pred_reference"] = merged["UNC_predicted"]
    else:
        merged["UNC_pred_reference"] = merged["UNC_predicted"]
        merged["LOD_pred_reference"] = merged["LOD_predicted"]
    merged["Se_pred_reference"] = merged["Se_predicted"]

    for col in ("Se", "UNC", "LOD"):
        ref_col = f"{col}_pred_reference"
        ff_col = f"{col}_pred_fluxforge"
        diff = merged[ff_col] - merged[ref_col]
        denom = merged[ref_col].replace(0.0, np.nan)
        merged[f"{col}_pct_diff"] = 100.0 * diff / denom

    merged.to_csv(args.output_dir / "naa_ann_4e_parity.csv", index=False)

    summary = {}
    for col in ("Se", "UNC", "LOD"):
        pct = merged[f"{col}_pct_diff"].abs()
        summary[col] = {
            "count": int(pct.notna().sum()),
            "mean_abs_pct": float(pct.mean(skipna=True)),
            "max_abs_pct": float(pct.max(skipna=True)),
        }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Reference source: {reference_path}")
    print(f"Detected LOD/UNC swap: {swap_reference}")
    print("Parity summary:")
    for key, stats in summary.items():
        print(f"  {key}: mean={stats['mean_abs_pct']:.2f}% max={stats['max_abs_pct']:.2f}%")


if __name__ == "__main__":
    main()
