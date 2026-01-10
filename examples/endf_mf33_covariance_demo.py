"""Minimal ENDF MF33 covariance ingest + conditioning demo.

This example builds a tiny synthetic ENDF-like MF33 section, parses it via
FluxForge's MF33 helper, validates the covariance matrix, and applies SVD
conditioning.

Run:
  python examples/endf_mf33_covariance_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from fluxforge.data.endf_covariance import (
    read_endf_mf33_section,
    validate_covariance_matrix,
    condition_covariance_svd,
)


def _endf_line(c1, c2, l1, l2, n1, n2, mat, mf, mt, nc=1) -> str:
    return (
        f"{c1:11.5E}{c2:11.5E}{l1:11d}{l2:11d}{n1:11d}{n2:11d}"
        f"{mat:4d}{mf:2d}{mt:3d}{nc:5d}\n"
    )


def _endf_data(vals, mat, mf, mt, nc=2) -> str:
    fields = "".join(f"{v:11.5E}" for v in vals)
    fields = fields.ljust(66)
    return f"{fields}{mat:4d}{mf:2d}{mt:3d}{nc:5d}\n"


def main() -> None:
    mat = 125
    mt = 102

    # LIST values: boundaries (1,10,100,1000) then diagonal cov (0.1,0.2,0.3)
    npl = 7
    text = "".join(
        [
            _endf_line(0.0, 0.0, 0, 0, npl, 0, mat, 33, mt, nc=1),
            _endf_data([1.0, 10.0, 100.0, 1000.0, 0.1, 0.2], mat, 33, mt, nc=2),
            _endf_data([0.3], mat, 33, mt, nc=3),
        ]
    )

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "synthetic.endf"
        p.write_text(text)

        rec = read_endf_mf33_section(p, mat=mat, mt=mt)
        assert rec is not None

        cov = rec.to_covariance_matrix().matrix
        vres = validate_covariance_matrix(cov)

        print("=== MF33 demo ===")
        print(f"Degraded: {rec.degraded}")
        if rec.degraded_warnings:
            print("Warnings:")
            for w in rec.degraded_warnings:
                print(f"  - {w}")
        print(vres.summary())

        cov2, diag = condition_covariance_svd(cov, target_condition=1e6)
        vres2 = validate_covariance_matrix(cov2)

        print("\n=== After SVD conditioning ===")
        print(f"Truncated: {diag['n_truncated']}, new condition ~ {diag['new_condition']:.2e}")
        print(vres2.summary())

        # Show diagonal std devs (sqrt of variances)
        print("\nDiag std devs:")
        print(np.sqrt(np.diag(cov2)))


if __name__ == "__main__":
    main()
