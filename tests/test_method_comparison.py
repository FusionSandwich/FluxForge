from __future__ import annotations

from fluxforge.analysis.method_comparison import (
    compute_pairwise_method_metrics,
    serialize_pairwise_method_metrics,
    spearman_rank_correlation,
    top_k_overlap,
)


def test_spearman_rank_correlation_identity_is_one() -> None:
    rank_map = {"a": 1, "b": 2, "c": 3, "d": 4}
    assert spearman_rank_correlation(rank_map, rank_map) == 1.0


def test_spearman_rank_correlation_reversed_is_negative_one() -> None:
    left = {"a": 1, "b": 2, "c": 3, "d": 4}
    right = {"a": 4, "b": 3, "c": 2, "d": 1}
    assert spearman_rank_correlation(left, right) == -1.0


def test_top_k_overlap_counts_shared_top_entries() -> None:
    left = {"a": 1, "b": 2, "c": 3, "d": 4}
    right = {"a": 1, "c": 2, "b": 3, "d": 4}
    assert top_k_overlap(left, right, k=2) == 1


def test_compute_pairwise_method_metrics_and_serialization() -> None:
    rank_by_method = {
        "M1": {"a": 1, "b": 2, "c": 3},
        "M2": {"a": 2, "b": 1, "c": 3},
        "Legacy": {"a": 1, "b": 2, "c": 3},
    }

    metrics = compute_pairwise_method_metrics(rank_by_method, top_k=2)
    assert len(metrics) == 3
    rows = serialize_pairwise_method_metrics(metrics)
    assert len(rows) == 3
    assert set(rows[0].keys()) == {
        "method_a",
        "method_b",
        "sample_count",
        "spearman_rank_correlation",
        "top_k_overlap",
        "top_k",
    }
