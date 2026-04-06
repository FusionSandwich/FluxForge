"""Utilities for comparing optimization-method rankings."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence


@dataclass(frozen=True)
class MethodPairMetric:
    """Pairwise ranking-similarity metrics for two methods."""

    method_a: str
    method_b: str
    sample_count: int
    spearman_rank_correlation: float
    top_k_overlap: int
    top_k: int

    def to_row(self) -> dict[str, float | int | str]:
        return {
            "method_a": self.method_a,
            "method_b": self.method_b,
            "sample_count": int(self.sample_count),
            "spearman_rank_correlation": float(self.spearman_rank_correlation),
            "top_k_overlap": int(self.top_k_overlap),
            "top_k": int(self.top_k),
        }


def spearman_rank_correlation(
    rank_a: Mapping[str, int],
    rank_b: Mapping[str, int],
) -> float:
    """Compute Spearman correlation for two sample->rank mappings."""

    labels = sorted(set(rank_a).intersection(rank_b))
    n = len(labels)
    if n < 2:
        return 1.0
    d_squared = 0.0
    for label in labels:
        delta = float(rank_a[label] - rank_b[label])
        d_squared += delta * delta
    return 1.0 - (6.0 * d_squared) / (n * (n * n - 1.0))


def top_k_overlap(
    rank_a: Mapping[str, int],
    rank_b: Mapping[str, int],
    *,
    k: int,
) -> int:
    """Compute overlap size between top-k sample sets for two methods."""

    top_a = {label for label, rank in rank_a.items() if int(rank) <= int(k)}
    top_b = {label for label, rank in rank_b.items() if int(rank) <= int(k)}
    return len(top_a.intersection(top_b))


def compute_pairwise_method_metrics(
    rank_by_method: Mapping[str, Mapping[str, int]],
    *,
    top_k: int = 5,
) -> tuple[MethodPairMetric, ...]:
    """Compute pairwise similarity metrics across all methods."""

    metrics: list[MethodPairMetric] = []
    method_names = list(rank_by_method.keys())
    for method_a, method_b in combinations(method_names, 2):
        left = rank_by_method.get(method_a, {})
        right = rank_by_method.get(method_b, {})
        shared = sorted(set(left).intersection(right))
        metrics.append(
            MethodPairMetric(
                method_a=method_a,
                method_b=method_b,
                sample_count=len(shared),
                spearman_rank_correlation=spearman_rank_correlation(left, right),
                top_k_overlap=top_k_overlap(left, right, k=top_k),
                top_k=int(top_k),
            )
        )
    return tuple(metrics)


def serialize_pairwise_method_metrics(
    metrics: Sequence[MethodPairMetric],
) -> list[dict[str, float | int | str]]:
    """Serialize pairwise metrics to flat rows for CSV/JSON output."""

    return [item.to_row() for item in metrics]
