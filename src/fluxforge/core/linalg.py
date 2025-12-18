"""Lightweight linear algebra helpers that avoid external dependencies."""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Sequence, Tuple

Vector = List[float]
Matrix = List[List[float]]


def to_vector(values: Iterable[float]) -> Vector:
    return [float(v) for v in values]


def zeros_matrix(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def identity(size: int) -> Matrix:
    m = zeros_matrix(size, size)
    for i in range(size):
        m[i][i] = 1.0
    return m


def transpose(mat: Matrix) -> Matrix:
    return [list(row) for row in zip(*mat)]


def matmul(a: Matrix, b: Matrix | Vector) -> Matrix | Vector:
    if isinstance(b[0], list):  # type: ignore[index]
        b_mat: Matrix = b  # type: ignore[assignment]
        result = zeros_matrix(len(a), len(b_mat[0]))
        for i in range(len(a)):
            for k in range(len(b_mat)):
                for j in range(len(b_mat[0])):
                    result[i][j] += a[i][k] * b_mat[k][j]
        return result
    else:
        b_vec: Vector = b  # type: ignore[assignment]
        result_vec: Vector = [0.0 for _ in range(len(a))]
        for i in range(len(a)):
            for k in range(len(b_vec)):
                result_vec[i] += a[i][k] * b_vec[k]
        return result_vec


def add_vectors(a: Vector, b: Vector) -> Vector:
    return [x + y for x, y in zip(a, b)]


def sub_vectors(a: Vector, b: Vector) -> Vector:
    return [x - y for x, y in zip(a, b)]


def scalar_multiply(vec: Vector, scalar: float) -> Vector:
    return [scalar * x for x in vec]


def diag(values: Vector) -> Matrix:
    m = zeros_matrix(len(values), len(values))
    for i, v in enumerate(values):
        m[i][i] = v
    return m


def cholesky(mat: Matrix) -> Matrix:
    n = len(mat)
    L = zeros_matrix(n, n)
    for i in range(n):
        for j in range(i + 1):
            sum_val = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                diff = mat[i][i] - sum_val
                if diff <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i][j] = math.sqrt(diff)
            else:
                L[i][j] = (mat[i][j] - sum_val) / L[j][j]
    return L


def invert_square_matrix(mat: Matrix) -> Matrix:
    n = len(mat)
    aug = [row[:] + identity_row for row, identity_row in zip(mat, identity(n))]
    # Forward elimination
    for col in range(n):
        pivot_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("Matrix is singular")
        aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            for j in range(2 * n):
                aug[r][j] -= factor * aug[col][j]
    return [row[n:] for row in aug]


def gram_matrix(mat: Matrix) -> Matrix:
    return matmul(transpose(mat), mat)  # type: ignore[arg-type]


def pseudo_inverse(mat: Matrix) -> Matrix:
    # Moore-Penrose via normal equations
    gram = gram_matrix(mat)
    gram_inv = invert_square_matrix(gram)
    return matmul(gram_inv, transpose(mat))  # type: ignore[arg-type]


def vector_mean(vec: Vector) -> float:
    return sum(vec) / len(vec)


def vector_average(values: Vector, weights: Vector) -> float:
    weighted_sum = sum(v * w for v, w in zip(values, weights))
    weight_total = sum(weights)
    if weight_total == 0:
        raise ValueError("Weights must not sum to zero")
    return weighted_sum / weight_total


def vector_sum(vec: Vector) -> float:
    return sum(vec)


def elementwise_square(vec: Vector) -> Vector:
    return [v * v for v in vec]


def elementwise_clip(vec: Vector, lower: float = 0.0) -> Vector:
    return [max(lower, v) for v in vec]


def elementwise_maximum(vec: Vector, floor: float) -> Vector:
    return [v if v > floor else floor for v in vec]


def percentile(values: List[Vector], q: float) -> Vector:
    if not values:
        return []
    n_rows = len(values)
    n_cols = len(values[0])
    result: Vector = []
    for col in range(n_cols):
        col_values = sorted(row[col] for row in values)
        rank = (q / 100) * (n_rows - 1)
        lower = int(math.floor(rank))
        upper = int(math.ceil(rank))
        if lower == upper:
            result.append(col_values[lower])
        else:
            interp = col_values[lower] + (col_values[upper] - col_values[lower]) * (rank - lower)
            result.append(interp)
    return result


def multivariate_normal_samples(mean: Vector, cov: Matrix, n: int) -> List[Vector]:
    L = cholesky(cov)
    dim = len(mean)
    samples: List[Vector] = []
    for _ in range(n):
        z = [random.gauss(0.0, 1.0) for _ in range(dim)]
        noise = [sum(L[i][k] * z[k] for k in range(dim)) for i in range(dim)]
        samples.append([m + n_val for m, n_val in zip(mean, noise)])
    return samples
