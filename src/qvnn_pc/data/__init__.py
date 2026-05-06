"""Data preparation utilities for quaternion-valued regression benchmarks."""

from .dataset import (
    BENCHMARK_BOUNDS,
    BENCHMARK_FUNCTIONS,
    QuaternionRegressionDataSplit,
    make_benchmark_regression_split,
)

__all__ = [
    "BENCHMARK_BOUNDS",
    "BENCHMARK_FUNCTIONS",
    "QuaternionRegressionDataSplit",
    "make_benchmark_regression_split",
]
