"""Quaternion-valued model implementations."""

from .shared import (
    OptimizerName,
    quaternion_adagrad_step_,
    quaternion_adadelta_step_,
    quaternion_mae_loss,
    quaternion_sgd_step_,
    scalar_regression_metrics,
)
from .cmlp import CMLPQuaternionMLP
from .ghr import (
    GHRQuaternionMLP,
)

__all__ = [
    "CMLPQuaternionMLP",
    "GHRQuaternionMLP",
    "OptimizerName",
    "quaternion_adagrad_step_",
    "quaternion_adadelta_step_",
    "quaternion_mae_loss",
    "quaternion_sgd_step_",
    "scalar_regression_metrics",
]
