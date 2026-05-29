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
from .classical_mlp import ClassicalMLP
from .cpcn import QuaternionPCNLayer, SplitQuaternionPredictiveCodingNetwork
from .ghr import (
    GHRQuaternionMLP,
)
from .pcn import (
    PCNLayer,
    PredictiveCodingNetwork,
    train_pcn,
)

__all__ = [
    "CMLPQuaternionMLP",
    "ClassicalMLP",
    "GHRQuaternionMLP",
    "QuaternionPCNLayer",
    "SplitQuaternionPredictiveCodingNetwork",
    "OptimizerName",
    "quaternion_adagrad_step_",
    "quaternion_adadelta_step_",
    "quaternion_mae_loss",
    "quaternion_sgd_step_",
    "scalar_regression_metrics",
    "PCNLayer",
    "PredictiveCodingNetwork",
    "train_pcn",
]
