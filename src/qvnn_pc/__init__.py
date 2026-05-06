"""Quaternion-valued neural network utilities."""

from .activations import (
    split_leaky_relu,
    split_leaky_relu_derivative,
    split_sigmoid,
    split_sigmoid_derivative,
    split_swish,
    split_swish_derivative,
    split_tanh,
    split_tanh_derivative,
)
from .data import (
    BENCHMARK_BOUNDS,
    BENCHMARK_FUNCTIONS,
    QuaternionRegressionDataSplit,
    make_benchmark_regression_split,
)
from .config import (
    ModelHyperparameters,
    default_hyperparameters_path,
    get_model_hyperparameters,
    load_hyperparameters,
)
from .functions import ackley_4d, griewank_4d, rastrigin_4d, schwefel_4d
from .models import (
    CMLPQuaternionMLP,
    GHRQuaternionMLP,
    OptimizerName,
    quaternion_adagrad_step_,
    quaternion_adadelta_step_,
    quaternion_mae_loss,
    quaternion_sgd_step_,
    scalar_regression_metrics,
)
from .quaternion import (
    hamilton_product,
    q_add,
    q_conj,
    q_hadamard,
    q_linear,
    q_matmul,
    q_mul,
    q_norm,
    q_norm_sq,
    quaternion_abs,
    quaternion_matmul,
    quaternion_mse,
)

__all__ = [
    "BENCHMARK_BOUNDS",
    "BENCHMARK_FUNCTIONS",
    "ArchitectureComparisonConfig",
    "CMLPQuaternionMLP",
    "GHRQuaternionMLP",
    "ModelHyperparameters",
    "OptimizerName",
    "QuaternionRegressionDataSplit",
    "ackley_4d",
    "default_hyperparameters_path",
    "get_model_hyperparameters",
    "griewank_4d",
    "hamilton_product",
    "load_hyperparameters",
    "q_add",
    "q_conj",
    "q_hadamard",
    "q_linear",
    "q_matmul",
    "q_mul",
    "q_norm",
    "q_norm_sq",
    "quaternion_adagrad_step_",
    "quaternion_adadelta_step_",
    "quaternion_mae_loss",
    "quaternion_abs",
    "quaternion_sgd_step_",
    "quaternion_matmul",
    "quaternion_mse",
    "rastrigin_4d",
    "run_architecture_comparison",
    "schwefel_4d",
    "scalar_regression_metrics",
    "make_benchmark_regression_split",
    "split_leaky_relu",
    "split_leaky_relu_derivative",
    "split_sigmoid",
    "split_sigmoid_derivative",
    "split_swish",
    "split_swish_derivative",
    "split_tanh",
    "split_tanh_derivative",
    "write_architecture_comparison",
]


def __getattr__(name: str):
    if name in {"ArchitectureComparisonConfig", "run_architecture_comparison", "write_architecture_comparison"}:
        from .architecture_comparison import (
            ArchitectureComparisonConfig,
            run_architecture_comparison,
            write_architecture_comparison,
        )

        mapping = {
            "ArchitectureComparisonConfig": ArchitectureComparisonConfig,
            "run_architecture_comparison": run_architecture_comparison,
            "write_architecture_comparison": write_architecture_comparison,
        }
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
