from .fixture import TensorRegressionFixture
from .pytest_plugin import make_torch_deterministic, pytest_addoption, tensor_regression
from .stats import get_simple_attributes

__all__ = [
    "get_simple_attributes",
    "make_torch_deterministic",
    "pytest_addoption",
    "tensor_regression",
    "TensorRegressionFixture",
]
