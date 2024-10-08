"""End-to-end tests for the fixture."""
from tensor_regression import TensorRegressionFixture
import pytest
import torch
from tensor_regression import fixture

from tensor_regression.fixture import get_gpu_names


@pytest.mark.parametrize(
    "include_gpu_in_stats", [False, True], ids="with_gpu={}".format
)
@pytest.mark.parametrize("precision", [None, 3], ids="precision={}".format)
def test_simple_cpu_values(
    tensor_regression: TensorRegressionFixture,
    monkeypatch: pytest.MonkeyPatch,
    precision: int | None,
    include_gpu_in_stats: bool,
):
    monkeypatch.setattr(tensor_regression, "simple_attributes_precision", precision)
    monkeypatch.setattr(fixture, get_gpu_names.__name__, lambda v: ["FAKE_GPU_NAME"])
    tensor_regression.check(
        {
            "a": torch.zeros(1),
            "b": torch.rand(3, 3, generator=torch.Generator().manual_seed(123)),
        },
        include_gpu_name_in_stats=include_gpu_in_stats,
    )


@pytest.mark.parametrize(
    "include_gpu_in_stats", [False, True], ids="with_gpu={}".format
)
@pytest.mark.parametrize("precision", [None, 3], ids="precision={}".format)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA")
def test_simple_GPU_values(
    tensor_regression: TensorRegressionFixture,
    precision: int | None,
    monkeypatch: pytest.MonkeyPatch,
    include_gpu_in_stats: bool,
):
    monkeypatch.setattr(tensor_regression, "simple_attributes_precision", precision)
    monkeypatch.setattr(fixture, get_gpu_names.__name__, lambda v: ["FAKE_GPU_NAME"])

    device = torch.device("cuda")
    data = {
        "a": torch.zeros(1, device=device),
        "b": torch.rand(
            3, 3, generator=torch.Generator(device).manual_seed(123), device=device
        ),
    }
    tensor_regression.check(data, include_gpu_name_in_stats=include_gpu_in_stats)
