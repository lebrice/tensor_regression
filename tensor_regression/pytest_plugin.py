from argparse import BooleanOptionalAction
from pathlib import Path

import pytest
import torch
from pytest_regressions import data_regression, ndarrays_regression

from .fixture import TensorRegressionFixture


def pytest_addoption(parser: pytest.Parser):
    group = parser.getgroup(
        "tensor_regression",
        description="Options for the tensor_regression plugin.",
        after="regressions",
    )
    group.addoption(
        "--gen-missing",
        action=BooleanOptionalAction,
        help="Whether to generate missing regression files or raise an error when a regression file is missing.",
    )
    group.addoption(
        "--stats-rounding-precision",
        type=int,
        metavar="N_DIGITS",
        help="Number of digits to round simple statistics to. Default is `None` (no rounding).",
    )


@pytest.fixture
def make_torch_deterministic():
    """Set torch to deterministic mode for unit tests that use the tensor_regression fixture."""
    mode_before = torch.get_deterministic_debug_mode()
    torch.set_deterministic_debug_mode("error")
    yield
    torch.set_deterministic_debug_mode(mode_before)


@pytest.fixture
def tensor_regression(
    datadir: Path,
    original_datadir: Path,
    request: pytest.FixtureRequest,
    ndarrays_regression: ndarrays_regression.NDArraysRegressionFixture,
    data_regression: data_regression.DataRegressionFixture,
    monkeypatch: pytest.MonkeyPatch,
    make_torch_deterministic: None,
) -> TensorRegressionFixture:
    """Similar to ndarrays_regression, but with slightly better supports for Tensors.

    See the docstring of `TensorRegressionFixture` for more info.
    """
    rounding: int | None = request.config.getoption(
        "--stats-rounding-precision",
        default=None,  # type: ignore
    )
    return TensorRegressionFixture(
        datadir=datadir,
        original_datadir=original_datadir,
        request=request,
        ndarrays_regression=ndarrays_regression,
        data_regression=data_regression,
        monkeypatch=monkeypatch,
        simple_attributes_precision=rounding,
    )
