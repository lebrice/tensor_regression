import contextlib
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from tensor_regression import TensorRegressionFixture


@pytest.fixture
def device(request: pytest.FixtureRequest) -> torch.device:
    device = getattr(request, "param", None)
    if device:
        assert isinstance(device, torch.device | str)
        # a device was specified with indirect parametrization.
        return torch.device(device) if isinstance(device, str) else device
    if torch.cuda.is_available():
        return torch.device("cuda", index=torch.cuda.current_device())
    return torch.device("cpu")


@contextlib.contextmanager
def seeded(seed: int, devices: list[torch.device] | None = None):
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    if devices is None:
        devices = [
            torch.device("cuda", index=i) for i in range(torch.cuda.device_count())
        ]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        yield

    random.setstate(random_state)
    np.random.set_state(np_random_state)


@pytest.fixture(autouse=True)
def seed(request: pytest.FixtureRequest, device: torch.device):
    seed: int = getattr(request, "param", 123)
    with seeded(seed=seed):
        yield seed


@pytest.mark.parametrize("label", [None, "some_label"], ids="label={}".format)
def test_check_tensor(
    tensor_regression: TensorRegressionFixture,
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    label: str | None,
    ndarrays_regression: NDArraysRegressionFixture,
):
    # Make it so our tensor regression fixture operates within a temporary directory, instead of
    # the actual test data dir (next to this test).
    monkeypatch.setattr(tensor_regression, "original_datadir", tmp_path)
    monkeypatch.setattr(tensor_regression, "generate_missing_files", True)

    x = torch.from_numpy(np.random.default_rng(seed=123).random(size=(3, 3))).to(
        device=device
    )
    tensor_regression.check(
        {"x": x}, additional_label=label, include_gpu_name_in_stats=False
    )

    # Check that stats were saved:
    # TODO: It's a bit confusing that the file name include `label_{label}`, but that's because
    # this test is parametrized.
    if label is None:
        stats_file = tmp_path / test_check_tensor.__name__ / f"label_{label}.yaml"
    else:
        stats_file = (
            tmp_path / test_check_tensor.__name__ / label / f"label_{label}.yaml"
        )

    assert stats_file.exists()

    ndarrays_regression.check({"x": x.detach().cpu().numpy()})

    # Check the saved tensor/array:
    array_file = stats_file.with_suffix(".npz")
    saved_x = np.load(array_file)["x"]
    np.testing.assert_equal(saved_x, x.cpu().numpy())

    # Check that a .gitignore file was added:
    gitignore_file = tmp_path / ".gitignore"
    assert gitignore_file.exists()
    assert "*.npz" in gitignore_file.read_text().splitlines()


def test_non_parametrized_test(
    tensor_regression: TensorRegressionFixture,
    seed: int,
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    # Make it so our tensor regression fixture operates within a temporary directory, instead of
    # the actual test data dir (next to this test).
    monkeypatch.setattr(tensor_regression, "original_datadir", tmp_path)
    monkeypatch.setattr(tensor_regression, "generate_missing_files", True)

    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(3, 3, generator=gen, device=device)
    tensor_regression.check({"x": x}, include_gpu_name_in_stats=False)
    stats_file = tmp_path / f"{test_non_parametrized_test.__name__}.yaml"
    assert stats_file.exists()

    tensor_regression.check(
        {"x": x.cpu()}, additional_label="cpu", include_gpu_name_in_stats=False
    )
    stats_file = tmp_path / test_non_parametrized_test.__name__ / "cpu.yaml"
    assert stats_file.exists()
