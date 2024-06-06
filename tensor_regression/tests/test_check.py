import contextlib
import random
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

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
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        yield

    random.setstate(random_state)
    np.random.set_state(np_random_state)


@pytest.fixture(params=[123], ids="seed={}".format)
def seed(request: pytest.FixtureRequest, device: torch.device):
    seed: int = getattr(request, "param")
    with seeded(seed=seed):
        yield seed


def test_check_tensor(
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

    # Check that stats were saved:
    # todo: use a pytest_regression fixture for the generated stats instead of hard-coding them? Or
    # would that become too confusing?
    stats_file = tmp_path / test_check_tensor.__name__ / f"seed_{seed}.yaml"
    assert stats_file.exists()
    assert yaml.safe_load(stats_file.read_text()) == yaml.safe_load(
        textwrap.dedent(
            f"""\
            x:
              device: {device}
              hash: 4237770438332455960
              max: 1.573499083518982
              mean: 0.41351261734962463
              min: -1.6879240274429321
              shape:
              - 3
              - 3
              sum: 3.7216134071350098
            """
        )
    )

    # Check the saved tensor/array:
    array_file = tmp_path / test_check_tensor.__name__ / f"seed_{seed}.npz"
    saved_x = np.load(array_file)["x"]
    np.testing.assert_equal(saved_x, x.cpu().numpy())

    # Check that a .gitignore file was added:
    gitignore_file = tmp_path / ".gitignore"
    assert gitignore_file.exists()
    assert "*.npz" in gitignore_file.read_text().splitlines()
