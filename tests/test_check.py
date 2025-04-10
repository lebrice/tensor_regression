import contextlib
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture

from tensor_regression import TensorRegressionFixture
from tensor_regression.fixture import FilesDidntExist


@contextlib.contextmanager
def seeded(seed: int, devices: list[torch.device] | None = None):
    random_state = random.getstate()
    np_random_state = np.random.get_state()
    if devices is None:
        devices = [torch.device("cuda", index=i) for i in range(torch.cuda.device_count())]
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


@pytest.mark.parametrize("precision", [None, 3], ids="precision={}".format)
@pytest.mark.parametrize("label", [None, "some_label"], ids="label={}".format)
def test_check_tensor(
    tensor_regression: TensorRegressionFixture,
    device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    label: str | None,
    ndarrays_regression: NDArraysRegressionFixture,
    precision: int | None,
):
    # Make it so our tensor regression fixture operates within a temporary directory, instead of
    # the actual test data dir (next to this test).
    monkeypatch.setattr(tensor_regression, "original_datadir", tmp_path)
    monkeypatch.setattr(tensor_regression, "generate_missing_files", True)
    monkeypatch.setattr(tensor_regression, "simple_attributes_precision", precision)

    x = torch.rand(
        3,
        3,
        generator=torch.Generator(device=device).manual_seed(123),
        device=device,
    )
    tensor_regression.check({"x": x}, additional_label=label, include_gpu_name_in_stats=False)

    # Check that stats were saved:
    # TODO: It's a bit confusing that the file name include `label_{label}`, but that's because
    # this test is parametrized.
    if label is None:
        stats_file = (
            tmp_path / test_check_tensor.__name__ / f"label_{label}_precision_{precision}.yaml"
        )
    else:
        stats_file = (
            tmp_path
            / test_check_tensor.__name__
            / label
            / f"label_{label}_precision_{precision}.yaml"
        )

    assert stats_file.exists()

    # ndarrays_regression.check({"x": x.detach().cpu().numpy()})

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


@pytest.mark.xfail(reason="Hashes aren't included in the regression files anymore.")
def test_reproduce_hashing_issue(
    tensor_regression: TensorRegressionFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    ndarrays_regression: NDArraysRegressionFixture,
):
    """BUG:
    - If the simple stats file is present, then its content is effectively ignored!
    --> This causes issues with our inconsistent _hash function for tensors!
    """
    # Make it so our tensor regression fixture operates within a temporary directory, instead of
    # the actual test data dir (next to this test).
    monkeypatch.setattr(tensor_regression, "original_datadir", tmp_path)
    # Reproduce the issue:
    stats_file = (tmp_path / test_reproduce_hashing_issue.__name__).with_suffix(".yaml")
    arrays_file = stats_file.with_suffix(".npz")

    # Create the arrays file.
    from _pytest.outcomes import Failed

    with pytest.raises((AssertionError, Failed)):
        ndarrays_regression.check({"x": np.zeros(1)}, fullpath=arrays_file)

    # Create a stats file with garbage contents
    stats_file.write_text("garbage")

    # this would not raise an error, but it now does:
    with pytest.raises(
        (AssertionError, Failed),
        match="-garbage",
    ):
        tensor_regression.check({"x": torch.zeros(1)}, include_gpu_name_in_stats=False)


@pytest.mark.parametrize("label", [None, "some_label"], ids="label={}".format)
def test_extra_hash_in_file_doesnt_cause_errors(
    tensor_regression: TensorRegressionFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    device: torch.device,
    label: str | None,
):
    """The 'hash' of the array was removed from the simple attributes file. We don't want to
    break backward compatibility, so if there are any extra "hash" keys in the files, they should
    be simply dropped instead of raising an error.
    """

    monkeypatch.setattr(tensor_regression, "original_datadir", tmp_path)
    monkeypatch.setattr(tensor_regression, "generate_missing_files", False)

    x = torch.zeros(1, device=device)
    with pytest.raises(FilesDidntExist):
        tensor_regression.check({"x": x}, additional_label=label, include_gpu_name_in_stats=False)

    # TODO: It's a bit confusing that the file name include `label_{label}`, but that's because
    # this test is parametrized.
    if label is None:
        stats_file = (
            tmp_path / test_extra_hash_in_file_doesnt_cause_errors.__name__ / f"label_{label}.yaml"
        )
    else:
        stats_file = (
            tmp_path
            / test_extra_hash_in_file_doesnt_cause_errors.__name__
            / label
            / f"label_{label}.yaml"
        )

    assert stats_file.exists()

    with open(stats_file, "r") as f:
        stats = yaml.safe_load(f.read())

    assert isinstance(stats, dict)
    assert "hash" not in stats, "The hash key should not be present in the file."
    with open(stats_file, "w") as f:
        f.write(yaml.dump({**stats, "hash": 8923489712398471}))

    # Should complain about the missing 'hash' key.
    tensor_regression.check({"x": x}, additional_label=label, include_gpu_name_in_stats=False)
