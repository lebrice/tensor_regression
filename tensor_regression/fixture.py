import contextlib
import os
import re
import warnings
from collections.abc import Mapping
from logging import getLogger as get_logger
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from _pytest.outcomes import Failed
from pytest_regressions.data_regression import DataRegressionFixture
from pytest_regressions.ndarrays_regression import NDArraysRegressionFixture
from torch import Tensor

from .flatten import flatten_dict
from .stats import get_simple_attributes
from .to_array import to_ndarray

logger = get_logger(__name__)

PRECISION: int | None = None
"""Number of decimals used when rounding the simple stats of Tensor / ndarray in the pre-check.

Full precision is used in the actual regression check, but this is just for the simple attributes
(min, max, mean, etc.) which seem to be slightly different on the GitHub CI than on a local
machine.

TODO: The way rounding is done here is actually very dumb. round(1e-5, 3) gives 0.000.
"""


def get_version_controlled_attributes(
    data_dict: dict[str, Any], precision: int | None
) -> dict[str, Any]:
    return {
        key: get_simple_attributes(value, precision=precision)
        for key, value in data_dict.items()
    }


class TensorRegressionFixture:
    """Save some statistics (and a hash) of tensors in a file that is saved with git, but save the
    entire tensors in gitignored files.

    This way, the first time the tests run, they re-generate the full regression files, and check
    that their contents' hash matches what is stored with git!

    TODO: Add a `--regen-missing` option (currently implicitly always true) that decides if we
    raise an error if a file is missing. (for example in unit tests we don't want this to be true!)
    """

    def __init__(
        self,
        datadir: Path,
        original_datadir: Path,
        request: pytest.FixtureRequest,
        ndarrays_regression: NDArraysRegressionFixture,
        data_regression: DataRegressionFixture,
        monkeypatch: pytest.MonkeyPatch,
        simple_attributes_precision: int | None = PRECISION,
    ) -> None:
        self.request = request
        self.datadir = datadir
        self.original_datadir = original_datadir

        self.ndarrays_regression = ndarrays_regression
        self.data_regression = data_regression
        self.monkeypatch = monkeypatch
        self.simple_attributes_precision = simple_attributes_precision
        self.generate_missing_files: bool | None = self.request.config.getoption(
            "--gen-missing",  # type: ignore
        )

    def get_source_file(
        self, extension: str, additional_subfolder: str | None = None
    ) -> Path:
        source_file, _test_file = get_test_source_and_temp_file_paths(
            extension=extension,
            request=self.request,
            original_datadir=self.original_datadir,
            datadir=self.datadir,
            additional_subfolder=additional_subfolder,
        )
        return source_file

    # Would be nice if this were a singledispatch method or something similar.

    def add_gitignore_file_if_needed(self):
        current_test_data_dir = self.original_datadir

        # For-elses can sometimes be useful:
        # Try to find an existing .gitignore
        for dir in current_test_data_dir.parents:
            if (gitignore_file := (dir / ".gitignore")).exists():
                break
        else:
            # Try to find a README.md, if found, put the .gitignore next to it.
            for dir in current_test_data_dir.parents:
                if (dir / "README.md").exists():
                    gitignore_file = dir / ".gitignore"
                    break
            else:
                # Worst case, create a .gitignore file in the test data directory (nest to the test
                # module).
                gitignore_file = current_test_data_dir / ".gitignore"

        if not gitignore_file.exists():
            logger.info(f"Making a new .gitignore file at {gitignore_file}")
            gitignore_file.write_text(
                "\n".join(
                    [
                        "# Ignore regression files containing tensors.",
                        "*.npz",
                    ]
                )
                + "\n"
            )
            return

        if "*.npz" in gitignore_file.read_text():
            logger.debug(
                "There is already an entry for npz files in the gitignore file."
            )
            return

        logger.info(f"Adding some lines to the .gitignore file at {gitignore_file}")
        with gitignore_file.open("a") as f:
            f.writelines(
                [
                    "# Ignore regression files containing tensors.",
                    "*.npz",
                    "",
                ]
            )

    def check(
        self,
        data_dict: Mapping[str, Any],
        tolerances: dict[str, dict[str, float]] | None = None,
        default_tolerance: dict[str, float] | None = None,
        include_gpu_name_in_stats: bool = True,
    ) -> None:
        # IDEA:
        # - Get the hashes of each array, and actually run the regression check first with those files.
        # - Then, if that check passes, run the actual check with the full files.
        # NOTE: If the array hash files exist, but the full files don't, then we should just
        # re-create the full files instead of failing.
        # __tracebackhide__ = True
        self.add_gitignore_file_if_needed()
        data_dict = flatten_dict(data_dict)

        if not isinstance(data_dict, dict):
            raise TypeError(
                "Only dictionaries with Tensors, NumPy arrays or array-like objects are "
                "supported on ndarray_regression fixture.\n"
                f"Object with type '{str(type(data_dict))}' was given."
            )

        # File some simple attributes of the full arrays/tensors. This one is saved with git.
        simple_attributes_source_file = self.get_source_file(extension=".yaml")

        # File with the full arrays/tensors. This one is ignored by git.
        arrays_source_file = self.get_source_file(extension=".npz")

        regen_all = self.request.config.getoption("regen_all")
        assert isinstance(regen_all, bool)

        if regen_all:
            assert self.generate_missing_files in [
                True,
                None,
            ], "--gen-missing contradicts --regen-all!"
            # Regenerate everything.
            if arrays_source_file.exists():
                arrays_source_file.unlink()
            if simple_attributes_source_file.exists():
                simple_attributes_source_file.unlink()

        if arrays_source_file.exists():
            logger.info(f"Full arrays file found at {arrays_source_file}.")
            if not simple_attributes_source_file.exists():
                # Weird: the simple attributes file doesn't exist. Re-create it if allowed.
                with dont_fail_if_files_are_missing(
                    enabled=bool(self.generate_missing_files)
                ):
                    self.pre_check(
                        data_dict,
                        simple_attributes_source_file=simple_attributes_source_file,
                        include_gpu_name_in_stats=include_gpu_name_in_stats,
                    )

            # We already generated the file with the full tensors (and we also already checked
            # that their hashes correspond to what we expect.)
            # 1. Check that they match the data_dict.
            logger.info("Checking the full arrays.")
            self.regular_check(
                data_dict=data_dict,
                fullpath=arrays_source_file,
                tolerances=tolerances,
                default_tolerance=default_tolerance,
            )
            # the simple attributes file should already have been generated and saved in git.
            assert simple_attributes_source_file.exists()
            # NOTE: No need to do this step here. Saves us a super super tiny amount of time.
            # logger.debug("Checking that the hashes of the full arrays still match.")
            # self.pre_check(
            #     data_dict,
            #     simple_attributes_source_file=simple_attributes_source_file,
            # )
            return

        if simple_attributes_source_file.exists():
            logger.debug(
                f"Simple attributes file found at {simple_attributes_source_file}."
            )
            logger.debug(f"Regenerating the full arrays at {arrays_source_file}")
            # Go straight to the full check.
            # TODO: Need to get the full error when the tensors change instead of just the check
            # for the hash, which should only be used when re-creating the full regression files.

            with dont_fail_if_files_are_missing():
                self.regular_check(
                    data_dict=data_dict,
                    fullpath=arrays_source_file,
                    tolerances=tolerances,
                    default_tolerance=default_tolerance,
                )
            logger.debug(
                "Checking if the newly-generated full tensor regression files match the expected "
                "attributes and hashes."
            )
            self.pre_check(
                data_dict,
                simple_attributes_source_file=simple_attributes_source_file,
                include_gpu_name_in_stats=include_gpu_name_in_stats,
            )
            return

        logger.warning(
            f"Creating the simple attributes file at {simple_attributes_source_file}."
        )

        with dont_fail_if_files_are_missing(enabled=bool(self.generate_missing_files)):
            self.pre_check(
                data_dict,
                simple_attributes_source_file=simple_attributes_source_file,
                include_gpu_name_in_stats=include_gpu_name_in_stats,
            )

        with dont_fail_if_files_are_missing(enabled=bool(self.generate_missing_files)):
            self.regular_check(
                data_dict=data_dict,
                fullpath=arrays_source_file,
                tolerances=tolerances,
                default_tolerance=default_tolerance,
            )

    def pre_check(
        self,
        data_dict: dict[str, Any],
        simple_attributes_source_file: Path,
        include_gpu_name_in_stats: bool,
    ) -> None:
        version_controlled_simple_attributes = get_version_controlled_attributes(
            data_dict, precision=self.simple_attributes_precision
        )
        # Run the regression check with the hashes (and don't fail if they don't exist)
        __tracebackhide__ = True
        if include_gpu_name_in_stats:
            # TODO: Figure out how to include/use the names of the GPUs:
            # - Should it be part of the hash? Or should there be a subfolder for each GPU type?
            _gpu_names = sorted(set(get_gpu_names(data_dict)))
            if len(_gpu_names) == 1:
                version_controlled_simple_attributes["GPU"] = _gpu_names[0]
            elif _gpu_names:
                version_controlled_simple_attributes["GPUS"] = _gpu_names

        self.data_regression.check(
            version_controlled_simple_attributes, fullpath=simple_attributes_source_file
        )

    def regular_check(
        self,
        data_dict: dict[str, Any],
        basename: str | None = None,
        fullpath: os.PathLike[str] | None = None,
        tolerances: dict[str, dict[str, float]] | None = None,
        default_tolerance: dict[str, float] | None = None,
    ) -> None:
        array_dict: dict[str, np.ndarray] = {}
        for key, array in data_dict.items():
            if isinstance(key, (int | bool | float)):
                new_key = f"{key}"
                assert new_key not in data_dict
                key = new_key
            assert isinstance(
                key, str
            ), f"The dictionary keys must be strings. Found key with type '{str(type(key))}'"

            ndarray_value = to_ndarray(array)
            if ndarray_value is None:
                logger.debug(
                    f"Got a value of `None` for key {key} not including it in the saved dict."
                )
            else:
                array_dict[key] = ndarray_value
        self.ndarrays_regression.check(
            array_dict,
            basename=basename,
            fullpath=fullpath,
            tolerances=tolerances,
            default_tolerance=default_tolerance,
        )
        return


def get_test_source_and_temp_file_paths(
    extension: str,
    request: pytest.FixtureRequest,
    original_datadir: Path,
    datadir: Path,
    additional_subfolder: str | None = None,
) -> tuple[Path, Path]:
    """Returns the path to the (maybe version controlled) source file and the path to the temporary
    file where test results might be generated during a regression test.

    NOTE: This is different than in pytest-regressions. Here we use a subfolder with the same name
    as the test function.
    """
    basename = re.sub(r"[\W]", "_", request.node.name)
    overrides_name = basename.removeprefix(request.node.function.__name__).lstrip("_")

    if extension.startswith(".") and overrides_name:
        # Remove trailing _'s if the extension starts with a dot.
        overrides_name = overrides_name.rstrip("_")

    if overrides_name:
        # There are overrides, so use a subdirectory.
        relative_path = Path(request.node.function.__name__) / overrides_name
    else:
        # There are no overrides, so use the regular base name.
        relative_path = Path(basename)

    relative_path = relative_path.with_suffix(extension)
    if additional_subfolder:
        relative_path = relative_path.parent / additional_subfolder / relative_path.name

    source_file = original_datadir / relative_path
    test_file = datadir / relative_path
    return source_file, test_file


def get_gpu_names(data_dict: dict[str, Any]) -> list[str]:
    """Returns the names of the GPUS that tensors in this dict are on."""
    return sorted(
        set(
            torch.cuda.get_device_name(tensor.device)
            for tensor in flatten_dict(data_dict).values()
            if isinstance(tensor, Tensor) and tensor.device.type == "cuda"
        )
    )


class FilesDidntExist(Failed):
    pass


@contextlib.contextmanager
def dont_fail_if_files_are_missing(enabled: bool = True):
    try:
        with _catch_fails_with_files_didnt_exist():
            yield
    except FilesDidntExist as exc:
        if enabled:
            logger.warning(exc)
            warnings.warn(RuntimeWarning(exc.msg))
        else:
            raise


@contextlib.contextmanager
def _catch_fails_with_files_didnt_exist():
    try:
        yield
    except Failed as failure_exc:
        if (
            failure_exc.msg
            and "File not found in data directory, created" in failure_exc.msg
        ):
            raise FilesDidntExist(
                failure_exc.msg
                + "\n(Use the --gen-missing flag to create any missing regression files.)",
                pytrace=failure_exc.pytrace,
            ) from failure_exc
        else:
            raise
