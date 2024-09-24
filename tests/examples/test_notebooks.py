# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test functionality of example notebooks."""

from __future__ import annotations

import inspect
import os
import pathlib

import pytest


def clean_example(trash: list[str]) -> None:
    """Clean the example folder after running the script."""
    for it in trash:
        path = pathlib.Path(it)
        if path.exists():
            path.unlink()


@pytest.mark.slow
@pytest.mark.parametrize(
    ("name", "trash"),
    [
        ("08_visualizer_extra", []),
        ("10_domain_generation", []),
        (
            "12_wannier_localization",
            ["CH4_WO_k0_0.cube", "CH4_WO_k0_1.cube", "CH4_WO_k0_2.cube", "CH4_WO_k0_3.cube"],
        ),
        ("19_band_structures", []),
    ],
)
def test_notebooks(name, trash):
    """Test the execution of a given Jupyter notebook."""
    pytest.importorskip("nglview", reason="nglview not installed, skip tests")
    pytest.importorskip("plotly", reason="plotly not installed, skip tests")
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbformat import read

    file_path = pathlib.Path(inspect.stack()[0][1]).parent
    os.chdir(file_path.joinpath(f"../../examples/{name}"))

    with open(f"{name}.ipynb", encoding="utf-8") as fh:
        nb = read(fh, as_version=4)
        ep = ExecutePreprocessor(timeout=60, kernel_name="python3")
        assert ep.preprocess(nb) is not None

    clean_example(trash)


if __name__ == "__main__":
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
