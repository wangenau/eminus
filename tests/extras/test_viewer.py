# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test viewer extra."""

import inspect
import os
import pathlib

import pytest

from eminus.extras.viewer import executed_in_notebook


@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    [
        "test_plot_bandstructure_dos",
        "test_view_atoms",
        "test_view_contour",
        "test_view_file",
        "test_view_kpts",
    ],
)
def test_viewer(name):
    """Test the execution of a given Jupyter notebook."""
    pytest.importorskip("nglview", reason="nglview not installed, skip tests")
    pytest.importorskip("plotly", reason="plotly not installed, skip tests")
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbformat import read

    file_path = pathlib.Path(inspect.stack()[0][1]).parent
    os.environ["PYTEST_TEST_DIR"] = str(file_path)
    with open(f"{file_path}/{name}.ipynb", encoding="utf-8") as fh:
        nb = read(fh, as_version=4)  # type: ignore[no-untyped-call]
        ep = ExecutePreprocessor(timeout=60)  # type: ignore[no-untyped-call]
        assert ep.preprocess(nb) is not None


def test_executed_in_notebook():
    """Test the notebook differentiation from a terminal."""
    assert not executed_in_notebook()


if __name__ == "__main__":
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
