#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
"""Test viewer extra."""

import inspect
import os
import pathlib

import pytest

from eminus.extras.viewer import executed_in_notebook


@pytest.mark.parametrize('name', ['test_view_atoms', 'test_view_file'])
def test_viewer(name):
    """Test the execution of a given Jupyter notebook."""
    pytest.importorskip('nglview', reason='nglview not installed, skip tests')
    pytest.importorskip('plotly', reason='plotly not installed, skip tests')
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbformat import read

    file_path = pathlib.Path(inspect.stack()[0][1]).parent
    os.environ['PYTEST_TEST_DIR'] = str(file_path)
    with open(f'{file_path}/{name}.ipynb', encoding='utf-8') as fh:
        nb = read(fh, as_version=4)
        ep = ExecutePreprocessor(timeout=60, kernel_name='python3')
        assert ep.preprocess(nb) is not None


def test_executed_in_notebook():
    """Test the notebook differentiation from a terminal."""
    assert not executed_in_notebook()


if __name__ == '__main__':
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
