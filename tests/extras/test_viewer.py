#!/usr/bin/env python3
'''Test viewer extra.'''
import inspect
import os
import pathlib

import pytest


@pytest.mark.extras
@pytest.mark.parametrize('name', ['test_view_atoms', 'test_view_file'])
def test_viewer(name):
    '''Test the execution of a given Jupyter notebook.'''
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbformat import read

    file_path = pathlib.Path(inspect.getfile(inspect.currentframe())).parent
    os.environ['PYTEST_TEST_DIR'] = str(file_path)
    with open(f'{file_path}/{name}.ipynb', 'r') as fh:
        nb = read(fh, as_version=4)
        ep = ExecutePreprocessor(timeout=60, kernel_name='python3')
        assert ep.preprocess(nb) is not None


if __name__ == '__main__':
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
