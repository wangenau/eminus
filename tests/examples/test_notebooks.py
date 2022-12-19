#!/usr/bin/env python3
'''Test functionality of example notebooks.'''
import inspect
import os
import pathlib

from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import read
import pytest


@pytest.mark.parametrize('name', ['08_visualizer_extra', '10_domain_generation'])
def test_notebooks(name):
    '''Test the execution of a given Jupyter notebook.'''
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe())).parent
    os.chdir(file_path.joinpath(f'../../examples/{name}'))

    with open(f'{name}.ipynb', 'r') as fh:
        nb = read(fh, as_version=4)
        ep = ExecutePreprocessor(timeout=60, kernel_name='python3')
        assert ep.preprocess(nb) is not None
    return


if __name__ == '__main__':
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
