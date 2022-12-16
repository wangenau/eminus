#!/usr/bin/env python3
'''Test functionality of example notebooks.'''
import inspect
import os
import pathlib

from nbconvert.preprocessors import ExecutePreprocessor
from nbformat import read


def execute_example(name):
    '''Test the execution of a given Jupyter notebook.'''
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe())).parent
    os.chdir(file_path.joinpath(f'../../examples/{name}'))

    try:
        with open(f'{name}.ipynb', 'r') as fh:
            nb = read(fh, as_version=4)
            ep = ExecutePreprocessor(timeout=60, kernel_name='python3')
            assert ep.preprocess(nb) is not None
    except Exception as err:
        print(f'Test for {name} failed.')
        raise SystemExit(err) from None
    else:
        print(f'Test for {name} passed.')
    return


def test_08():
    execute_example('08_visualizer_extra')


def test_10():
    execute_example('10_domain_generation')


if __name__ == '__main__':
    import time
    start = time.perf_counter()
    test_08()
    test_10()
    end = time.perf_counter()
    print(f'Tests for example notebooks execution passed in {end - start:.3f} s.')
