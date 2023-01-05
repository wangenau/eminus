#!/usr/bin/env python3
'''Test functionality of example scripts.'''
import inspect
import os
import pathlib
import runpy

import pytest


def execute_example(name):
    '''Test the execution of a given Python script.'''
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe())).parent
    os.chdir(file_path.joinpath(f'../../examples/{name}'))

    runpy.run_path(f'{name}.py')
    return


def clean_example(trash):
    '''Clean the example folder after running the script.'''
    for it in trash:
        path = pathlib.Path(it)
        if path.exists():
            path.unlink()
    return


def test_01():
    execute_example('01_installation_test')


@pytest.mark.slow
def test_02():
    execute_example('02_minimal_example')


def test_03():
    execute_example('03_atoms_objects')


def test_04():
    execute_example('04_dft_calculations')


def test_05():
    execute_example('05_input_output')
    clean_example(['CH4.json', 'CH4_density.cube'])


@pytest.mark.slow
def test_06():
    execute_example('06_advanced_functionalities')
    clean_example(['Ne_1.cube', 'Ne_2.cube', 'Ne_3.cube', 'Ne_4.cube'])


@pytest.mark.slow
@pytest.mark.extras
def test_07():
    execute_example('07_fod_extra')
    clean_example(['CH4_FLO_1.cube', 'CH4_FLO_2.cube', 'CH4_FLO_3.cube', 'CH4_FLO_4.cube',
                   'CH4_fods.xyz'])


@pytest.mark.slow
@pytest.mark.extras
def test_09():
    execute_example('09_sic_calculations')


@pytest.mark.slow
def test_11():
    execute_example('11_simpledft_examples')


@pytest.mark.slow
def test_12():
    execute_example('12_germanium_solid')
    clean_example(['Ge_solid_density.cube'])


if __name__ == '__main__':
    import pytest
    file_path = pathlib.Path(inspect.getfile(inspect.currentframe()))
    pytest.main(file_path)
