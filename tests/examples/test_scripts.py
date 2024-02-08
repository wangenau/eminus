#!/usr/bin/env python3
"""Test functionality of example scripts."""
import inspect
import os
import pathlib
import runpy

import pytest


def execute_example(name):
    """Test the execution of a given Python script."""
    file_path = pathlib.Path(inspect.stack()[0][1]).parent
    os.chdir(file_path.joinpath(f'../../examples/{name}'))

    runpy.run_path(f'{name}.py')


def clean_example(trash):
    """Clean the example folder after running the script."""
    for it in trash:
        path = pathlib.Path(it)
        if path.exists():
            path.unlink()


def test_01():  # noqa: D103
    execute_example('01_installation_test')


@pytest.mark.slow()
def test_02():  # noqa: D103
    execute_example('02_minimal_example')


def test_03():  # noqa: D103
    execute_example('03_atoms_objects')


def test_04():  # noqa: D103
    execute_example('04_dft_calculations')


def test_05():  # noqa: D103
    execute_example('05_input_output')
    clean_example(['CH4.json', 'CH4_density.cube'])


@pytest.mark.slow()
def test_06():  # noqa: D103
    execute_example('06_advanced_functionalities')
    clean_example(['Ne_1.cube', 'Ne_2.cube', 'Ne_3.cube', 'Ne_4.cube'])


@pytest.mark.slow()
def test_07():  # noqa: D103
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    execute_example('07_fod_extra')
    clean_example(['CH4_FLO_1.cube', 'CH4_FLO_2.cube', 'CH4_FLO_3.cube', 'CH4_FLO_4.cube',
                   'CH4_fods.xyz'])


@pytest.mark.slow()
def test_09():  # noqa: D103
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    execute_example('09_sic_calculations')


@pytest.mark.slow()
def test_11():  # noqa: D103
    execute_example('11_germanium_solid')
    clean_example(['Ge_solid_density.cube'])


@pytest.mark.slow()
def test_13():  # noqa: D103
    pytest.importorskip('matplotlib', reason='matplotlib not installed, skip tests')
    execute_example('13_geometry_optimization')


def test_14():  # noqa: D103
    pytest.importorskip('matplotlib', reason='matplotlib not installed, skip tests')
    execute_example('14_custom_functionals')


def test_15():  # noqa: D103
    pytest.importorskip('matplotlib', reason='matplotlib not installed, skip tests')
    execute_example('15_custom_functionals_spin')


@pytest.mark.slow()
def test_16():  # noqa: D103
    pytest.importorskip('pyscf', reason='pyscf not installed, skip tests')
    execute_example('16_fod_optimization')
    clean_example(['CH4_fods.xyz'])


@pytest.mark.slow()
def test_17():  # noqa: D103
    execute_example('17_reduced_density_gradient')


def test_18():  # noqa: D103
    execute_example('18_k_points')


def test_20():  # noqa: D103
    execute_example('20_smeared_occupations')


if __name__ == '__main__':
    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
