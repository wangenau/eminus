import runpy
from os import chdir, remove
from os.path import exists

from eminus import __path__


def execute_example(name):
    '''Test the execution of a given Python script.'''
    chdir(f'{__path__[0]}/../examples/{name}')

    try:
        runpy.run_path(f'{name}.py')
    except Exception as err:
        print(f'Test for {name} failed.')
        raise SystemExit(err) from None
    else:
        print(f'Test for {name} passed.')
    return


def clean_example(trash):
    '''Clean the example folder after running the script.'''
    for it in trash:
        if exists(it):
            remove(it)
    return


def test_01():
    execute_example('01_installation_test')


def test_02():
    execute_example('02_minimal_example')


def test_03():
    execute_example('03_atoms_objects')


def test_04():
    execute_example('04_dft_calculations')


def test_05():
    execute_example('05_input_output')
    clean_example(['CH4_density.cube', 'CH4.pkl'])


def test_06():
    execute_example('06_advanced_functionalities')
    clean_example(['Ne_1.cube', 'Ne_2.cube', 'Ne_3.cube', 'Ne_4.cube'])


def test_07():
    execute_example('07_fod_addon')
    clean_example(['CH4_FLO_1.cube', 'CH4_FLO_2.cube', 'CH4_FLO_3.cube', 'CH4_FLO_4.cube',
                   'CH4_fods.xyz'])


def test_08():
    pass


def test_09():
    execute_example('09_sic_calculations')


if __name__ == '__main__':
    test_01()
    test_02()
    test_03()
    test_04()
    test_05()
    test_06()
    test_07()
    test_08()
    test_09()
