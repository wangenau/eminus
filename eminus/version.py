#!/usr/bin/env python3
'''Package version number and version info function.'''
import importlib
import os
import platform
import sys

__version__ = '2.1.2'
logo = (' ___ _____ _ ___ _ _ ___ \n'
        '| -_|     | |   | | |_ -|\n'
        '|___|_|_|_|_|_|_|___|___|\n')


def info():
    '''Print version numbers and availability of packages.'''
    dependencies = ('numpy', 'scipy')
    extras = ('plotly', 'nglview', 'notebook', 'pyscf', 'pyflosic2')
    dev = ('pylibxc', 'pytest', 'flake8', 'sphinx', 'furo')

    print(logo)
    print('--- Platform infos ---'
          f'\nPlatform    : {platform.system()} {platform.machine()}'
          f'\nRelease     : {platform.release()} {platform.version()}'
          '\n\n--- Version infos ---'
          f'\npython      : {sys.version.split()[0]}'
          f'\neminus      : {__version__}')
    for pkg in dependencies + extras + dev:
        try:
            module = importlib.import_module(pkg)
            try:
                print(f'{pkg.ljust(12)}: {module.__version__}')
            except AttributeError:
                # pylibxc does not use the standard version identifier
                print(f'{pkg.ljust(12)}: {module.version.__version__}')
        except ModuleNotFoundError:
            if pkg in dependencies:
                print(f'{pkg.ljust(12)}: Dependency not installed')
            elif pkg in extras:
                print(f'{pkg.ljust(12)}: Extra not installed')

    print('\n--- Performance infos ---')
    try:
        THREADS = int(os.environ['OMP_NUM_THREADS'])
    except KeyError:
        THREADS = 1
        print('INFO: No OMP_NUM_THREADS environment variable was found.\n'
              'To improve performance, add "export OMP_NUM_THREADS=THREADS" to your ".bashrc".\n'
              'Make sure to replace "THREADS", typically with the number of cores your CPU has.')
    finally:
        print(f'FFT operations will run on {THREADS} thread{"s" if THREADS != 1 else ""}.')
    return


if __name__ == '__main__':
    info()
