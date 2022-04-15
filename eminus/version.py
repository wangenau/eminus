#!/usr/bin/env python3
'''
Package version number and version info function.
'''
from os import environ
from sys import version

__version__ = '1.0.1'
dependencies = ('numpy', 'scipy')
addons = ('vispy', 'nglview', 'notebook', 'jupyter_rfb', 'pyflosic2')


def info():
    '''Print version numbers and availability of used packages.'''
    print('--- Version infos ---')
    print(f'python      : {version.split()[0]}')
    print(f'eminus      : {__version__}')
    for pkg in dependencies + addons:
        try:
            exec(f'import {pkg}')
            print(f'{pkg.ljust(12)}: {eval(pkg).__version__}')
        except ModuleNotFoundError:
            if pkg in dependencies:
                print(f'{pkg.ljust(12)}: Dependency not installed')
            elif pkg in addons:
                print(f'{pkg.ljust(12)}: Addon not installed')

    print('\n--- Performance infos ---')
    try:
        THREADS = int(environ['OMP_NUM_THREADS'])
    except KeyError:
        THREADS = 1
        print('INFO: No OMP_NUM_THREADS environment variable was found.\n'
              'To improve performance, add "export OMP_NUM_THREADS=THREADS" to your ".bashrc".\n'
              'Make sure to replace "THREADS", typically with the number of cores your CPU has.')
    finally:
        print(f'FFT operations will run on {THREADS} thread{"s" if THREADS!=1 else ""}.')
    return


if __name__ == '__main__':
    info()
