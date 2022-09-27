#!/usr/bin/env python3
'''Package version number and version info function.'''
import os
import sys

__version__ = '2.1.0'
dependencies = ('numpy', 'scipy')
extras = ('vispy', 'nglview', 'notebook', 'jupyter_rfb', 'pylibxc', 'pyflosic2')
dev = ('pytest', 'flake8', 'sphinx', 'furo')


def info():
    '''Print version numbers and availability of packages.'''
    print('--- Version infos ---')
    print(f'python      : {sys.version.split()[0]}')
    print(f'eminus      : {__version__}')
    for pkg in dependencies + extras + dev:
        try:
            exec(f'import {pkg}')
            try:
                print(f'{pkg.ljust(12)}: {eval(pkg).__version__}')
            except AttributeError:
                # PyLibXC does not use the standard version identifier
                print(f'{pkg.ljust(12)}: {eval(pkg).version.__version__}')
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
