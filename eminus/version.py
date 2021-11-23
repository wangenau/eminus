#!/usr/bin/env python3
'''
Package version number and version info function.
'''
from sys import version

__version__ = '1.0.1'


def info():
    '''Print version numbers and availability of used packages.'''
    print('--- Version infos ---')
    print(f'python       : {version.split()[0]}')
    print(f'eminus       : {__version__}')
    for pkg in ('numpy', 'scipy', 'vispy', 'nglview', 'jupyter_rfb', 'pyflosic_dev'):
        try:
            exec(f'import {pkg}')
            print(f'{pkg.ljust(13)}: {eval(pkg).__version__}')
        except ModuleNotFoundError:
            print(f'{pkg.ljust(13)}: not installed')
    return
