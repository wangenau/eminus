#!/usr/bin/env python3
'''
Package version number and version info function.
'''
import sys

__version__ = '0.0.1'


def info():
    '''Print version numbers and availability of used packages.'''
    print('--- Version infos ---')
    print(f'python       : {sys.version.split()[0]}')
    print(f'plainedft    : {__version__}')
    for addon in ('numpy', 'scipy', 'vispy', 'nglview', 'pyflosic_dev'):
        try:
            exec(f'import {addon}')
            print(f'{addon.ljust(13)}: {eval(addon).__version__}')
        except ModuleNotFoundError:
            print(f'{addon.ljust(13)}: not installed')
    return
