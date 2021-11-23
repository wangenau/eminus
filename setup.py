#!/usr/bin/env python3
from setuptools import find_packages, setup

with open('eminus/version.py', 'r') as fh:
    version = {}
    exec(fh.read(), version)

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='eminus',
    version=version['__version__'],
    description='A plane wave density funtional theory code.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.com/wangenau/eminus',
    author='Wanja Timm Schulze',
    author_email='wangenau@protonmail.com',
    license='APACHE2.0',
    packages=find_packages(),
    install_requires=['numpy', 'scipy'],
    extras_require={
        'addons': ['jupyter_rfb', 'nglview', 'vispy']  # , 'pyflosic_dev']
    },
    python_requires='>=3.6',
    package_data={'eminus.pade_gth': ['*.gth']},
    zip_safe=False
)
