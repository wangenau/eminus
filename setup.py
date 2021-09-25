#!/usr/bin/env python3
from setuptools import find_packages, setup

with open('plainedft/version.py', 'r') as fh:
    version = {}
    exec(fh.read(), version)

with open('README.md', 'r') as fh:
    long_description = fh.read()

setup(
    name='plainedft',
    version=version['__version__'],
    description='Simple plane wave density funtional theory code.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://gitlab.com/wangenau/plainedft',
    author='Wanja Schulze',
    author_email='wangenau@protonmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=['numpy', 'scipy'],
    extras_require={
        'addons': ['nglview', 'pyscf']  # , 'pyflosic_dev']
    },
    python_requires='>=3.6',
    include_package_data=True,
    zip_safe=False
)
