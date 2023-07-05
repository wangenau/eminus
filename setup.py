#!/usr/bin/env python3
"""Setup file to make eminus installable.

For a full list of options see the documentation:
https://setuptools.pypa.io/en/latest/references/keywords.html
"""
import re

from setuptools import find_packages, setup

with open('eminus/version.py', 'r') as fh:
    version = re.search(r"__version__ = '(.*?)'", fh.read()).group(1)

with open('README.md', 'r') as readme, open('CHANGELOG.md', 'r') as changelog:
    long_description = readme.read() + '\n\n' + changelog.read().split('\n----\n')[0]

extras = {
    'libxc': [
        'pyscf>=1.7.3'  # Libxc interface via PySCF
    ],
    'torch': [
        'torch>=1.8'  # Faster FFT operators using Torch
    ],
    'viewer': [
        'nglview>=2.6.5',  # Molecule and isosurface viewer
        'plotly>=4.5'      # Grid visualization
    ]
}
extras['fods'] = extras['libxc']  # PyCOM FOD guessing method uses PySCF
extras['all'] = [dep for values in extras.values() for dep in values]
extras['dev'] = [
    'coverage>=4.4',       # Generate coverage reports
    'furo>=2022.02.14.1',  # Documentation theme
    'notebook',            # Run and convert notebooks to HTML
    'pytest>=5.4',         # Test utilities
    'pytest-cov>=2.6.1',   # Collect test coverage data
    'ruff>=0.0.276',       # Style guide checker
    'sphinx>=4'            # Documentation builder
]

setup(
    name='eminus',
    version=version,
    description='A plane wave density functional theory code.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Wanja Timm Schulze',
    author_email='wangenau@protonmail.com',
    url='https://github.com/wangenau/eminus',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development'
    ],
    license='APACHE2.0',
    install_requires=[
        'numpy>=1.17',
        'scipy>=1.6'
    ],
    package_data={'eminus.psp': ['**/*-q*']},
    zip_safe=False,
    extras_require=extras,
    python_requires='>=3.7',
    project_urls={
        'Bug Tracker': 'https://gitlab.com/wangenau/eminus/-/issues',
        'Changelog': 'https://wangenau.gitlab.io/eminus/changelog.html',
        'Documentation': 'https://wangenau.gitlab.io/eminus',
        'Source code': 'https://gitlab.com/wangenau/eminus'
    }
)
