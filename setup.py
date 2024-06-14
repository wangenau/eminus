# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Setup file to make eminus installable.

For a full list of options see the documentation:
https://setuptools.pypa.io/en/latest/references/keywords.html
"""

from __future__ import annotations

import pathlib
import re

from setuptools import find_packages, setup

version: str = re.search(
    r"__version__ = '(.*?)'", pathlib.Path('eminus/version.py').read_text(encoding='utf-8')
).group(1)  # type: ignore[union-attr]
long_description: str = pathlib.Path('README.md').read_text(encoding='utf-8')

extras: dict[str, list[str]] = {
    'dispersion': [
        'dftd3>=0.6'  # Interface for DFT-D3 dispersion corrections
    ],
    'libxc': [
        'pyscf>=2.1'  # Libxc interface via PySCF
    ],
    'torch': [
        'torch>=1.8'  # Faster FFT operators
    ],
    'viewer': [
        'nglview>=2.6.5',  # Molecule viewer
        'plotly>=4.5',  # Various visualizations
    ],
}
extras['fods'] = extras['libxc']  # PyCOM FOD guessing method uses PySCF
extras['symmetry'] = extras['libxc']  # k-point symmetrization uses PySCF
extras['all'] = [dep for values in extras.values() for dep in values]
extras['dev'] = [
    'coverage>=4.4',  # Generate coverage reports
    'furo>=2022.02.14.1',  # Documentation theme
    'matplotlib>=1.5',  # Plotting library for examples
    'mypy>=0.990',  # Static type checker
    'notebook',  # Run and convert notebooks to HTML
    'pytest>=5.4',  # Test utilities
    'ruff>=0.2',  # Linter
    'sphinx>=4',  # Documentation builder
    'sphinx-design>=0.2',  # More directives for Sphinx
    'sphinxcontrib-bibtex>=2',  # Use bib files for citations in Sphinx
]

setup(
    name='eminus',
    version=version,
    description='A pythonic plane wave density functional theory (DFT) code.',
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
        'Programming Language :: Python :: 3.12',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development',
    ],
    license='APACHE2.0',
    keywords=['Python', 'DFT', 'DFT++'],
    include_package_data=True,
    install_requires=[
        'numpy>=1.17',
        'scipy>=1.6',
    ],
    extras_require=extras,
    python_requires='>=3.7',
    project_urls={
        'Bug Tracker': 'https://gitlab.com/wangenau/eminus/-/issues',
        'Changelog': 'https://wangenau.gitlab.io/eminus/changelog.html',
        'Documentation': 'https://wangenau.gitlab.io/eminus',
        'Source code': 'https://gitlab.com/wangenau/eminus',
    },
)
