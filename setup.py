#!/usr/bin/env python3
'''Setup file to make eminus installable.

For a full list of options see the documentation:
https://setuptools.pypa.io/en/latest/references/keywords.html
'''
from setuptools import find_packages, setup

with open('eminus/version.py', 'r') as fh:
    version = {}
    exec(fh.read(), version)

with open('README.md', 'r') as fh:
    long_description = fh.read()

extras = {
    'fods': [
        'pyflosic2>=2.0.0rc0'  # PyCOM FOD guesser
    ],
    'libxc': [
        'pylibxc2>=6'  # More exchange-correlation functionals
    ],
    'viewer': [
        'jupyter_rfb>=0.1.1',  # Run VisPy in notebooks
        'nglview>=2.6.5',      # Molecule and isosurface viewer
        'vispy>=0.8'           # Grid visualization
    ]
}
extras['all'] = [dep for values in extras.values() for dep in values]
extras['dev'] = [
    'flake8>=3.7',               # Style guide checker
    'flake8-docstrings>=1.4',    # Docstring style guide extension
    'flake8-import-order>=0.9',  # Import statement style guide extension
    'furo>=2022.02.14.1',        # Documentation theme
    'notebook',                  # Run and convert notebooks to HTML
    'pytest>=2.8',               # Test utilities
    'sphinx>=4'                  # Documentation builder
]


setup(
    name='eminus',
    version=version['__version__'],
    description='A plane wave density functional theory code.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Wanja Timm Schulze',
    author_email='wangenau@protonmail.com',
    url='https://wangenau.gitlab.io/eminus',
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development'
    ],
    license='APACHE2.0',
    install_requires=[
        'numpy>=1.17',
        'scipy>=1.4'
    ],
    keywords=['ESP'],
    package_data={
        'eminus.pade': ['*-q*']
    },
    zip_safe=False,
    extras_require=extras,
    python_requires='>=3.6',
    project_urls={
        'Bug Tracker': 'https://gitlab.com/wangenau/eminus/-/issues',
        'Changelog': 'https://wangenau.gitlab.io/eminus/changelog.html',
        'Source': 'https://gitlab.com/wangenau/eminus'
    }
)
