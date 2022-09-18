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

setup(
    name='eminus',
    version=version['__version__'],
    description='A plane wave density functional theory code.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Wanja Timm Schulze',
    author_email='wangenau@protonmail.com',
    url='https://esp42.gitlab.io/sage/eminus',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development'
    ],
    license='APACHE2.0',
    install_requires=[
        'numpy>=1.19.5',
        'scipy>=1.4'
    ],
    keywords=['ESP'],
    package_data={
        'eminus.pade': ['*-q*']
    },
    zip_safe=False,
    extras_require={
        'addons': [
            'jupyter_rfb>=0.1.1',   # Run VisPy in notebooks
            'nglview>=2.6.5',       # Molecule and isosurface viewer
            'pyflosic2>=2.0.0rc0',  # PyCOM FOD guesser
            'pylibxc2>=6',          # More exchange-correlation functionals
            'vispy>=0.8'            # Grid visualization
        ],
        'dev': [
            'flake8>=3.7',               # Style guide checker
            'flake8-docstrings>=1.4',    # Docstring style guide extension
            'flake8-import-order>=0.9',  # Import statement style guide extension
            'furo>=2022.02.14.1',        # Documentation theme
            'notebook',                  # Run and convert notebooks to HTML
            'pytest>=2.8',               # Test utilities
            'sphinx>=4'                  # Documentation builder
        ]
    },
    python_requires='>=3.6',
    project_urls={
        'Bug Tracker': 'https://gitlab.com/esp42/sage/eminus/-/issues',
        'Changelog': 'https://esp42.gitlab.io/sage/eminus/changelog.html',
        'Source': 'https://gitlab.com/esp42/sage/eminus'
    }
)
