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
    description='A plane wave density funtional theory code.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Wanja Timm Schulze',
    author_email='wangenau@protonmail.com',
    url='https://nextdft.gitlab.io/eminus',
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
    install_requires=['numpy', 'scipy'],
    keywords=['NextDFT'],
    package_data={'eminus.pade_gth': ['*.gth']},
    zip_safe=False,
    extras_require={
        'addons': ['jupyter_rfb', 'nglview', 'notebook', 'pyflosic2', 'pylibxc2', 'vispy']
    },
    python_requires='>=3.6',
    project_urls={
        'Documentation': 'https://nextdft.gitlab.io/eminus',
        'Source': 'https://gitlab.com/nextdft/eminus',
        'Tracker': 'https://gitlab.com/nextdft/eminus/-/issues'
    }
)
