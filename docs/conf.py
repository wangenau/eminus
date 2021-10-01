#!/usr/bin/env python3
'''
Configuration file for the Sphinx documentation builder. For a full list of options see the
documentation: https://www.sphinx-doc.org/en/master/usage/configuration.html
'''

import sphinx_rtd_theme

import plainedft


project = 'PlaineDFT'
author = 'Wanja Schulze'
copyright = '2021, Wanja Schulze'
version = plainedft.__version__
release = plainedft.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    sphinx_rtd_theme.__name__
]
templates_path = ['_templates']
pygments_style = 'sphinx'

html_theme = 'sphinx_rtd_theme'
html_show_sphinx = False

autodoc_preserve_defaults = True
