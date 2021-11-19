#!/usr/bin/env python3
'''
Configuration file for the Sphinx documentation builder. For a full list of options see the
documentation: https://www.sphinx-doc.org/en/master/usage/configuration.html
'''

import sphinx_rtd_theme

import eminus


project = 'eminus'
author = 'Wanja Timm Schulze'
copyright = '2021, Wanja Timm Schulze'
version = eminus.__version__
release = eminus.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    sphinx_rtd_theme.__name__
]
templates_path = ['_templates']
pygments_style = 'sphinx'

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'display_version': False,
    'logo_only': True,
    'style_nav_header_background': '#e3e3e3'
}
html_logo = 'logo/eminus_logo.png'
html_favicon = 'logo/eminus_favicon.png'
html_show_sphinx = False

autodoc_preserve_defaults = True


def skip(app, what, name, obj, would_skip, options):
    '''Exclude all dunder methods.'''
    if '__' in name:
        return True
    return would_skip


def setup(app):
    '''Use autodoc-skip-member.'''
    app.connect('autodoc-skip-member', skip)
