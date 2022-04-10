#!/usr/bin/env python3
'''
Configuration file for the Sphinx documentation builder. For a full list of options see the
documentation: https://www.sphinx-doc.org/en/master/usage/configuration.html
'''
import eminus


project = 'eminus'
author = 'Wanja Timm Schulze'
copyright = '2021, Wanja Timm Schulze'
version = eminus.__version__
release = eminus.__version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode'
]
templates_path = ['_templates']
pygments_style = 'friendly'
pygments_dark_style = 'native'

language = 'en'

html_theme = 'furo'
html_favicon = 'logo/eminus_favicon.png'
html_theme_options = {
    'light_logo': 'logo/eminus_logo.png',
    'light_css_variables': {
        'color-brand-primary': '#006700',
        'color-brand-content': '#1a962b',
    },
    'dark_logo': 'logo/eminus_logo_dark.png',
    'dark_css_variables': {
        'color-brand-primary': '#70a973',
        'color-brand-content': '#a0dba2',
    },
}
html_static_path = ['']
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
