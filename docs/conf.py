#!/usr/bin/env python3
'''Sphinx documentation builder configuration file.

For a full list of options see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
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
    'sphinx.ext.napoleon',
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
napoleon_use_rtype = False


def dunder_skip(app, what, name, obj, would_skip, options):
    '''Exclude all dunder methods.'''
    if name.startswith('_'):
        return True
    return would_skip


def setup(app):
    '''Customized build process.'''
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    import examples_builder

    app.connect('builder-inited', examples_builder.generate)
    app.connect('autodoc-skip-member', dunder_skip)
    app.connect('build-finished', examples_builder.remove)
    return
