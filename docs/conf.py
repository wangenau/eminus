#!/usr/bin/env python3
'''Sphinx documentation builder configuration file.

For a full list of options see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
'''
import eminus

project = 'eminus'
author = 'Wanja Timm Schulze'
copyright = '2021-2022, Wanja Timm Schulze'
version = eminus.__version__
release = eminus.__version__.rpartition('.')[0]

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
        'color-brand-content': '#1a962b'
    },
    'dark_logo': 'logo/eminus_logo_dark.png',
    'dark_css_variables': {
        'color-brand-primary': '#70a973',
        'color-brand-content': '#a0dba2'
    },
    'footer_icons': [
        {
            'name': 'GitLab',
            'url': 'https://gitlab.com/nextdft/eminus',
            'html': '<svg stroke="currentColor" fill="currentColor" viewBox="0 0 16 16"><g transform="matrix(.083313 0 0 .083313 -7.8292 -8.1245)"><path d="m282.83 170.73-0.27-0.69-26.14-68.22a6.81 6.81 0 0 0-2.69-3.24 7 7 0 0 0-8 0.43 7 7 0 0 0-2.32 3.52l-17.65 54h-71.47l-17.65-54a6.86 6.86 0 0 0-2.32-3.53 7 7 0 0 0-8-0.43 6.87 6.87 0 0 0-2.69 3.24l-26.19 68.19-0.26 0.69a48.54 48.54 0 0 0 16.1 56.1l0.09 0.07 0.24 0.17 39.82 29.82 19.7 14.91 12 9.06a8.07 8.07 0 0 0 9.76 0l12-9.06 19.7-14.91 40.06-30 0.1-0.08a48.56 48.56 0 0 0 16.08-56.04z"></path></g></svg>'  # noqa: E501
        }
    ]
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
    app.connect('build-finished', examples_builder.clean)
    return
