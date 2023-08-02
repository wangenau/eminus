#!/usr/bin/env python3
"""Sphinx documentation builder configuration file.

For a full list of options see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
import datetime
from typing import Any, Callable, Dict, List, Union

from sphinx.ext.autosummary.generate import AutosummaryRenderer

import eminus

project: str = 'eminus'
author: str = 'Wanja Timm Schulze'
copyright: str = f'2021-{datetime.datetime.now().year}, Wanja Timm Schulze'  # noqa: A001
version: str = eminus.__version__
release: str = version

extensions: List[str] = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode'
]
templates_path: List[str] = ['_templates']
pygments_style: str = 'friendly'
pygments_dark_style: str = 'native'

language: str = 'en'

html_theme: str = 'furo'
html_favicon: str = '_static/logo/eminus_favicon.png'
html_theme_options: Dict[str, Union[str, Dict[str, str], List[Dict[str, str]]]] = {
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
            'url': 'https://gitlab.com/wangenau/eminus',
            'html': '<svg stroke="currentColor" fill="currentColor" viewBox="0 0 16 16"><g transform="matrix(.083313 0 0 .083313 -7.8292 -8.1245)"><path d="m282.83 170.73-0.27-0.69-26.14-68.22a6.81 6.81 0 0 0-2.69-3.24 7 7 0 0 0-8 0.43 7 7 0 0 0-2.32 3.52l-17.65 54h-71.47l-17.65-54a6.86 6.86 0 0 0-2.32-3.53 7 7 0 0 0-8-0.43 6.87 6.87 0 0 0-2.69 3.24l-26.19 68.19-0.26 0.69a48.54 48.54 0 0 0 16.1 56.1l0.09 0.07 0.24 0.17 39.82 29.82 19.7 14.91 12 9.06a8.07 8.07 0 0 0 9.76 0l12-9.06 19.7-14.91 40.06-30 0.1-0.08a48.56 48.56 0 0 0 16.08-56.04z"></path></g></svg>'  # noqa: E501
        }
    ]
}
html_static_path: List[str] = ['_static']
html_css_files: List[str] = ['css/custom.css']
html_show_sphinx: bool = False

autodoc_member_order: str = 'groupwise'
autodoc_preserve_defaults: bool = True
napoleon_use_rtype: bool = False


def dunder_skip(app: Any, what: Any, name: str, obj: Any, would_skip: bool, options: Any) -> bool:
    """Exclude all dunder methods."""
    if name.startswith('_'):
        return True
    return would_skip


def setup(app: Any) -> None:
    """Customize build process."""
    import pathlib
    import sys
    sys.path.append(str(pathlib.Path(__file__).parent.resolve()))
    import examples_builder

    app.connect('builder-inited', examples_builder.generate)
    app.connect('autodoc-skip-member', dunder_skip)
    app.connect('build-finished', examples_builder.clean)


def remove_package_name(fullname: str) -> str:
    """Remove the package name from a given fullname."""
    parts = fullname.split('.')
    if len(parts) == 1:
        return parts[0]
    return '.'.join(parts[1:])


# Back up of the original init function
old_init: Callable[[Any, Any], None] = AutosummaryRenderer.__init__


def patched_init(self: Any, app: Any) -> None:
    """Patch the AutosummaryRenderer init function to add the remove_package_name function."""
    old_init(self, app)
    self.env.filters['remove_package_name'] = remove_package_name


# Monkey patch the init function
AutosummaryRenderer.__init__ = patched_init
