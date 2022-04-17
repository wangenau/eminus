#!/usr/bin/env python3
'''Sphinx documentation builder configuration file.

For a full list of options see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
'''
import glob
import os
import shutil

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
    if '__' in name:
        return True
    return would_skip


def examples_generate(app):
    '''Automatically generate examples page from examples folder.'''
    # Copy template file and create examples folder
    os.makedirs('docs/_examples', exist_ok=True)
    shutil.copy2('docs/_templates/custom-examples.rst', 'docs/_examples/examples.rst')
    # Get list of examples from subfolders
    examples = os.listdir('examples')
    examples = [name for name in examples if os.path.isdir(os.path.join('examples', name))]
    examples.sort()

    with open('docs/_examples/examples.rst', 'a') as f_index:
        for example in examples:
            # Create example subfile
            with open(f'docs/_examples/{example}.rst', 'w') as fp:
                fp.write(f'.. _{example}:\n')
                # Include readme
                fp.write(f'\n.. include:: ../../examples/{example}/README.rst\n')
                # Include script if one exists
                if os.path.exists(f'examples/{example}/{example}.py'):
                    fp.write(f'\n.. literalinclude:: ../../examples/{example}/{example}.py\n')
                # Add download buttons
                fp.write('\nDownload')
                files = glob.glob(f'examples/{example}/[!README.rst]*')
                files.sort()
                for file in files:
                    fp.write(f' :download:`{file.split("/")[-1]} <../../{file}>`')
            # Add example subfile to index
            f_index.write(f'\n   {example}.rst')


def examples_remove(app, exception):
    '''Remove generated examples after build.'''
    shutil.rmtree('docs/_examples')


def setup(app):
    '''Customized build process.'''
    app.connect('builder-inited', examples_generate)
    app.connect('autodoc-skip-member', dunder_skip)
    app.connect('build-finished', examples_remove)
