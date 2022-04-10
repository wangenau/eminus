#!/usr/bin/env python3
'''
Configuration file for the Sphinx documentation builder. For a full list of options see the
documentation: https://www.sphinx-doc.org/en/master/usage/configuration.html
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


def dunder_skip(app, what, name, obj, would_skip, options):
    '''Exclude all dunder methods.'''
    if '__' in name:
        return True
    return would_skip


def examples_generate(app):
    '''Automatically generate examples page from examples folder.'''
    # Copy template file
    shutil.copy2('docs/_templates/custom-examples.rst', 'docs/examples.rst')
    # Get list of examples from subfolders
    examples = os.listdir('examples')
    examples = [name for name in examples if os.path.isdir(os.path.join('examples', name))]
    examples.sort()
    # Append to examples.rst
    with open('docs/examples.rst', 'a') as fp:
        for example in examples:
            with open(f'examples/{example}/README.md', 'r') as readme:
                lines = readme.readlines()
                # Generate heading from first readme line
                heading = lines[0].strip().replace('# ', '')
                fp.write(f'\n\n{heading}\n{"=" * len(heading)}\n')
                # Generate info text from readme
                for line in lines[1:]:
                    fp.write(line)
            # Include script
            fp.write(f'\n.. literalinclude:: ../examples/{example}/{example}.py\n')
            # Add download buttons
            fp.write('\nDownload')
            files = glob.glob(f'examples/{example}/[!README.md]*')
            files.sort()
            for file in files:
                fp.write(f' :download:`{file.split("/")[-1]} <../{file}>`')


def examples_cleanup(app, exception):
    '''Remove examples.rst after build.'''
    os.remove('docs/examples.rst')


def setup(app):
    '''Customize build process.'''
    app.connect('builder-inited', examples_generate)
    app.connect('autodoc-skip-member', dunder_skip)
    app.connect('build-finished', examples_cleanup)
