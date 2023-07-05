#!/usr/bin/env python3
"""Custom examples pages generation utilities."""
import pathlib
import re
import shutil


def generate(*args):
    """Automatically generate examples page from examples folder."""
    # Copy template file and create examples folder
    pathlib.Path('docs/_examples').mkdir(exist_ok=True)
    shutil.copy2('docs/_templates/custom-examples.rst', 'docs/_examples/examples.rst')
    # Get list of examples from subfolders
    examples = sorted(pathlib.Path('examples').iterdir())
    examples = [name for name in examples if name.is_dir()]

    with open('docs/_examples/examples.rst', 'a') as f_index:
        for example in examples:
            # Create example subfile
            with open(f'docs/_examples/{example.name}.rst', 'w') as fp:
                fp.write(f'.. _{example.name}:\n')
                # Include readme
                fp.write(f'\n.. include:: ../../{example}/README.rst\n')
                # Parse the script if it exists
                if example.joinpath(f'{example.name}.py').exists():
                    fp.write(parse(f'{example}/{example.name}.py'))
                if example.joinpath(f'{example.name}.ipynb').exists():
                    fp.write('\nSee a preview of the notebook `here <https://wangenau.gitlab.io/'
                             f'eminus/_static/{example.name}.html>`_.\n')
                # Add download links
                fp.write('\nDownload')
                files = sorted(example.glob('[!README.rst, !__pycache_]*'))
                for file in files:
                    fp.write(f' :download:`{str(file).split("/")[-1]} <../../{file}>`')
            # Add example subfile to index
            f_index.write(f'\n   {example.name}.rst')
    return


def parse(script):
    """Parse Python scripts to display them as rst files."""
    # Start with a horizontal line
    rst = '\n----\n'
    last_block_was_code = False

    with open(script, 'r') as fh:
        for line in fh.readlines():
            # Text blocks start with "# # "
            if line.startswith('# # '):
                rst += f'\n{line.replace("# # ", "")}'
                last_block_was_code = False
            # Otherwise it is a code block
            else:
                if not last_block_was_code:
                    rst += '\n.. code-block:: python\n\n'
                rst += f'   {line}'
                last_block_was_code = True

    # Add the code annotation in front of the first "`"
    comp = re.compile(r'\`(.*?)\`')
    return re.sub(comp, r':code:`\1`', rst)


def clean(*args):
    """Remove generated examples after build."""
    shutil.rmtree('docs/_examples')
    return
