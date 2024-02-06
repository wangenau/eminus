#!/usr/bin/env python3
"""Custom examples pages generation utilities."""
import pathlib
import re
import shutil
from typing import Any


def generate(app: Any) -> None:
    """Automatically generate examples page from examples folder."""
    # Copy template file and create examples folder
    pathlib.Path('docs/_examples').mkdir(exist_ok=True)
    shutil.copy2('docs/_templates/custom-examples.rst', 'docs/_examples/examples.rst')
    # Get list of examples from subfolders
    examples = sorted(pathlib.Path('examples').iterdir())
    examples = [name for name in examples if name.is_dir()]

    with open('docs/_examples/examples.rst', 'a', encoding='utf-8') as f_index:
        for example in examples:
            # Create example subfile
            with open(f'docs/_examples/{example.name}.rst', 'w', encoding='utf-8') as fp:
                fp.write(f'.. _{example.name}:\n')
                # Include readme
                fp.write(f'\n.. include:: ../../{example}/README.rst\n')
                # Parse the script if it exists
                if example.joinpath(f'{example.name}.py').exists():
                    fp.write(parse(f'{example}/{example.name}.py'))
                if example.joinpath(f'{example.name}.ipynb').exists():
                    fp.write('\nTo see a preview of the notebook click the button below.'
                             '\n\n.. button-link:: https://wangenau.gitlab.io/'
                             f'eminus/_static/{example.name}.html\n'
                             '   :color: primary\n   :outline:\n\n   Preview\n')
                # Add download links
                fp.write('\nDownload')
                files = sorted(example.glob('*'))
                exclude = ['.ipynb_checkpoints', '__pycache__', 'README.rst']
                for file in files:
                    if file.name not in exclude:
                        fp.write(f' :download:`{file.name} <../../{file}>`')
                        # Copy all images to the examples folder
                        if file.suffix == '.png':
                            shutil.copy2(file, 'docs/_examples/')
            # Add example subfile to index
            f_index.write(f'\n   {example.name}.rst')


def parse(script: str) -> str:
    """Parse Python scripts to display them as rst files."""
    # Start with a horizontal line
    rst = '\n----\n'
    last_block_was_code = False

    with open(script, encoding='utf-8') as fh:
        for line in fh:
            # Text blocks start with "# # "
            if line.startswith('# # '):
                rst += f'\n{line.replace("# # ", "")}'
                last_block_was_code = False
                # If there is a figure referenced add the image below the line
                if '.png' in line:
                    # Get the image name by searching for its file ending
                    words = line.split()
                    for word in words:
                        if '.png' in word:
                            image_name = word
                    # Replace backticks
                    image_name = image_name.replace('`', '')
                    rst += f'\n.. image:: {image_name}\n   :align: center\n\n'
            # Otherwise it is a code block
            else:
                if not last_block_was_code:
                    rst += '\n.. code-block:: python\n\n'
                rst += f'   {line}'
                last_block_was_code = True

    # Add the code annotation in front of the first "`"
    comp = re.compile(r'\`(.*?)\`')
    return comp.sub(r':code:`\1`', rst)


def clean(app: Any, exception: Any) -> None:
    """Remove generated examples after build."""
    shutil.rmtree('docs/_examples')
