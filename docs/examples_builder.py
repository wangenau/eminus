#!/usr/bin/env python3
'''Custom examples pages generation utilities.'''
import glob
import os
import re
import shutil


def generate(app):
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
                # Parse the script if one exists
                if os.path.exists(f'examples/{example}/{example}.py'):
                    fp.write(parse(f'examples/{example}/{example}.py'))
                if os.path.exists(f'examples/{example}/{example}.ipynb'):
                    fp.write('\nSee a preview of the notebook '
                             '`here <https://gitlab.com/nextdft/eminus/-/blob/master/'
                            f'examples/{example}/{example}.ipynb>`_.\n')
                # Add download buttons
                fp.write('\nDownload')
                files = glob.glob(f'examples/{example}/[!README.rst, !__pycache_]*')
                files.sort()
                for file in files:
                    fp.write(f' :download:`{file.split("/")[-1]} <../../{file}>`')
            # Add example subfile to index
            f_index.write(f'\n   {example}.rst')
    return


def parse(file):
    '''Parse Python files to display them as rst files.'''
    # Start with a horizontal line
    rst = '\n----\n'
    last_block_was_code = False

    with open(file, 'r') as fh:
        for line in fh.readlines():
            # Text blocks start with ##
            if line.startswith('##'):
                rst += f'\n{line.replace("## ", "")}'
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


def remove(app, exception):
    '''Remove generated examples after build.'''
    shutil.rmtree('docs/_examples')
    return
