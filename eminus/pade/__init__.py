#!/usr/bin/env python3
'''GTH pseudopotential files.

Reference: Phys. Rev. B 54, 1703.
'''

if __name__ == '__main__':
    import inspect
    import os
    import shutil
    import urllib.request
    import zipfile

    psp_path = os.path.dirname(inspect.getfile(inspect.currentframe()))
    file = 'master.zip'
    # Download files
    url = f'https://github.com/cp2k/cp2k-data/archive/refs/heads/{file}'
    urllib.request.urlretrieve(url, file)
    # Unpack files
    with zipfile.ZipFile(file, 'r') as fzip:
        fzip.extractall()
    # Move files
    pade_path = f'{psp_path}/cp2k-data-master/potentials/Goedecker/cp2k/pade'
    for f in os.listdir(pade_path):
        shutil.move(f'{pade_path}/{f}', f'{psp_path}/{f}')
    # Cleanup
    os.remove(file)
    shutil.rmtree('cp2k-data-master')
