#!/usr/bin/env python3
"""PADE GTH pseudopotential files.

Reference: Phys. Rev. B 54, 1703.
"""

if __name__ == '__main__':
    import importlib.resources
    import shutil
    import urllib.request
    import zipfile

    with importlib.resources.path('eminus.psp', 'pade') as psp_path:
        file = 'master.zip'
        # Download files
        url = f'https://github.com/cp2k/cp2k-data/archive/refs/heads/{file}'
        urllib.request.urlretrieve(url, file)  # noqa: S310
        # Unpack files
        with zipfile.ZipFile(file, 'r') as fzip:
            fzip.extractall()
        # Move files
        pade_path = psp_path.joinpath('cp2k-data-master/potentials/Goedecker/cp2k/pade')
        for f in pade_path.iterdir():
            shutil.move(str(f), psp_path.joinpath(f.name))
        # Cleanup
        psp_path.joinpath(file).unlink()
        shutil.rmtree('cp2k-data-master')
