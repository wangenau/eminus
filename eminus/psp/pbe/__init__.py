# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""PBE GTH pseudopotential files.

Reference: Theor. Chem. Acc. 114, 145.
"""

if __name__ == "__main__":
    import importlib.resources
    import shutil
    import urllib.request
    import zipfile

    try:
        psp_path = importlib.resources.files("eminus.psp.pbe")
    except AttributeError:
        with importlib.resources.path("eminus.psp", "pbe") as p:
            psp_path = p
    file = "master.zip"
    # Download files
    url = f"https://github.com/cp2k/cp2k-data/archive/refs/heads/{file}"
    urllib.request.urlretrieve(url, file)  # noqa: S310
    # Unpack files
    with zipfile.ZipFile(file, "r") as fzip:
        fzip.extractall()
    # Move files
    gth_path = psp_path.joinpath("cp2k-data-master/potentials/Goedecker/cp2k/pbe")
    for f in gth_path.iterdir():
        shutil.move(str(f), psp_path.joinpath(f.name))
    # Cleanup
    psp_path.joinpath(file).unlink()
    shutil.rmtree("cp2k-data-master")
