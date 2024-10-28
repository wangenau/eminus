<!--
SPDX-FileCopyrightText: 2021 The eminus developers
SPDX-License-Identifier: Apache-2.0
-->
![eminus logo](https://gitlab.com/wangenau/eminus/-/raw/main/docs/_static/logo/eminus_logo.png)

# eminus
[![Version](https://img.shields.io/pypi/v/eminus?color=1a962b&logo=python&logoColor=a0dba2&label=Version)](https://pypi.org/project/eminus)
[![Python](https://img.shields.io/pypi/pyversions/eminus?color=1a962b&logo=python&logoColor=a0dba2&label=Python)](https://wangenau.gitlab.io/eminus/installation.html)
[![License](https://img.shields.io/badge/license-Apache2.0-1a962b?logo=python&logoColor=a0dba2&label=License)](https://wangenau.gitlab.io/eminus/license.html)
[![Coverage](https://img.shields.io/gitlab/pipeline-coverage/wangenau%2Feminus?branch=main&color=1a962b&logo=gitlab&logoColor=a0dba2&label=Coverage)](https://wangenau.gitlab.io/eminus/htmlcov)
[![Chat](https://img.shields.io/badge/Chat-Discord-1a962b?logo=discord&logoColor=a0dba2)](https://discord.gg/k2XwdMtVec)

eminus is a pythonic electronic structure theory code.
It implements plane wave density functional theory (DFT) with self-interaction correction (SIC) functionalities.
The goal is to create a simple code that is easy to read and easy to extend while using minimal dependencies.
It is built upon the [DFT++](https://arxiv.org/abs/cond-mat/9909130) pragmas proposed by Tomas Arias et al. that aim to let programming languages and theory coincide.
This can be shown by, e.g., solving the Poisson equation. In the operator notation of DFT++ the equation reads

$$
\phi(\boldsymbol r) = -4\pi\mathcal L^{-1}\mathcal O\mathcal J n(\boldsymbol r).
$$

The corresponding Python code (implying that the operators have been implemented properly) reads

```python
def get_phi(atoms, n):
    return -4 * np.pi * atoms.Linv(atoms.O(atoms.J(n)))
```

## Installation

The [package](https://pypi.org/project/eminus) and all necessary dependencies can be installed with

```terminal
pip install eminus
```

More information about installing eminus can be found [here](https://wangenau.gitlab.io/eminus/installation.html).

## Documentation

To learn more about the features, usage, or implementation of eminus, take a look inside the [documentation](https://wangenau.gitlab.io/eminus).

## Citation

A supplementary paper is available on [arXiv](https://arxiv.org/abs/2410.19438). The following BibTeX key can be used

```terminal
@Misc{Schulze2021,
  author    = {Schulze, Wanja T. and Schwalbe, Sebastian and Trepte, Kai and Gr{\"a}fe, Stefanie},
  title     = {eminus -- Pythonic electronic structure theory},
  year      = {2024},
  doi       = {10.48550/arXiv.2410.19438},
  publisher = {arXiv},
}
```

To cite a specific version one can select and cite it with [Zenodo](https://doi.org/10.5281/zenodo.5720635).

## License

This project is licensed under the Apache 2.0 License. See the [license page](https://wangenau.gitlab.io/eminus/license.html) for more details.
