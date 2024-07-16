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

eminus is a pythonic plane wave density functional theory (DFT) code with self-interaction correction (SIC) functionalities.
The goal is to create a simple code that is easy to read and easy to extend while using minimal dependencies.
It is built upon the [DFT++](https://arxiv.org/abs/cond-mat/9909130) pragmas proposed by Tomas Arias et al. that aim to let programming languages and theory coincide.
This can be shown by, e.g., solving the Poisson equation. In the operator notation of DFT++ the equation reads

$$
\boldsymbol \phi = -4\pi\hat L^{-1}\hat O\hat J \boldsymbol n.
$$

The corresponding Python code (implying that the operators have been implemented properly) reads

```python
phi = -4 * np.pi * Linv(O(J(n)))
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

The project has been published with [Zenodo](https://doi.org/10.5281/zenodo.5720635) and has an assigned DOI. The following BibTeX key can be used

```terminal
  @Misc{Schulze2021,
   author    = {Wanja Timm Schulze and Kai Trepte and Sebastian Schwalbe},
   title     = {eminus},
   year      = {2021},
   month     = nov,
   doi       = {10.5281/zenodo.5720635},
   publisher = {Zenodo},
  }
```

## License

This project is licensed under the Apache 2.0 License. See the [license page](https://wangenau.gitlab.io/eminus/license.html) for more details.
