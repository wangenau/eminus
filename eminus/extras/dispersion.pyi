# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
from ..scf import SCF

def get_Edisp(
    scf: SCF,
    version: str = ...,
    atm: bool = ...,
    xc: str | None = ...,
) -> float: ...
