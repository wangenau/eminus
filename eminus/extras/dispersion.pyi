# SPDX-FileCopyrightText: 2024 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from ..scf import SCF

def get_Edisp(
    scf: SCF,
    version: str = ...,
    atm: bool = ...,
    xc: str | None = ...,
) -> float: ...
