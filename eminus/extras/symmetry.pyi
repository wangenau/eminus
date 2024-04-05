# SPDX-FileCopyrightText: 2021 Wanja Timm Schulze <wangenau@protonmail.com>
# SPDX-License-Identifier: Apache-2.0
from ..atoms import Atoms

def symmetrize(
    atoms: Atoms,
    space_group: bool = ...,
    time_reversal: bool = ...,
) -> None: ...
