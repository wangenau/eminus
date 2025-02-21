# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Modified Perdew-Wang LDA correlation.

Reference: Phys. Rev. B 45, 13244.
"""

from .lda_c_pw import lda_c_pw, lda_c_pw_spin


def lda_c_pw_mod(n, **kwargs):
    """Modified Perdew-Wang parametrization of the correlation functional (spin-paired).

    Corresponds to the functional with the label LDA_C_PW_MOD and ID 13 in Libxc.

    Reference: Phys. Rev. B 45, 13244.

    Args:
        n: Real-space electronic density.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        Modified PW correlation energy density and potential.
    """
    return lda_c_pw(n, A=0.0310907, **kwargs)


def lda_c_pw_mod_spin(n, zeta, **kwargs):
    """Modified Perdew-Wang parametrization of the correlation functional (spin-polarized).

    Corresponds to the functional with the label LDA_C_PW_MOD and ID 13 in Libxc.

    Reference: Phys. Rev. B 45, 13244.

    Args:
        n: Real-space electronic density.
        zeta: Relative spin polarization.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        Modified PW correlation energy density and potential.
    """
    # fzeta0 = 4 / (9 * (2**(1 / 3) - 1))
    return lda_c_pw_spin(
        n,
        zeta,
        A=(0.0310907, 0.01554535, 0.0168869),
        fzeta0=1.709920934161365617563962776245,
        **kwargs,
    )
