# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Modified Chachiyo LDA correlation.

Reference: Comput. Theor. Chem. 1172, 112669.
"""

from .lda_c_chachiyo import lda_c_chachiyo, lda_c_chachiyo_spin


def lda_c_chachiyo_mod(n, **kwargs):
    """Modified Chachiyo parametrization of the correlation functional (spin-paired).

    Corresponds to the functional with the label LDA_C_CHACHIYO_MOD and ID 307 in Libxc.

    Reference: Comput. Theor. Chem. 1172, 112669.

    Args:
        n: Real-space electronic density.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        Modified Chachiyo correlation energy density and potential.
    """
    # Same as lda_c_chachiyo
    return lda_c_chachiyo(n, **kwargs)


def chachiyo_scaling_mod(zeta):
    """Weighting factor between the paramagnetic and the ferromagnetic case.

    Reference: Comput. Theor. Chem. 1172, 112669.

    Args:
        zeta: Relative spin polarization.

    Returns:
        Weighting factor and its derivative.
    """
    gzeta = ((1 + zeta) ** (2 / 3) + (1 - zeta) ** (2 / 3)) / 2
    fzeta = 2 * (1 - gzeta**3)

    dfdzeta = -2 * gzeta**2 * (1 / (1 + zeta) ** (1 / 3) - 1 / (1 - zeta) ** (1 / 3))
    return fzeta, dfdzeta


def lda_c_chachiyo_mod_spin(n, zeta, **kwargs):
    """Modified Chachiyo parametrization of the correlation functional (spin-polarized).

    Corresponds to the functional with the label LDA_C_CHACHIYO_MOD and ID 307 in Libxc.

    Reference: Comput. Theor. Chem. 1172, 112669.

    Args:
        n: Real-space electronic density.
        zeta: Relative spin polarization.

    Keyword Args:
        **kwargs: Throwaway arguments.

    Returns:
        Modified Chachiyo correlation energy density and potential.
    """
    return lda_c_chachiyo_spin(n, zeta, weight_function=chachiyo_scaling_mod, **kwargs)
