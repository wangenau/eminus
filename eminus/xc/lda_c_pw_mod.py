#!/usr/bin/env python3
'''Modified Perdew-Wang LDA correlation.

Reference: Phys. Rev. B 45, 13244.
'''
from .lda_c_pw import lda_c_pw, lda_c_pw_spin


def lda_c_pw_mod(n, **kwargs):
    '''Modified Perdew-Wang parametrization of the correlation functional (spin-paired).

    Corresponds to the functional with the label LDA_C_PW_MOD and ID 13 in Libxc.
    Reference: Phys. Rev. B 45, 13244.

    Args:
        n (ndarray): Real-space electronic density.

    Returns:
        tuple[ndarray, ndarray]: PW correlation energy density and potential.
    '''
    return lda_c_pw(n, A=0.0310907, **kwargs)


def lda_c_pw_mod_spin(n, zeta, **kwargs):
    '''Modified Perdew-Wang parametrization of the correlation functional (spin-polarized).

    Corresponds to the functional with the label LDA_C_PW_MOD and ID 13 in Libxc.
    Reference: Phys. Rev. B 45, 13244.

    Args:
        n (ndarray): Real-space electronic density.
        zeta (ndarray): Relative spin polarization.

    Returns:
        tuple[ndarray, ndarray]: PW correlation energy density and potential.
    '''
    return lda_c_pw_spin(n, zeta, A=(0.0310907, 0.01554535, 0.0168869),
                         fzeta0=1.709920934161365617563962776245, **kwargs)
