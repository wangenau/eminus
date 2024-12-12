# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any

from numpy import floating
from numpy.typing import NDArray

def get_a(theta: NDArray[floating]) -> NDArray[floating]: ...
def get_alpha(
    rs: NDArray[floating],
    theta: NDArray[floating],
    p: Parameters,
) -> NDArray[floating]: ...
def get_b(
    theta: NDArray[floating],
    b1: float,
    b2: float,
    b3: float,
    b4: float,
    b5: floating,
) -> NDArray[floating]: ...
def get_b5(
    omega: float,
    b3: float,
) -> floating: ...
def get_c(
    theta: NDArray[floating],
    c1: float,
    c2: float,
    e1: float,
    e2: float,
    e3: float,
    e4: float,
    e5: float,
) -> NDArray[floating]: ...
def get_d(
    theta: NDArray[floating],
    d1: float,
    d2: float,
    d3: float,
    d4: float,
    d5: float,
) -> NDArray[floating]: ...
def get_dadtheta(theta: NDArray[floating]) -> NDArray[floating]: ...
def get_dalphadrs(
    rs: NDArray[floating],
    theta: NDArray[floating],
    p: Parameters,
) -> NDArray[floating]: ...
def get_dalphadtheta(
    rs: NDArray[floating],
    theta: NDArray[floating],
    p: Parameters,
) -> NDArray[floating]: ...
def get_dbdtheta(
    theta: NDArray[floating],
    b1: float,
    b2: float,
    b3: float,
    b4: float,
    b5: floating,
) -> NDArray[floating]: ...
def get_dcdtheta(
    theta: NDArray[floating],
    c1: float,
    c2: float,
    e1: float,
    e2: float,
    e3: float,
    e4: float,
    e5: float,
) -> NDArray[floating]: ...
def get_dddtheta(
    theta: NDArray[floating],
    d1: float,
    d2: float,
    d3: float,
    d4: float,
    d5: float,
) -> NDArray[floating]: ...
def get_dedtheta(
    theta: NDArray[floating],
    e1: float,
    e2: float,
    e3: float,
    e4: float,
    e5: float,
) -> NDArray[floating]: ...
def get_dfxc_zeta_paramsdrs(
    rs: NDArray[floating],
    omega: float,
    a: NDArray[floating],
    b: NDArray[floating],
    c: NDArray[floating],
    d: NDArray[floating],
    e: NDArray[floating],
) -> NDArray[floating]: ...
def get_dfxc_zeta_paramsdtheta(
    rs: NDArray[floating],
    omega: float,
    a: NDArray[floating],
    b: NDArray[floating],
    c: NDArray[floating],
    d: NDArray[floating],
    e: NDArray[floating],
    dadtheta: NDArray[floating],
    dbdtheta: NDArray[floating],
    dcdtheta: NDArray[floating],
    dddtheta: NDArray[floating],
    dedtheta: NDArray[floating],
) -> NDArray[floating]: ...
def get_dfxc_zetadrs(
    rs: NDArray[floating],
    theta: NDArray[floating],
    p: Parameters,
) -> NDArray[floating]: ...
def get_dfxc_zetadtheta(
    rs: NDArray[floating],
    theta: NDArray[floating],
    p: Parameters,
) -> NDArray[floating]: ...
def get_dfxcdndn_params(
    nup: NDArray[floating],
    ndn: NDArray[floating],
    T: float,
    p0: Parameters,
    p1: Parameters,
    p2: Parameters,
) -> NDArray[floating]: ...
def get_dfxcdnup_params(
    nup: NDArray[floating],
    ndn: NDArray[floating],
    T: float,
    p0: Parameters,
    p1: Parameters,
    p2: Parameters,
) -> NDArray[floating]: ...
def get_dhdrs(
    rs: NDArray[floating],
    p: Parameters,
) -> NDArray[floating]: ...
def get_dlamdrs(
    rs: NDArray[floating],
    theta: NDArray[floating],
    p: Parameters,
) -> NDArray[floating]: ...
def get_dlamdtheta(
    rs: NDArray[floating],
    theta: NDArray[floating],
    p: Parameters,
) -> NDArray[floating]: ...
def get_dphidrs(
    rs: NDArray[floating],
    theta: NDArray[floating],
    zeta: NDArray[floating],
    p: Parameters,
) -> NDArray[floating]: ...
def get_dphidtheta(
    rs: NDArray[floating],
    theta: NDArray[floating],
    zeta: NDArray[floating],
    p: Parameters,
) -> NDArray[floating]: ...
def get_dphidzeta(
    rs: NDArray[floating],
    theta: NDArray[floating],
    zeta: NDArray[floating],
    p: Parameters,
) -> NDArray[floating]: ...
def get_drsdn(n: NDArray[floating]) -> NDArray[floating]: ...
def get_dtheta0dtheta(zeta: NDArray[floating]) -> NDArray[floating]: ...
def get_dtheta0dzeta(
    theta: NDArray[floating],
    zeta: NDArray[floating],
) -> NDArray[floating]: ...
def get_dthetadnup(
    T: float,
    nup: NDArray[floating],
) -> NDArray[floating]: ...
def get_dzetadndn(
    nup: NDArray[floating],
    ndn: NDArray[floating],
) -> NDArray[floating]: ...
def get_dzetadnup(
    nup: NDArray[floating],
    ndn: NDArray[floating],
) -> NDArray[floating]: ...
def get_e(
    theta: NDArray[floating],
    e1: float,
    e2: float,
    e3: float,
    e4: float,
    e5: float,
) -> NDArray[floating]: ...
def get_fxc(
    rs: NDArray[floating],
    theta: NDArray[floating],
    zeta: NDArray[floating],
    p0: Parameters,
    p1: Parameters,
    p2: Parameters,
) -> NDArray[floating]: ...
def get_fxc_nupndn(
    nup: NDArray[floating],
    ndn: NDArray[floating],
    T: float,
    p0: Parameters,
    p1: Parameters,
    p2: Parameters,
) -> NDArray[floating]: ...
def get_fxc_zeta(
    rs: NDArray[floating],
    theta: NDArray[floating],
    p: Parameters,
) -> NDArray[floating]: ...
def get_fxc_zeta_params(
    rs: NDArray[floating],
    omega: float,
    a: NDArray[floating],
    b: NDArray[floating],
    c: NDArray[floating],
    d: NDArray[floating],
    e: NDArray[floating],
) -> NDArray[floating]: ...
def get_gdsmfb_parameters() -> tuple[Parameters, Parameters, Parameters]: ...
def get_h(
    rs: NDArray[floating],
    p: Parameters,
) -> NDArray[floating]: ...
def get_lambda(
    rs: NDArray[floating],
    theta: NDArray[floating],
    p: Parameters,
) -> NDArray[floating]: ...
def get_n(rs: NDArray[floating]) -> NDArray[floating]: ...
def get_phi(
    rs: NDArray[floating],
    theta: NDArray[floating],
    zeta: NDArray[floating],
    p: Parameters,
) -> NDArray[floating]: ...
def get_rs_from_n(n: NDArray[floating]) -> NDArray[floating]: ...
def get_theta(
    T: float,
    n: NDArray[floating],
    zeta: NDArray[floating],
) -> NDArray[floating]: ...
def get_theta0(
    theta: NDArray[floating],
    zeta: NDArray[floating],
) -> NDArray[floating]: ...
def get_theta1(
    theta: NDArray[floating],
    zeta: NDArray[floating],
) -> NDArray[floating]: ...
def lda_xc_gdsmfb(
    n: NDArray[floating],
    **kwargs: Any,
) -> tuple[NDArray[floating], NDArray[floating], None]: ...
def lda_xc_gdsmfb_spin(
    n: NDArray[floating],
    zeta: NDArray[floating],
    **kwargs: Any,
) -> tuple[NDArray[floating], NDArray[floating], None]: ...

class Parameters:
    def __init__(self, params: dict[str, float | floating]) -> None: ...
