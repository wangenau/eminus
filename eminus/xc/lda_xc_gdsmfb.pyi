
from numpy import (
    float64,
    ndarray,
)

def get_a(theta: ndarray) -> ndarray: ...


def get_alpha(rs: ndarray, theta: ndarray, p: Parameters) -> ndarray: ...


def get_b(theta: ndarray, b1: float, b2: float, b3: float, b4: float, b5: float64) -> ndarray: ...


def get_b5(omega: float, b3: float) -> float64: ...


def get_c(
    theta: ndarray,
    c1: float,
    c2: float,
    e1: float,
    e2: float,
    e3: float,
    e4: float,
    e5: float
) -> ndarray: ...


def get_d(theta:
        ndarray,
        d1: float,
        d2: float,
        d3: float,
        d4: float,
        d5: float) -> ndarray: ...


def get_dadtheta(theta: ndarray) -> ndarray: ...


def get_dalphadrs(rs: ndarray, theta: ndarray, p: Parameters) -> ndarray: ...


def get_dalphadtheta(rs: ndarray, theta: ndarray, p: Parameters) -> ndarray: ...


def get_dbdtheta(theta: ndarray,
        b1: float,
        b2: float,
        b3: float,
        b4: float,
        b5: float64) -> ndarray: ...


def get_dcdtheta(
    theta: ndarray,
    c1: float,
    c2: float,
    e1: float,
    e2: float,
    e3: float,
    e4: float,
    e5: float
) -> ndarray: ...


def get_dddtheta(theta: ndarray,
        d1: float,
        d2: float,
        d3: float,
        d4: float,
        d5: float) -> ndarray: ...


def get_dedtheta(theta: ndarray,
        e1: float,
        e2: float,
        e3: float,
        e4: float,
        e5: float) -> ndarray: ...


def get_dfxc_zeta_paramsdrs(
    rs: ndarray,
    omega: float,
    a: ndarray,
    b: ndarray,
    c: ndarray,
    d: ndarray,
    e: ndarray
) -> ndarray: ...


def get_dfxc_zeta_paramsdtheta(
    rs: ndarray,
    omega: float,
    a: ndarray,
    b: ndarray,
    c: ndarray,
    d: ndarray,
    e: ndarray,
    dadtheta: ndarray,
    dbdtheta: ndarray,
    dcdtheta: ndarray,
    dddtheta: ndarray,
    dedtheta: ndarray
) -> ndarray: ...


def get_dfxc_zetadrs(rs: ndarray, theta: ndarray, p: Parameters) -> ndarray: ...


def get_dfxc_zetadtheta(rs: ndarray, theta: ndarray, p: Parameters) -> ndarray: ...


def get_dfxcdndn_params(
    nup: ndarray,
    ndn: ndarray,
    T: float,
    p0: Parameters,
    p1: Parameters,
    p2: Parameters
) -> ndarray: ...


def get_dfxcdnup_params(
    nup: ndarray,
    ndn: ndarray,
    T: float,
    p0: Parameters,
    p1: Parameters,
    p2: Parameters
) -> ndarray: ...


def get_dhdrs(rs: ndarray, p: Parameters) -> ndarray: ...


def get_dlamdrs(rs: ndarray, theta: ndarray, p: Parameters) -> ndarray: ...


def get_dlamdtheta(rs: ndarray, theta: ndarray, p: Parameters) -> ndarray: ...


def get_dphidrs(
    rs: ndarray,
    theta: ndarray,
    zeta: ndarray,
    p: Parameters
) -> ndarray: ...


def get_dphidtheta(
    rs: ndarray,
    theta: ndarray,
    zeta: ndarray,
    p: Parameters
) -> ndarray: ...


def get_dphidzeta(
    rs: ndarray,
    theta: ndarray,
    zeta: ndarray,
    p: Parameters
) -> ndarray: ...


def get_drsdn(n: ndarray) -> ndarray: ...


def get_dtheta0dtheta(zeta: ndarray) -> ndarray: ...


def get_dtheta0dzeta(theta: ndarray, zeta: ndarray) -> ndarray: ...


def get_dthetadnup(T: float, nup: ndarray) -> ndarray: ...


def get_dzetadndn(nup: ndarray, ndn: ndarray) -> ndarray: ...


def get_dzetadnup(nup: ndarray, ndn: ndarray) -> ndarray: ...


def get_e(theta: ndarray, e1: float, e2: float, e3: float, e4: float, e5: float) -> ndarray: ...


def get_fxc(
    rs: ndarray,
    theta: ndarray,
    zeta: ndarray,
    p0: Parameters,
    p1: Parameters,
    p2: Parameters
) -> ndarray: ...


def get_fxc_nupndn(
    nup: ndarray,
    ndn: ndarray,
    T: float,
    p0: Parameters,
    p1: Parameters,
    p2: Parameters
) -> ndarray: ...


def get_fxc_zeta(rs: ndarray, theta: ndarray, p: Parameters) -> ndarray: ...


def get_fxc_zeta_params(
    rs: ndarray,
    omega: float,
    a: ndarray,
    b: ndarray,
    c: ndarray,
    d: ndarray,
    e: ndarray
) -> ndarray: ...


def get_gdsmfb_parameters(
) -> tuple[Parameters, Parameters, Parameters]: ...


def get_h(rs: ndarray, p: Parameters) -> ndarray: ...


def get_lambda(rs: ndarray, theta: ndarray, p: Parameters) -> ndarray: ...


def get_n(rs: ndarray) -> ndarray: ...


def get_phi(
    rs: ndarray,
    theta: ndarray,
    zeta: ndarray,
    p: Parameters
) -> ndarray: ...


def get_rs_from_n(n: ndarray) -> ndarray: ...


def get_theta(T: float, n: ndarray, zeta: ndarray) -> ndarray: ...


def get_theta0(theta: ndarray, zeta: ndarray) -> ndarray: ...


def get_theta1(theta: ndarray, zeta: ndarray) -> ndarray: ...


def lda_xc_gdsmfb_spin(n: ndarray, zeta: ndarray, **kwargs) -> tuple[ndarray, ndarray, None]: ...


class Parameters:
    def __init__(self, params: dict[str, int | float | float64]) -> None: ...
