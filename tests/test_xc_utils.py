# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""Test exchange-correlation functional utilities."""

import numpy as np
import pytest
from numpy.random import default_rng
from numpy.testing import assert_allclose

from eminus import config
from eminus.xc import get_xc, get_xc_defaults, parse_functionals, parse_xc_type


@pytest.mark.parametrize(
    ("xc", "ref"),
    [
        ("svwn", ["lda_x", "lda_c_vwn"]),
        ("lda_x", ["lda_x", "mock_xc"]),
        ("s,pw", ["lda_x", "lda_c_pw"]),
        ("s", ["lda_x", "mock_xc"]),
        ("s,", ["lda_x", "mock_xc"]),
        ("pw", ["lda_c_pw", "mock_xc"]),
        (",pw", ["mock_xc", "lda_c_pw"]),
        ("", ["mock_xc", "mock_xc"]),
        (",", ["mock_xc", "mock_xc"]),
        ("libxc:1,l:7", ["libxc:1", "l:7"]),
        ("libxc:1,", ["libxc:1", "mock_xc"]),
        (",:7", ["mock_xc", ":7"]),
        ("s,l:7", ["lda_x", "l:7"]),
        (":MGGA_X_TPSS,l:231", [":MGGA_X_TPSS", "l:231"]),
    ],
)
def test_parse_functionals(xc, ref):
    """Test the xc string parsing."""
    f_x, f_c = parse_functionals(xc)
    assert f_x == ref[0]
    assert f_c == ref[1]


@pytest.mark.parametrize(
    ("xc", "ref"),
    [
        (["lda_x", "lda_c_vwn"], "lda"),
        (["gga_x_pbe", "gga_c_pbe"], "gga"),
        (["lda_x", "gga_c_pbe"], "gga"),
        (["gga_x_pbe", "lda_c_vwn"], "gga"),
    ],
)
def test_parse_xc_type(xc, ref):
    """Test the pseudopotential parsing."""
    psp = parse_xc_type(xc)
    assert psp == ref


@pytest.mark.parametrize(
    ("xc", "ref"),
    [
        (["l:1", "l:7"], "lda"),
        (["libxc:gga_x_pbe", "l:gga_c_pbe"], "gga"),
        (["libxc:1", "gga_c_pbe"], "gga"),
        (["libxc:gga_x_pbe", "lda_c_vwn"], "gga"),
        (["gga_x_pbe", "libxc:7"], "gga"),
        ([":MGGA_X_SCAN", "libxc:MGGA_C_SCAN"], "meta-gga"),
        ([":263", "gga_x_pbe"], "meta-gga"),
        ([":263", "l:7"], "meta-gga"),
    ],
)
def test_parse_xc_type_pyscf(xc, ref):
    """Test the pseudopotential parsing using PySCF."""
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    config.use_pylibxc = False
    psp = parse_xc_type(xc)
    assert psp == ref


@pytest.mark.parametrize(
    ("xc", "ref"),
    [
        (["l:1", "l:7"], "lda"),
        (["libxc:gga_x_pbe", "l:gga_c_pbe"], "gga"),
        (["libxc:1", "gga_c_pbe"], "gga"),
        (["libxc:gga_x_pbe", "lda_c_vwn"], "gga"),
        (["gga_x_pbe", "libxc:7"], "gga"),
        ([":MGGA_X_SCAN", "libxc:MGGA_C_SCAN"], "meta-gga"),
        ([":263", "gga_x_pbe"], "meta-gga"),
        ([":263", "l:7"], "meta-gga"),
    ],
)
def test_parse_xc_type_pylibxc(xc, ref):
    """Test the pseudopotential parsing using pylibxc."""
    pytest.importorskip("pylibxc", reason="pylibxc not installed, skip tests")
    config.use_pylibxc = True
    psp = parse_xc_type(xc)
    assert psp == ref


def test_libxc_str():
    """Test that strings that start with libxc get properly parsed."""
    pytest.importorskip("pyscf", reason="pyscf not installed, skip tests")
    # Create a random mock density
    rng = default_rng()
    n_spin = np.abs(rng.standard_normal((1, 10000)))
    e_out, v_out, _, _ = get_xc("1,7", n_spin, 1)
    e_test, v_test, _, _ = get_xc("l:1,l:7", n_spin, 1)
    assert_allclose(e_out, e_test)
    assert_allclose(v_out, v_test)


def test_get_xc_defaults():
    """Test that the xc defaults are correctly parsed."""
    assert get_xc_defaults("svwn5") == {"A": 0.0310907, "b": 3.72744, "c": 12.9352, "x0": -0.10498}


if __name__ == "__main__":
    import inspect
    import pathlib

    file_path = pathlib.Path(inspect.stack()[0][1])
    pytest.main(file_path)
