#!/usr/bin/env python3
"""Linear algebra calculation utilities."""
import functools
import re

import numpy as np
from scipy.linalg import norm

import eminus

from . import config
from .units import rad2deg


def dotprod(a, b):
    """Efficiently calculate the expression a * b.

    Add an extra check to make sure the result is never zero since this function is used as a
    denominator in minimizers.

    Args:
        a (ndarray): Array of vectors.
        b (ndarray): Array of vectors.

    Returns:
        float: The expressions result
    """
    eps = 1e-15  # 2.22e-16 is the range of float64 machine precision
    # The dot product of complex vectors looks like the expression below, but this is slow
    # res = np.real(np.trace(a.conj().T @ b))
    # We can calculate the trace faster by taking the sum of the Hadamard product
    res = np.sum(a.conj() * b)
    if abs(res) < eps:
        return eps
    return np.real(res)


def Ylm_real(l, m, G):  # noqa: C901
    """Calculate real spherical harmonics from cartesian coordinates.

    Reference: https://scipython.com/blog/visualizing-the-real-forms-of-the-spherical-harmonics

    Args:
        l (int): Angular momentum number.
        m (int): Magnetic quantum number.
        G (ndarray): Reciprocal lattice vector or array of lattice vectors.

    Returns:
        ndarray: Real spherical harmonics.
    """
    eps = 1e-9
    # Account for single vectors
    G = np.atleast_2d(G)

    # No need to calculate more for l=0
    if l == 0:
        return 0.5 * np.sqrt(1 / np.pi) * np.ones(len(G))

    # cos(theta)=Gz/|G|
    Gm = norm(G, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        cos_theta = G[:, 2] / Gm
    # Account for small magnitudes, if norm(G) < eps: cos_theta=0
    cos_theta[Gm < eps] = 0

    # Vectorized version of sin(theta)=sqrt(max(0, 1-cos_theta^2))
    sin_theta = np.sqrt(np.amax((np.zeros_like(cos_theta), 1 - cos_theta**2), axis=0))

    # phi=arctan(Gy/Gx)
    phi = np.arctan2(G[:, 1], G[:, 0])
    # If Gx=0: phi=pi/2*sign(Gy)
    phi_idx = np.abs(G[:, 0]) < eps
    phi[phi_idx] = np.pi / 2 * np.sign(G[phi_idx, 1])

    if l == 1:
        if m == -1:  # py
            return 0.5 * np.sqrt(3 / np.pi) * sin_theta * np.sin(phi)
        if m == 0:   # pz
            return 0.5 * np.sqrt(3 / np.pi) * cos_theta
        if m == 1:   # px
            return 0.5 * np.sqrt(3 / np.pi) * sin_theta * np.cos(phi)
    elif l == 2:
        if m == -2:  # dxy
            return np.sqrt(15 / 16 / np.pi) * sin_theta**2 * np.sin(2 * phi)
        if m == -1:  # dyz
            return np.sqrt(15 / 4 / np.pi) * cos_theta * sin_theta * np.sin(phi)
        if m == 0:   # dz2
            return 0.25 * np.sqrt(5 / np.pi) * (3 * cos_theta**2 - 1)
        if m == 1:   # dxz
            return np.sqrt(15 / 4 / np.pi) * cos_theta * sin_theta * np.cos(phi)
        if m == 2:   # dx2-y2
            return np.sqrt(15 / 16 / np.pi) * sin_theta**2 * np.cos(2 * phi)
    elif l == 3:
        if m == -3:
            return 0.25 * np.sqrt(35 / 2 / np.pi) * sin_theta**3 * np.sin(3 * phi)
        if m == -2:
            return 0.25 * np.sqrt(105 / np.pi) * sin_theta**2 * cos_theta * np.sin(2 * phi)
        if m == -1:
            return 0.25 * np.sqrt(21 / 2 / np.pi) * sin_theta * (5 * cos_theta**2 - 1) * np.sin(phi)
        if m == 0:
            return 0.25 * np.sqrt(7 / np.pi) * (5 * cos_theta**3 - 3 * cos_theta)
        if m == 1:
            return 0.25 * np.sqrt(21 / 2 / np.pi) * sin_theta * (5 * cos_theta**2 - 1) * np.cos(phi)
        if m == 2:
            return 0.25 * np.sqrt(105 / np.pi) * sin_theta**2 * cos_theta * np.cos(2 * phi)
        if m == 3:
            return 0.25 * np.sqrt(35 / 2 / np.pi) * sin_theta**3 * np.cos(3 * phi)

    raise ValueError(f'No definition found for Ylm({l}, {m}).')


def handle_spin_gracefully(func, *args, **kwargs):
    """Handle spin calculating the function for each channel separately.

    This can only be applied if the only spin-dependent indexing is the wave function W.

    Implementing the explicit handling of spin adds an extra layer of complexity where one has to
    loop over the spin states in many places. We can hide this complexity using this decorator while
    still supporting many use cases, e.g., the operators previously act on arrays containing wave
    functions of all states and of one state only. This decorator maintains this functionality and
    adds the option to act on arrays containing wave functions of all spins and all states as well.

    Args:
        func (Callable): Function that acts on spin-states.
        args: Pass-through arguments.
        kwargs: Pass-through keyword arguments.

    Returns:
        Callable: Decorator.
    """
    @functools.wraps(func)
    def decorator(obj, W, *args, **kwargs):
        if W.ndim == 3:
            return np.asarray([func(obj, Wspin, *args, **kwargs) for Wspin in W])
        return func(obj, W, *args, **kwargs)
    return decorator


def handle_k_gracefully(func, *args, **kwargs):
    """Handle k-points calculating the function for each channel separately.

    This uses the same principle as described in :func:`~eminus.utils.handle_spin_gracefully`.

    Args:
        func (Callable): Function that acts on k-point.
        args: Pass-through arguments.
        kwargs: Pass-through keyword arguments.

    Returns:
        Callable: Decorator.
    """
    @functools.wraps(func)
    def decorator(obj, W, *args, **kwargs):
        if isinstance(W, list) or (isinstance(W, np.ndarray) and W.ndim == 4):
            return [func(obj, Wk, *args, **kwargs) for Wk in W]
        return func(obj, W, *args, **kwargs)
    return decorator


def handle_k_indexable(func, *args, **kwargs):
    """Handle k-points calculating the function for each channel with an index.

    Similar to :func:`~eminus.utils.handle_k_gracefully`, but with using an explicit index in the
    signature.

    Args:
        func (Callable): Function that acts on k-point.
        args: Pass-through arguments.
        kwargs: Pass-through keyword arguments.

    Returns:
        Callable: Decorator.
    """
    @functools.wraps(func)
    def decorator(obj, W, *args, **kwargs):
        if isinstance(W, list) or (isinstance(W, np.ndarray) and W.ndim == 4):
            return [func(obj, Wk, ik, *args, **kwargs) for ik, Wk in enumerate(W)]
        return func(obj, W, *args, **kwargs)
    return decorator


def handle_k_reducable(func, *args, **kwargs):
    """Handle k-points calculating the function for each channel and reducing it.

    Similar to :func:`~eminus.utils.handle_k_gracefully`, but with using an explicit index in the
    signature and summing up all results in the end.

    Args:
        func (Callable): Function that acts on k-point.
        args: Pass-through arguments.
        kwargs: Pass-through keyword arguments.

    Returns:
        Callable: Decorator.
    """
    @functools.wraps(func)
    def decorator(obj, W, *args, **kwargs):
        if isinstance(W, list) or (isinstance(W, np.ndarray) and W.ndim == 4):
            # The Python sum allows summing single values and NumPy arrays elementwise
            return sum([func(obj, Wk, ik, *args, **kwargs) for ik, Wk in enumerate(W)])
        return func(obj, W, *args, **kwargs)
    return decorator


def skip_k(func, *args, **kwargs):
    """Handle calculations that do not support k-points.

    Args:
        func (Callable): Function that acts on k-point.
        args: Pass-through arguments.
        kwargs: Pass-through keyword arguments.

    Returns:
        Callable: Decorator.
    """
    @functools.wraps(func)
    def decorator(obj, W, *args, **kwargs):
        if isinstance(W, list) or (isinstance(W, np.ndarray) and W.ndim == 4):
            obj._atoms.kpts._assert_gamma_only()
            ret = func(obj, W[0], *args, **kwargs)
            if isinstance(ret, np.ndarray) and ret.ndim == 3:
                return [ret]
            return ret
        return func(obj, W, *args, **kwargs)
    return decorator


def handle_torch(func, *args, **kwargs):
    """Use a function optimized with Torch if available.

    Args:
        func (Callable): Function with a Torch alternative.
        args: Pass-through arguments.
        kwargs: Pass-through keyword arguments.

    Returns:
        Callable: Decorator.
    """
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        if config.use_torch:
            func_torch = getattr(eminus.extras.torch, func.__name__)
            return func_torch(*args, **kwargs)
        return func(*args, **kwargs)
    return decorator


def pseudo_uniform(size, seed=1234):
    """Lehmer random number generator, following MINSTD.

    Reference: Commun. ACM. 12, 85.

    Args:
        size (tuple): Dimension of the array to create.

    Keyword Args:
        seed (int): Seed to initialize the random number generator.

    Returns:
        ndarray: Array with (pseudo) random numbers.
    """
    W = np.zeros(size, dtype=complex)
    mult = 48271
    mod = (2**31) - 1
    x = (seed * mult + 1) % mod
    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                x = (x * mult + 1) % mod
                W[i, j, k] = x / mod
    return W


def add_maybe_none(a, b):
    """Add a and b together, when one or both can potentially be None.

    Args:
        a (ndarray | None): Array or None.
        b (ndarray | None): Array or None.

    Returns:
        ndarray: Sum of a and b.
    """
    if a is b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def molecule2list(molecule):
    """Expand a chemical formula to a list of chemical symbols.

    No charges or parentheses are allowed, only chemical symbols followed by their amount.

    Args:
        molecule (str): Simplified chemical formula (case sensitive).

    Returns:
        list: Atoms of the molecule expanded to a list.
    """
    # Insert a whitespace before every capital letter, these can appear once or none at all
    # Or insert before digits, these can appear at least once
    tmp_list = re.sub(r'([A-Z?]|\d+)', r' \1', molecule).split()
    atom_list = []
    for ia in tmp_list:
        if ia.isdigit():
            # If ia is an integer append the previous atom ia-1 times
            atom_list += [atom_list[-1]] * (int(ia) - 1)
        else:
            # If ia is a string add it to the results list
            atom_list += [ia]
    return atom_list


def atom2charge(atom, path=None):
    """Get the valence charges for a list of chemical symbols from GTH files.

    Args:
        atom (list): Atom symbols.
        path (str | None): Directory of GTH files.

    Returns:
        list: Valence charges per atom.
    """
    # Import here to prevent circular imports
    from .io import read_gth

    if path is not None:
        if path.lower() in {'pade', 'pbe'}:
            psp_path = path.lower()
        else:
            psp_path = path
    else:
        psp_path = 'pbe'
    return [read_gth(ia, psp_path=psp_path)['Zion'] for ia in atom]


def vector_angle(a, b):
    """Calculate the angle between two vectors.

    Args:
        a (ndarray): Vector.
        b (ndarray): Vector.

    Returns:
        float: Angle between a and b in Degree.
    """
    # Normalize vectors first
    a_norm = a / norm(a)
    b_norm = b / norm(b)
    angle = np.arccos(a_norm @ b_norm)
    return rad2deg(angle)


def get_lattice(lattice_vectors):
    """Generate a cell for given lattice vectors.

    Args:
        lattice_vectors (ndarray): Lattice vectors.

    Returns:
        ndarray: Lattice vertices.
    """
    # Vertices of a cube
    vertices = np.array([[0, 0, 0],
                         [0, 0, 1],
                         [0, 1, 0],
                         [0, 1, 1],
                         [1, 0, 0],
                         [1, 0, 1],
                         [1, 1, 0],
                         [1, 1, 1]])
    # Connected vertices of a cube with the above ordering
    edges = np.array([[0, 1],
                      [0, 2],
                      [0, 4],
                      [1, 3],
                      [1, 5],
                      [2, 3],
                      [2, 6],
                      [3, 7],
                      [4, 5],
                      [4, 6],
                      [5, 7],
                      [6, 7]])
    # Scale vertices with the lattice vectors
    # Select pairs of vertices to plot them later
    # The resulting return value is similar to the get_brillouin_zone function
    return [(vertices @ lattice_vectors)[e, :] for e in edges]
