# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .typing import Array2D, IntArray

# Create a custom Callable type for some decorators
class HandleType(Protocol):
    def __call__(
        self,
        obj: Any,
        W: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...

def dotprod(
    a: NDArray[np.complex128],
    b: NDArray[np.complex128],
) -> float: ...
def Ylm_real(
    l: int,
    m: int,
    G: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def handle_spin_gracefully(
    func: HandleType,
    *args: Any,
    **kwargs: Any,
) -> HandleType: ...
def handle_k_gracefully(
    func: HandleType,
    *args: Any,
    **kwargs: Any,
) -> HandleType: ...
def handle_k_indexable(
    func: HandleType,
    *args: Any,
    **kwargs: Any,
) -> HandleType: ...
def handle_k_reducable(
    func: HandleType,
    *args: Any,
    **kwargs: Any,
) -> HandleType: ...
def skip_k(
    func: HandleType,
    *args: Any,
    **kwargs: Any,
) -> HandleType: ...
def handle_torch(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> Callable[..., Any]: ...
def pseudo_uniform(
    size: IntArray,
    seed: int = ...,
) -> NDArray[np.complex128]: ...
def add_maybe_none(
    a: NDArray[np.complex128] | None,
    b: NDArray[np.complex128] | None,
) -> NDArray[np.complex128] | None: ...
def molecule2list(molecule: str) -> list[str]: ...
def atom2charge(
    atom: Sequence[str],
    path: str | None = ...,
) -> list[int]: ...
def vector_angle(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]: ...
def get_lattice(lattice_vectors: Array2D) -> list[NDArray[np.float64]]: ...
