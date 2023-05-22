.. _nomenclature:

Nomenclature
************

The source code uses various nomenclatures for variables for easier reading and understanding.
The most common variables and their meaning will be listed here. Some variables not listed here are explained in docstrings.
Since most variables are :code:`ndarrays` the respective shape will be displayed as well. If the variable is not a ndarray, the respective datatype will be shown.

Atoms variables
===============

.. list-table::
   :widths: 15 45 40
   :header-rows: 1

   * - Variable
     - Meaning
     - Type/Shape
   * - :code:`Natoms`
     - Number of atoms
     - :code:`int`
   * - :code:`R`
     - Lattice vectors
     - :code:`(3, 3)`
   * - :code:`Omega`
     - Unit cell volume
     - :code:`float`
   * - :code:`r`
     - Real space sampling points
     - :code:`(Number of sampling points, 3)`
   * - :code:`G`
     - Reciprocal space sampling points
     - :code:`(Number of G-vectors, 3)`
   * - :code:`G2`
     - Squared G-vectors
     - :code:`(Number of G-vectors)`
   * - :code:`active`
     - Indices for a selection of G-vectors
     - :code:`tuple ((Number of active G-vectors),)`
   * - :code:`G2c`
     - Selected squared G-vectors
     - :code:`(Number of active G-vectors)`
   * - :code:`Sf`
     - Structure factor per atom
     - :code:`(Natoms, Number of active G-vectors)`

The input variables for the Atoms object are explained here: :class:`~eminus.atoms.Atoms`.


Field variables
===============

.. list-table::
   :widths: 15 45 40
   :header-rows: 1

   * - Variable
     - Meaning
     - Shape
   * - :code:`n`
     - Real space electronic density
     - :code:`(Number of sampling points)`
   * - :code:`n_spin`
     - Real space spin densities
     - :code:`(Nspin, Number of sampling points)`
   * - :code:`dn_spin`
     - Real space gradients per axis of spin densities
     - :code:`(Nspin, Number of sampling points, 3)`
   * - :code:`n_single`
     - Real space single-particle density
     - :code:`(Number of sampling points)`
   * - :code:`zeta`
     - Real space relative spin polarization
     - :code:`(Number of sampling points)`
   * - :code:`W`
     - Reciprocal space unconstrained wave functions
     - :code:`(Nspin, Number of active G-vectors, Nstate)`
   * - :code:`Y`
     - Reciprocal space constrained wave functions
     - :code:`(Nspin, Number of active G-vectors, Nstate)`
   * - :code:`Yrs`
     - Real space constrained wave functions
     - :code:`(Nspin, Number of sampling points, Nstate)`
   * - :code:`psi`
     - Reciprocal space Hamiltonian eigenstates
     - :code:`(Nspin, Number of active G-vectors, Nstate)`
   * - :code:`phi`
     - Real space electrostatic Hartree field
     - :code:`(Number of sampling points)`
   * - :code:`exc`
     - Real space exchange-correlation energy density
     - :code:`(Number of sampling points)`
   * - :code:`vxc`
     - Real space exchange-correlation potential (dexc/dn)
     - :code:`(Nspin, Number of sampling points)`
   * - :code:`vsigma`
     - Real space gradient correction (n dexc/d|dn|^2)
     - :code:`(1 or 3, Number of sampling points)`
   * - :code:`Vloc`
     - Reciprocal space local pseudopotential contribution
     - :code:`(Number of active G-vectors)`
   * - :code:`Vnonloc`
     - Reciprocal space non-local pseudopotential contribution
     - :code:`(Nspin, Number of active G-vectors, Nstate)`
   * - :code:`kso`
     - Real space Kohn-Sham orbitals
     - :code:`(Nspin, Number of sampling points, Nstate)`
   * - :code:`fo`
     - Real space Fermi orbitals
     - :code:`(Nspin, Number of sampling points, Nstate)`
   * - :code:`flo`
     - Real space Fermi-Löwdin orbitals
     - :code:`(Nspin, Number of sampling points, Nstate)`
   * - :code:`wo`
     - Real space Wannier orbitals
     - :code:`(Nspin, Number of sampling points, Nstate)`


Pseudopotential variables
=========================

.. list-table::
   :widths: 15 45 40
   :header-rows: 1

   * - Variable
     - Meaning
     - Type/Shape
   * - :code:`GTH`
     - Combination of GTH parameters for all atoms species
     - :code:`dict`
   * - :code:`psp`
     - GTH parameters for one atom species
     - :code:`dict`
   * - :code:`NbetaNL`
     - Number of projector functions
     - :code:`int`
   * - :code:`betaNL`
     - Atom-centered projector functions
     - :code:`(Number of active G-vectors, NbetaNL)`
   * - :code:`prj2beta`
     - Map projector functions to atom species data
     - :code:`(3, Natoms, 4, 7)`


Miscellaneous variables
=======================

.. list-table::
   :widths: 15 45 40
   :header-rows: 1

   * - Variable
     - Meaning
     - Type/Shape
   * - :code:`F`
     - Diagonal matrix of occupation numbers
     - :code:`(Nstate, Nstate)`
   * - :code:`U`
     - Overlap of wave functions
     - :code:`(Nstate, Nstate)`
   * - :code:`fods`
     - List of FOD positions
     - :code:`list [(Number of up-FODs, 3), (Number of down-FODs, 3)]`
   * - :code:`elec_symbols`
     - List of FOD identifier atoms
     - :code:`list`
