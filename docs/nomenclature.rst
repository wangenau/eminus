..
   SPDX-FileCopyrightText: 2021 The eminus developers
   SPDX-License-Identifier: Apache-2.0

.. _nomenclature:

Nomenclature
************

The source code uses various nomenclatures for variables for easier reading and understanding.
The most common variables and their meaning will be listed here. Some variables not listed here are explained in docstrings.
Since most variables are :code:`ndarrays` the respective shape will be displayed as well. If the variable is not a ndarray, the respective datatype will be shown.
If the list is indexed over k-points it will be labeled as `list over k`.

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
   * - :code:`Z`
     - Nuclei charges per atom
     - :code:`(Natoms)`
   * - :code:`Ns`
     - Number of real-space grid points
     - :code:`int`
   * - :code:`s`
     - Real-space grid points
     - :code:`(3)`
   * - :code:`r`
     - Real-space sampling points
     - :code:`(Ns, 3)`
   * - :code:`G`
     - Reciprocal space sampling points
     - :code:`(Number of G-vectors, 3)`
   * - :code:`G2`
     - Squared G-vectors
     - :code:`(Number of G-vectors)`
   * - :code:`Gk2`
     - Squared G+k-vectors
     - :code:`list over k + 1 [(Number of G-vectors)]`
   * - :code:`active`
     - Indices for a selection of G+k-vectors
     - :code:`list over k + 1 [(Number of active G-vectors)]`
   * - :code:`G2c`
     - Selected squared G-vectors
     - :code:`(Number of active G-vectors)`
   * - :code:`Gk2c`
     - Selected squared G+k-vectors
     - :code:`list over k + 1 [(Number of active G-vectors)]`
   * - :code:`Sf`
     - Structure factor per atom
     - :code:`(Natoms, Number of active G-vectors)`

| The variables of the Atoms object are explained here: :class:`~eminus.atoms.Atoms`.
| The variables of the Occupations object are documented here: :class:`~eminus.occupations.Occupations`.
| The variables of the KPoints object are documented here: :class:`~eminus.kpoints.KPoints`.


Field variables
===============

.. list-table::
   :widths: 15 45 40
   :header-rows: 1

   * - Variable
     - Meaning
     - Shape
   * - :code:`n`
     - Real-space electronic density
     - :code:`(Ns)`
   * - :code:`n_spin`
     - Real-space spin densities
     - :code:`(Nspin, Ns)`
   * - :code:`dn_spin`
     - Real-space gradients per axis of spin densities
     - :code:`(Nspin, Ns, 3)`
   * - :code:`tau`
     - Real-space spin kinetic energy densities
     - :code:`(Nspin, Ns)`
   * - :code:`n_single`
     - Real-space single-particle density
     - :code:`(Ns)`
   * - :code:`zeta`
     - Real-space relative spin polarization
     - :code:`(Ns)`
   * - :code:`W`
     - Reciprocal space unconstrained wave functions
     - :code:`list over k [(Nspin, Number of active G-vectors, Nstate)]`
   * - :code:`Y`
     - Reciprocal space constrained wave functions
     - :code:`list over k [(Nspin, Number of active G-vectors, Nstate)]`
   * - :code:`Z`
     - Reciprocal space unconstrained unoccupied wave functions
     - :code:`list over k [(Nspin, Number of active G-vectors, Nempty)]`
   * - :code:`D`
     - Reciprocal space constrained unoccupied wave functions
     - :code:`list over k [(Nspin, Number of active G-vectors, Nempty)]`
   * - :code:`Yrs`
     - Real-space constrained wave functions
     - :code:`list over k [(Nspin, Ns, Nstate)]`
   * - :code:`psi`
     - Reciprocal space Hamiltonian eigenstates
     - :code:`list over k [(Nspin, Number of active G-vectors, Nstate)]`
   * - :code:`phi`
     - Real-space electrostatic Hartree field
     - :code:`(Ns)`
   * - :code:`exc`
     - Real-space exchange-correlation energy density
     - :code:`(Ns)`
   * - :code:`vxc`
     - Real-space exchange-correlation potential (dexc/dn)
     - :code:`(Nspin, Ns)`
   * - :code:`vsigma`
     - Real-space gradient-dependent potential contribution (n dexc/d|dn|^2)
     - :code:`(1 or 3, Ns)`
   * - :code:`vtau`
     - Real-space tau-dependent potential contribution (dexc/dtau)
     - :code:`(1 or 3, Ns)`
   * - :code:`Vloc`
     - Reciprocal space local pseudopotential contribution
     - :code:`(Number of active G-vectors)`
   * - :code:`Vnonloc`
     - Reciprocal space non-local pseudopotential contribution
     - :code:`(Nspin, Number of active G-vectors, Nstate)`
   * - :code:`kso`
     - Real-space Kohn-Sham orbitals
     - :code:`list over k [(Nspin, Ns, Nstate)]`
   * - :code:`fo`
     - Real-space Fermi orbitals
     - :code:`list [(Nspin, Ns, Nstate)]`
   * - :code:`flo`
     - Real-space Fermi-Löwdin orbitals
     - :code:`list [(Nspin, Ns, Nstate)]`
   * - :code:`wo`
     - Real-space Wannier orbitals
     - :code:`list [(Nspin, Ns, Nstate)]`
   * - :code:`scdm`
     - Real-space orbitals localized via the SCDM method
     - :code:`list [(Nspin, Ns, Nstate)]`

| The variables of the SCF object are explained here: :class:`~eminus.scf.SCF`.
| The variables of the Energy object are documented here: :class:`~eminus.energies.Energy`.


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

The GTH variables are listed here: :class:`~eminus.gth.GTH`.


Miscellaneous variables
=======================

.. list-table::
   :widths: 15 45 40
   :header-rows: 1

   * - Variable
     - Meaning
     - Type/Shape
   * - :code:`Nk`
     - Number of k-points
     - :code:`int`
   * - :code:`k`
     - k-point coordinates
     - :code:`(Nk, 3)`
   * - :code:`wk`
     - k-point weights
     - :code:`(Nk)`
   * - :code:`f`
     - Occupation numbers per spin and state
     - :code:`(Nk, Nspin, Nstate)`
   * - :code:`F`
     - Diagonal matrix of occupation numbers
     - :code:`list over k [(Nspin, Nstate, Nstate)]`
   * - :code:`U`
     - Overlap of wave functions
     - :code:`(Nstate, Nstate)`
   * - :code:`epsilon`
     - Eigenenergies
     - :code:`list over k [(Nspin, Nstate)]`
   * - :code:`fods`
     - List of FOD positions
     - :code:`list [(Number of up-FODs, 3), (Number of down-FODs, 3)]`
   * - :code:`elec_symbols`
     - List of FOD identifier atoms
     - :code:`list`

The configuration variables are listed here: :class:`~eminus.config.ConfigClass`.
