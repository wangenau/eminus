..
   SPDX-FileCopyrightText: 2021 The eminus developers
   SPDX-License-Identifier: Apache-2.0

.. _theory:

Theory
******

This theory section serves as a short, but incomplete introduction to electronic structure theory and the theoretical foundations of the code.
To get a more complete overview one can start by reading the cited articles and/or one can take a look inside a master thesis :cite:`Schulze2021`.

----

One of the main goals in electronic structure theory is to calculate the properties of atoms, molecules, or periodic systems.
A way to access these properties is by solving the underlying Schrödinger equation of the system.
While the Schrödinger equation in general is time-dependent, we will focus on the time-independent one

.. math::

   \hat{\cal H} \Psi(\{\boldsymbol r_i\}, \{\boldsymbol r_I\}) = E \Psi(\{\boldsymbol r_i\}, \{\boldsymbol r_I\}),

where :math:`\hat{\cal H}` is the Hamiltonian, :math:`\Psi` a wave function that depends on the positions of all electrons :math:`\{\boldsymbol r_i\}` and all nuclei :math:`\{\boldsymbol r_I\}`, and :math:`E` the energy of the system.
Using the Born-Oppenheimer approximation :cite:`Born1927` the motions of nuclei and electrons can be separated.

Density functional theory
=========================

When trying to solve the Schrödinger equation numerically one will face the problem that the computational complexity grows exponentially with the number of electrons.
One approach to reduce the computational complexity while solving the Schrödinger equation approximately is density functional theory (DFT).

Based on the basic theorems provided by Hohenberg and Kohn (HK) :cite:`Hohenberg1964`, the external potential for an electronic system can be fully determined (up to a constant) by the ground state particle density :math:`n_0(\boldsymbol r)`.
While theoretically important, its usability came with Kohn and Sham (KS) :cite:`Kohn1965` where they introduced an auxiliary system of non-interacting particles that has the same density as the interacting system.
The KS eigenvalue equation will be written with an effective external potential :math:`\hat{V}_\mathrm{eff}(\boldsymbol r)` that acts on these particles

.. math::

   \hat{\cal H}_{\mathrm{KS}} \Psi_i(\boldsymbol r) = \left( \hat{T} + \hat{V}_\mathrm{eff}(\boldsymbol r) \right) \Psi_i(\boldsymbol r) = \epsilon_i \Psi_i(\boldsymbol r).

The total energy for this system can be expressed as

.. math::

   E_{\mathrm{KS}}[n(\boldsymbol r)] = T_s[n(\boldsymbol r)] + E_{en}[n(\boldsymbol r)] + E_{\mathrm{H}}[n(\boldsymbol r)] + E_{nn} + E_{\mathrm{XC}}[n(\boldsymbol r)],

with :math:`T_s` as the single-particle kinetic energy, :math:`E_{en}` as the energy that adds due to the external field of the nuclei, :math:`E_{\mathrm{H}}` as the Hartree, or Coulomb energy, and :math:`E_{nn}` as the interaction energy of the nuclei.
The difference between the HK and KS formulations shall be compensated by the exchange-correlation (XC) functional :math:`E_{\mathrm{XC}}`.

Exchange and correlation
========================

While the exact XC functional is not known, there exist a vast variety of functionals that can be used for actual calculations.
One of the earliest and simplest of them is the local (spin) density approximation (L(S)DA).
Here, the XC energy is assumed to be the same as in a homogeneous electron gas.
Climbing the Jacobs ladder of XC functionals :cite:`Perdew2001`, more advanced approaches can be taken incorporating more information, e.g, by including the gradient of the density leading to the generalized gradient approximation (GGA), or by including the kinetic energy density (the gradient of orbitals) described as *meta*-GGAs.

Self-interaction correction
===========================

While DFT can lead to reasonable results for a variety of systems, the underlying density functional approximations will lead to errors.
One of these errors is the so-called self-interaction error (SIE), which comes from an artificial interaction of electrons with themselves.
Self-interaction correction (SIC) is a method that introduces an energy expression that corrects the total energy of a given system.
For a one-electron system, the exact total energy can be described using only the kinetic energy :math:`T_s` and the external potential energies :math:`E_{en}` and :math:`E_{nn}`.
Therefore, the Hartree energy and exchange-correlation energy should cancel themselves out.
Violating this condition will result in the SIE energy

.. math::
   E_{\mathrm{SI}}[n_1^{\sigma}] = E_{\mathrm{H}}[n_1^{\sigma}] + E_{\mathrm{XC}}[n_1^{\sigma}, 0],

where :math:`n_1^{\sigma}` is the one-particle density for one electron.
In the formulation of Perdew and Zunger (PZ) :cite:`Perdew1981` this error will be removed by subtracting the SIE for all :math:`N^\sigma` orbitals with spin :math:`\sigma` from the total energy

.. math::

   E_{\mathrm{PZ}}[n^{\alpha}, n^{\beta}] = E_{\mathrm{KS}}[n^{\alpha}, n^{\beta}] - \sum_\sigma \sum_i^{N^\sigma} E_{\mathrm{SI}}[n_i^{\sigma}].

Algebraic formulation of DFT
============================
Arias et al. introduced a completely new algebraic formulation of DFT called DFT++ :cite:`IsmailBeigi2000` that uses an operator-based pragma formulation, similar to operators used in quantum mechanics.
To handle all wave functions :math:`\psi_i` together, the matrix :math:`\boldsymbol W` is used to store their expansion coefficients.
Using the overlap operator :math:`\hat O` one can build the coefficients of orthonormal constrained wave functions

.. math::

   \boldsymbol Y = \boldsymbol W \left( \boldsymbol W^{\dagger}\hat O \boldsymbol W \right)^{-1/2}.

Applying the forward transformation operator :math:`\hat I` on :math:`\boldsymbol W` results in a matrix of function values of :math:`\psi_i` discretized on a real-space grid.
The operator :math:`\hat J` would reverse said transformation.
The density can then be built using

.. math::

   \boldsymbol n = (\hat I \boldsymbol W)\boldsymbol F(\hat I \boldsymbol W)^{\dagger},

where :math:`\boldsymbol F` is a diagonal matrix of fillings/occupations numbers per state :math:`\psi_i`.
Using the Laplace operator :math:`\hat L` the LDA energy functional can be expressed as

.. math::

   E_{\mathrm{LDA}} = -\frac{1}{2}\,\mathrm{Tr}(\boldsymbol F \boldsymbol W^\dagger \hat L \boldsymbol W) + (\hat J \boldsymbol n)^{\dagger} \left[ V_{\mathrm{ion}} + \hat O \hat J \epsilon_{\mathrm{XC}}(\boldsymbol n) - \frac{1}{2} \hat O \left( 4\pi\hat L^{-1}\hat O \hat J \boldsymbol n \right) \right],

with :math:`V_{\mathrm{ion}}` for the potential induced by the nuclei and :math:`\epsilon_{\mathrm{XC}}` as the XC energy density

.. math::

   E^{\mathrm{LDA}}_{\mathrm{XC}}[n(\boldsymbol r)] = \int\mathrm{d}\boldsymbol r\, n(\boldsymbol r) \epsilon_{\mathrm{XC}}[n(\boldsymbol r)].

----

.. bibliography::
