..
   SPDX-FileCopyrightText: 2021 The eminus developers
   SPDX-License-Identifier: Apache-2.0

.. _overview:

Overview
********

Features
========

..
   Hex color codes:
   729ece   aec7e8
   ff9e4a   ffbb78
   67bf5c   98df8a
   ed665d   ff9896
   ad8bc9   c5b0d5
   a8786e   c49c94
   ed97ca   f7b6c2
   a2a2a2   c7c7c7
   cdcc5d   dbdb8d
   6dccda   9edae5

.. raw:: html

   <link href="https://fonts.googleapis.com/css?family=Comfortaa" rel="stylesheet">
   <div class="colored-heading" style="background:#729ece">Main</div>

* Plane wave DFT code
* Minimal dependencies
* Modular extension system
* Well-documented Python3 code following pep8
* Well-tested with automated tests and coverage above 95%
* Many supported platforms: Linux, macOS, Windows, Nix, Docker, ...

.. raw:: html

   <div class="colored-heading" style="background:#ff9e4a">Functionals</div>
   <ul class="colored-ul">
      <li style="background:#ffbb78">LDA</li>
      <li style="background:#ffbb78">GGA</li>
      <li style="background:#ffbb78">meta-GGA</li>
      <li style="background:#ffbb78">Libxc</li>
   </ul>

   <div class="colored-heading" style="background:#67bf5c">Potentials</div>
   <ul class="colored-ul">
      <li style="background:#98df8a">All-electron Coulomb</li>
      <li style="background:#98df8a">GTH</li>
   </ul>

   <div class="colored-heading" style="background:#ed665d">SCF</div>
   <ul class="colored-ul">
      <li style="background:#ff9896">Steepest descent</li>
      <li style="background:#ff9896">Line minimization</li>
      <li style="background:#ff9896">Conjugate gradient</li>
      <li style="background:#ff9896">Customizable schemes</li>
   </ul>

   <div class="colored-heading" style="background:#ad8bc9">Orbitals</div>
   <ul class="colored-ul">
      <li style="background:#c5b0d5">Kohn-Sham</li>
      <li style="background:#c5b0d5">Fermi</li>
      <li style="background:#c5b0d5">Fermi-Löwdin</li>
      <li style="background:#c5b0d5">Wannier</li>
      <li style="background:#c5b0d5">SCDM</li>
   </ul>

   <div class="colored-heading" style="background:#a8786e">Properties</div>
   <ul class="colored-ul">
      <li style="background:#c49c94">Energy contributions</li>
      <li style="background:#c49c94">Dipole moments</li>
      <li style="background:#c49c94">Ionization potentials</li>
      <li style="background:#c49c94">Orbital spreads</li>
      <li style="background:#c49c94">Centers of mass</li>
      <li style="background:#c49c94">Field properties</li>
   </ul>

   <div class="colored-heading" style="background:#ed97ca">SIC</div>
   <ul class="colored-ul">
      <li style="background:#f7b6c2">Fixed density SIC</li>
      <li style="background:#f7b6c2">FLO-SIC</li>
      <li style="background:#f7b6c2">PyCOM</li>
   </ul>

   <div class="colored-heading" style="background:#a2a2a2">Visualization</div>
   <ul class="colored-ul">
      <li style="background:#c7c7c7">Molecules</li>
      <li style="background:#c7c7c7">Orbitals</li>
      <li style="background:#c7c7c7">Densities</li>
      <li style="background:#c7c7c7">Grids</li>
      <li style="background:#c7c7c7">Files</li>
      <li style="background:#c7c7c7">Contours</li>
      <li style="background:#c7c7c7">Brillouin zones</li>
      <li style="background:#c7c7c7">Band structures</li>
   </ul>

   <div class="colored-heading" style="background:#cdcc5d">Files</div>
   <ul class="colored-ul">
      <li style="background:#dbdb8d">XYZ</li>
      <li style="background:#dbdb8d">TRAJ</li>
      <li style="background:#dbdb8d">CUBE</li>
      <li style="background:#dbdb8d">POSCAR</li>
      <li style="background:#dbdb8d">PDB</li>
      <li style="background:#dbdb8d">JSON</li>
   </ul>

   <div class="colored-heading" style="background:#6dccda">Domains</div>
   <ul class="colored-ul">
      <li style="background:#9edae5">Spherical</li>
      <li style="background:#9edae5">Cuboidal</li>
      <li style="background:#9edae5">Isovalue</li>
   </ul>

Workflow
========

The following code samples show the workflow of how a bandstructure of a silicon crystal can be created.

Create the unit cell and display it.

.. code-block:: python

   from eminus import Cell, SCF
   from eminus.extras import plot_bandstructure

   cell = Cell('Si', 'diamond', ecut=10, a=10.2631, bands=8)
   cell.view()

.. image:: /_static/overview/cell.png
   :align: center
   :width: 65%

Run the DFT calculation.

.. code-block:: python

   scf = SCF(cell)
   scf.run()

Define the band path and display the Brillouin zone.

.. code-block:: python

   scf.kpts.path = 'LGXU,KG'
   scf.kpts.Nk = 25
   scf.kpts.build().view()

.. image:: /_static/overview/bz.png
   :align: center
   :width: 65%

Calculate the eigenenergies and plot the band structure.

.. code-block:: python

   scf.converge_bands()
   plot_bandstructure(scf)

.. image:: /_static/overview/band_structure.png
   :align: center
   :width: 65%

Find this example with more comments in the :doc:`examples section <_examples/19_band_structures>`.
