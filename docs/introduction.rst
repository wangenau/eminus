.. _introduction:

Introduction
************

.. image:: https://img.shields.io/pypi/v/eminus?color=1a962b
   :target: https://pypi.org/project/eminus

.. image:: https://img.shields.io/badge/language-Python3-green
   :target: https://www.python.org

.. image:: https://img.shields.io/badge/license-APACHE2-lightgrey
   :target: https://gitlab.com/wangenau/eminus/-/blob/main/LICENSE

.. image:: https://gitlab.com/wangenau/eminus/badges/main/coverage.svg
   :target: https://gitlab.com/wangenau/eminus/-/graphs/main/charts

.. image:: https://zenodo.org/badge/431079841.svg
   :target: https://zenodo.org/badge/latestdoi/431079841

eminus is a plane wave density functional theory (DFT) code with self-interaction correction (SIC) functionalities.
The goal is to create a simple code that is easy to read and easy to extend while using minimal dependencies.
It is built upon the `DFT++ <https://arxiv.org/abs/cond-mat/9909130>`_ pragmas proposed by Tomas Arias et al. that aim to let programming languages and theory coincide.
This can be shown by, e.g., solving the Poisson equation. In the operator notation of DFT++ the equation reads

.. math::

   \boldsymbol \phi = 4\pi\hat L^{-1}\hat O\hat J \boldsymbol n.

The corresponding Python code (implying that the operators have been implemented properly) reads

.. code-block:: python

   phi = -4 * np.pi * Linv(O(J(n)))

Feature overview
================

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
   <style>
   .colored-heading{
      display:table;
      color:black;
      text-align:center;
      padding:0.5em 0;
      margin-bottom:0.5em;
      width:12.5em;
      border-radius:2.5em;
      font-family:'Comfortaa';
      font-weight:bold;
      font-size:0.85em;
   }
   .colored-ul{
      list-style:none;
      padding:0;
      margin:0 0 0.5em;
   }
   .colored-ul > li{
      display:inline-block;
      margin-right:0.125em;
      margin-bottom:0.5em;
      color:black;
      text-align:center;
      padding:0.5em 0;
      width:13em;
      border-radius:2.5em;
      font-family:'Comfortaa';
      font-size:0.85em;
   }
   </style>
   <div class="colored-heading" style="background:#729ece;">Main</div>

* Plane wave DFT code
* Minimal dependencies
* Modular extension system
* Well-documented Python3 code following pep8
* Well-tested with automated tests and coverage above 95%
* Many supported platforms: Linux, macOS, Windows, Nix, Docker, ...

.. raw:: html

   <div class="colored-heading" style="background:#ff9e4a;">Functionals</div>
   <ul class="colored-ul">
      <li style="background:#ffbb78;">LDA,VWN</li>
      <li style="background:#ffbb78;">LDA,PW</li>
      <li style="background:#ffbb78;">LDA,Chachiyo</li>
      <li style="background:#ffbb78;">GGA,PBE</li>
      <li style="background:#ffbb78;">GGA,PBEsol</li>
      <li style="background:#ffbb78;">GGA,Chachiyo</li>
      <li style="background:#ffbb78;">Libxc</li>
   </ul>

   <div class="colored-heading" style="background:#67bf5c;">Potentials</div>
   <ul class="colored-ul">
      <li style="background:#98df8a;">All-electron Coulomb</li>
      <li style="background:#98df8a;">GTH</li>
   </ul>

   <div class="colored-heading" style="background:#ed665d;">SCF</div>
   <ul class="colored-ul">
      <li style="background:#ff9896;">Steepest descent</li>
      <li style="background:#ff9896;">Line minimization</li>
      <li style="background:#ff9896;">Conjugate gradient</li>
      <li style="background:#ff9896;">Customizable schemes</li>
   </ul>

   <div class="colored-heading" style="background:#ad8bc9;">Orbitals</div>
   <ul class="colored-ul">
      <li style="background:#c5b0d5;">Kohn-Sham</li>
      <li style="background:#c5b0d5;">Fermi</li>
      <li style="background:#c5b0d5;">Fermi-Löwdin</li>
      <li style="background:#c5b0d5;">Wannier</li>
   </ul>

   <div class="colored-heading" style="background:#a8786e;">Properties</div>
   <ul class="colored-ul">
      <li style="background:#c49c94;">Energy contributions</li>
      <li style="background:#c49c94;">Dipole moments</li>
      <li style="background:#c49c94;">Ionization potentials</li>
      <li style="background:#c49c94;">Orbital spreads</li>
      <li style="background:#c49c94;">Centers of mass</li>
      <li style="background:#c49c94;">Kinetic energy densities</li>
   </ul>

   <div class="colored-heading" style="background:#ed97ca;">SIC</div>
   <ul class="colored-ul">
      <li style="background:#f7b6c2;">Fixed density SIC</li>
      <li style="background:#f7b6c2;">FLO-SIC</li>
      <li style="background:#f7b6c2;">PyCOM</li>
   </ul>

   <div class="colored-heading" style="background:#a2a2a2;">Visualization</div>
   <ul class="colored-ul">
      <li style="background:#c7c7c7;">Molecules</li>
      <li style="background:#c7c7c7;">Orbitals</li>
      <li style="background:#c7c7c7;">Densities</li>
      <li style="background:#c7c7c7;">Grids</li>
      <li style="background:#c7c7c7;">Files</li>
   </ul>

   <div class="colored-heading" style="background:#cdcc5d;">Files</div>
   <ul class="colored-ul">
      <li style="background:#dbdb8d;">XYZ</li>
      <li style="background:#dbdb8d;">CUBE</li>
      <li style="background:#dbdb8d;">PDB</li>
      <li style="background:#dbdb8d;">JSON</li>
   </ul>

   <div class="colored-heading" style="background:#6dccda;">Domains</div>
   <ul class="colored-ul">
      <li style="background:#9edae5;">Spherical</li>
      <li style="background:#9edae5;">Cuboidal</li>
      <li style="background:#9edae5;">Isovalue</li>
   </ul>

How to cite
===========

The project has been published with `Zenodo <https://doi.org/10.5281/zenodo.5720635>`_ and has an assigned DOI. The following BibTeX key can be used

.. code-block:: TeX

  @Misc{Schulze2021,
   author    = {Wanja Timm Schulze and Kai Trepte and Sebastian Schwalbe},
   title     = {eminus},
   year      = {2021},
   month     = nov,
   doi       = {10.5281/zenodo.5720635},
   publisher = {Zenodo},
  }
