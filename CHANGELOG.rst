..
   SPDX-FileCopyrightText: 2024 The eminus developers
   SPDX-License-Identifier: Apache-2.0

Changelog
=========

dev
---
- Miscellaneous
   - Change build backend from setuptools to hatchling
      - Improve build performance
      - Support PEP 639 license identifiers
      - Support reproducible builds
   - Make the dev optional-dependency a dependency-group

v3.0.3 - Mar 18, 2025
---------------------
- New features
   - Allow setting of potential parameters
   - Add long-range Coulomb potential
   - Add a callback method for customizable SCF workflows
- Miscellaneous
   - Improve performance of CUBE writing
   - Improve Coulomb potential to use it with different atomic species
   - Fix aspect mode in cell viewer which previously distorted the cell
   - Fix sampling of non-cubic cells
   - Fix density viewer of non-cubic cells
   - Fix sign of SCF differences
   - Fix number of bands when overwriting atomic charges
   - Use the creation year of a file in copyright notices

v3.0.2 - Jan 16, 2025
---------------------
- New features
   - eminus paper release!
   - Add Jax powered FFT operators as an extra (thanks to @artemfilatov1)
- Miscellaneous
   - Improve GDSMFB thermal functional
   - Rename use_torch to backend in config
   - Drop Python 3.7 support
   - Rework type hints
   - Update citation information

v3.0.1 - Dec 18, 2024
---------------------
- New features
   - Add GDSMFB thermal functional (thanks to @theonov13)
   - Improve HDF5 extra
      - Proper dataset and group creation
      - Support compression
      - Recognize more file types
   - Weekly builds
   - Produce PyPI and Docker build attestations
- Miscellaneous
   - Miscellaneous CI improvements
   - Migrate from pip to uv in Docker and CI
   - Update Docker image to Python 3.13
      - Use Jupyter lab instead of notebook

v3.0.0 - Oct 28, 2024
---------------------
- New features
   - eminus preprint release!
   - Full type hint support!
   - Rewritten minimizer
      - Massive speedup for more k-points
      - Fixed some convergence issues
   - Add non-iterative SCDM localization
      - Use them as the initial guess for Wannier localization
      - Use Wannier orbital COMs for FLO generations if no FODs are given
   - Add magnetization functions
   - Add POSCAR read and write functions
   - Add a simple HDF5 file extra
   - Allow setting of external functional parameters (internal and in pylibxc)
- Coding style
   - Reformat the codebase using Ruff
   - Activate more linting rules
   - Add SPDX license identifiers
   - Modernize CI pipelines
   - Add CI release pipelines
   - Move tox.ini and setup.py contents to pyproject.toml
   - Merge all handle_k decorators into one
- Miscellaneous
   - Fix hexagonal grid generation
   - Fix gradient convergence check, get_ip, and Efermi in extras/viewer
   - Allow plotting densities in viewer functions for all unit cell types
   - Add an option to plot both spin channels in band structure plots
   - Add DOS calculation and plot functions
   - Add an isovalue keyword to the density viewer
   - Reduce the default surfaces from 20 to 10 in the density viewer to improve performance
   - Add pass-through keyword arguments in the Cell creation
   - Add view and write class methods to Atoms, SCF, and KPoints objects
   - Set default values for uninitialized SCF attributes to None
   - Mark the log attribute as private in Atoms and SCF classes
   - Sync GTH files (this changes values for Na-q9)
   - Small tests improvements
   - Update Docker image to Python 3.12
   - Indicate Python 3.13 support
   - Use Python 3.13 as the CI base image
   - Add an eminus Discord server
   - Add citation information
- Breaking
   - Cleanup main namespace by only including unified read and write functions
   - The rewritten minimizer will change the convergence behavior of some systems!

----

v2.7.1 - Feb 09, 2024
---------------------
- New features
   - Stabilize Fermi smearing!
- Updated docs
   - Restyle many documentation pages
   - Add a citation page
   - Add an overview page with a workflow example
   - Add a smearing example
- Miscellaneous
   - Small performance improvements
   - Temperature unit conversion functions
   - Tests for the smearing implementation
   - Update Ruff rules
   - Misc coding style updates

v2.7.0 - Jan 19, 2024
---------------------
- New features
   - Add k-points!
      - Add k-point dependent calculations
      - Add a k-points object
      - Add a band structure, k-point, and Brillouin zone viewer
      - Add minimization functions for fixed Hamiltonians
   - Add a symmetry extra to symmetrize k-points
- Updated docs
   - Add k-point examples
   - Increase coverage precision
- Coding style
   - Activate several Ruff rules
   - Lint check notebooks
   - Rewrite operator handling
   - Add a lot of new tests
- Miscellaneous
   - Add a contour line viewer
   - Plot lattice vectors in the view_atoms function
   - Add a NixOS CI test
   - Add a Nix lock file
   - Use Python 3.12 as the CI base image
   - Move Matplotlib to dev extras
   - Unpin the notebook version
   - Small performance improvements, e.g, in Atoms object creation
- Experimental
   - Smearing functionalities

----

v2.6.1 - Oct 04, 2023
---------------------
- New features
   - Add a Cell generation function
   - Add k-point generation functionalities
   - Add support to handle trajectory files
- Updated docs
   - Add a FOD optimization and a reduced density gradient example
   - Add references to data
- Miscellaneous
   - Breaking
      - Rename X to pos in Atoms
      - Merge R into a in Atoms
   - Indicate Python 3.12 support
   - Support viewing multiple files
   - Support non-cubic cells in Atoms, io, and viewer functions
   - Support viewing trajectory files
   - Fix Nix flake

v2.6.0 - Aug 07, 2023
---------------------
- New features
   - Complete rewrite of the Atoms and SCF classes
      - Easily allow systems with different charge or multiplicity
      - Document all public properties
      - Use properties when parsing input arguments
      - Allow direct setting of attributes
      - Better input handling
      - Use an Occupations object to store electronic states information in Atoms
      - Use a GTH object to store GTH data in SCF
      - Add some properties to the objects, e.g., the volume element dV in Atoms
      - Indicate non-input arguments and non-results as private or read-only
      - Breaking
         - Use unrestricted instead of Nspin
         - Use spin and charge instead of Nstate and f
         - Remove f and s as keyword arguments, can be set after initialization
         - Remove cgform as a keyword argument, use the run function to pass it to minimizers
         - Rename min keyword to opt
         - Merge symmetric with guess
   - Add DFT-D3 dispersion correction as an extra
- Updated docs
   - Add a theory introduction page
   - Add documentation to module data/constants
   - Add a list of all packages and their respective licenses
   - Re-add documentation of operators to Atoms
   - Add a custom functional example
   - Improve the geometry optimization example
   - Add PNGs to the downloads section
   - Sort attributes groupwise
   - Fix a lot of typos
- Coding style
   - Type check with mypy
   - Fix a lot of type warnings from mypy
   - Add type hints to scripts in docs and setup.py
   - Rename some arguments to not shadow builtins
- Miscellaneous
   - Create the eminus-benchmarks repository
      -  Move the SimpleDFT example to said repository
   - Small performance improvements, mostly for meta-GGAs
   - Add an error message when attempting to use operators of an unbuilt Atoms object
   - Add Matplotlib to the viewer setup to generate images in the examples
   - More tests, e.g, for different spin and charge states
   - Add a small demo function

----

v2.5.0 - Jul 10, 2023
---------------------
- New features
   - Add meta-GGA functionals!
      - Use all meta-GGAs that don't need a Laplacian from Libxc using pylibxc or PySCF
   - Improve minimizer
      - Add new auto minimizer that functions like pccg but can fallback to sd steps
      - Add Dai-Yuan conjugate-gradient form
      - Fancier-looking output from the minimizer
      - Option to converge the gradient norm
      - Print <S^2> after an unrestricted calculation
      - Add eigenenergies to the debug output
   - Improve file viewer
      - Support PDB files
      - Allow usage outside of notebooks
- Updated docs
   - Update the introduction page in the documentation
   - Upload the HTML coverage report
   - Add a simple geometry optimization example
- Coding style
   - Simplify H function
   - Simplify minimizer module
   - Reduce McCabe code complexity
   - Switch linter from flake8 to Ruff
   - Comply with different linting rules, e.g., use triple-quotes in docstrings
   - More tests and more coverage
- Miscellaneous
   - Performance fix by using precomputed values correctly
   - Improve GGA performance
   - Do an unpaired calculation automatically if the system is unpaired
   - Option to use a symmetric initial guess, i.e., the same guess for both spin channels
   - Add trajectory keyword to XYZ and PDB writer to append geometries
   - Read the field data from CUBE files
   - New functions for the
      - Electron localization function (ELF)
      - Positive-definite kinetic energy density
      - Reduced density gradient
      - Expectation value of S^2 and the multiplicity calculated from it
   - Option to set a path to directories containing GTH pseudopotential files
   - The SCF class now contains the xc_type and is_converged variables
   - Support functional parsing using pylibxc
   - Allow using custom densities when using the atoms viewer
   - Remove Gaussian initial guess
   - Remove exc_only keyword from functionals since it was basically unused
   - Fix GTH files not being installed when using the PyPI version
   - Fix mapping of field entries with the respective real-space coordinate
   - Fix GGA SIC evaluation

----

v2.4.0 - May 23, 2023
---------------------
- New features
   - Add GGA functionals!
      - Add internal PBE, PBEsol, and Chachiyo functionals
      - Option to use all GGAs from Libxc using pylibxc or PySCF
- Miscellaneous
   - Add Thomas-Fermi and von Weizsaecker kinetic energy density functions
   - Rewrite functionals for better readability
   - Fix Torch operators in some edge cases
   - Merge configuration files in tox.ini
   - Update minimum versions of dependencies

----

v2.3.0 - May 02, 2023
---------------------
- New features
   - Add Torch powered FFT operators as an extra
      - Up to 20% faster calculations
   - Add a consolidated configuration class
      - Easier configuration and more performance infos
   - Add a complete test suite
      - Add CI/CD coverage reports
   - Nix developer shell support
- Miscellaneous
   - Rewritten FODs guess function
   - Simplify the FOD interface in io and viewer
   - Fix a plethora of small bugs
   - Update Docker image to Python 3.11

----

v2.2.2 - Mar 03, 2023
---------------------
- New features
   - Improve performance, i.e, in operators, dotprod, and density calculations
   - Large and/or spin-polarized systems are much faster!
- Coding style
   - Make Energies a dataclass
- Miscellaneous
   - Drop Python 3.6 support
   - Raise minimum version SciPy from 1.4 to 1.6
   - Add repository statistics to the PyPI sidebar

v2.2.1 - Feb 22, 2023
---------------------
- Hotfix for the broken PyPI installation
- Use MANIFEST.in over package_data
- Skip tests if pylibxc is not installed

v2.2.0 - Feb 21, 2023
---------------------
- New features
   - Supercell Wannier localization
   - Rewritten xc parser
   - Modularize each functional
   - Greatly improve functional performance
   - Add modified functional variants
   - Modularize io module
   - Rewritten save and load functions to use JSON
   - Add a bunch of tests
   - Add a small ASCII logo in the info function
   - Update logo typography
- Updated docs
   - Add a nomenclature page of commonly used variables
   - Remove the package name from the module headings
   - Document members of classes
   - Add a germanium solid example
- Coding style
   - More secure coding practices
   - Remove the usage of eval, exec, and pickle
- Miscellaneous
   - Rename save and load to write_json and read_json
   - Fix PW spin-polarized functional
   - Align Chachiyo functional with Libxc
   - Add a recenter method to the Atoms and SCF class
   - Use pc-1 over pc-0 in the PyCOM extra
   - Add a pyproject.toml

----

v2.1.2 - Dec 15, 2022
---------------------
- New features
	- Add a Dockerfile and -container
	- Rewrite the grid view function as an atoms viewer
	- Use plotly over VisPy
	- Option to plot densities from SCF objects
- Updated docs
	- Add Docker instructions under Installation section
	- Update examples to use the new atoms viewer
- Miscellaneous
	- Unify read, write, and view functions
	- Add an optional density threshold for functionals
	- Add covalent radii and CPK colors to data
	- Add changelog to the PyPI description
	- Fix flake8 configuration file
	- Fix Libxc functional warnings

v2.1.1 - Oct 24, 2022
---------------------
- New features
	- Use the PySCF Libxc interface if pylibxc is not installed
	- Rework the addons/extras functionality inclusion
	- Dependencies can now be installed individually
	- Rework the Atoms object initialization
- Miscellaneous
	- Test different platforms and more Python versions in CI
	- Add kernel aliases to Atoms and SCF methods
	- Allow mixing Libxc and internal functionals
	- Add platform version in the info function
	- Improve logging in some places
	- Improve file writer formatting
	- Rename addons to extras
	- Rename filehandler to io
	- Update PyPI identifiers (e.g. to display Python 3.11 support)

v2.1.0 - Sep 19, 2022
---------------------
- New features
    - Support for spin-polarized calculations!
    - Rewritten GTH parser to use the CP2K file format
    - This adds support for the elements Ac to Lr
    - Built-in Chachiyo correlation functional
    - New pseudo-random starting guess for comparisons with SimpleDFT
- Updated docs
    - Improve displaying of examples in the documentation
    - Convert notebooks to HTML pages
    - New overview image
    - Minify pages
- Miscellaneous
    - Minimal versions for dependencies
    - GUI option for viewer and better examples
    - Rename Ns to Nstate to avoid confusion with Nspin
    - Adapt to newer NumPy RNG generators (use SFC64)
    - Update default numerical parameters
    - Option to set charge directly in atom when calculating single atoms
    - Adapt print precision from convergence tolerance
    - CI tests for the minimal Python version
    - Some code style improvements (e.g. using pathlib over os.path)
    - Misc performance improvements (e.g. in Ylm_real and get_Eewald)
    - Fix some bugs (e.g. the Libxc interface for spin-polarized systems)

----

v2.0.0 - May 20, 2022
---------------------
- Performance improved by 10-30%
- New features
   - SCF class
   - Domains
   - Libxc interface
   - Examples
   - CG minimizer
   - Simplify and optimize operators
- Updated docs
   - New theme with dark mode
   - Add examples, changelog, and license pages
   - Add dev information
   - Enable compression
- Coding style
   - Improve comments and references
   - A lot of refactoring and renaming
   - Google style docstrings
   - Use loggers
   - Unify coding style
   - Remove legacy code
- Miscellaneous
   - Improve setup.py
   - More tests
   - Improve readability
   - Fix various bugs

----

v1.0.1 - Nov 23, 2021
---------------------
- Add branding
- Fix GTH files not included in PyPI build

v1.0.0 - Nov 17, 2021
---------------------
- Initial release
