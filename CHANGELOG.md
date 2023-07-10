Changelog
=========

v2.5.0 - Jul 10, 2023
---------------------
- New features
   - Add meta-GGA functionals!
      - Use all meta-GGAs that don't need a Laplacian from Libxc using pylibxc or PySCF
   - Improved minimizer
      - Add new auto minimizer that functions like pccg but can fallback to sd steps
      - Add Dai-Yuan conjugate-gradient form
      - Fancier-looking output from the minimizer
      - Option to converge the gradient norm
      - Print <S^2> after an unrestricted calculation
      - Add eigenenergies to the debug output
   - Improved file viewer
      - Support PDB files
      - Allow usage outside of notebooks
- Updated docs
   - Update the introduction page in the documentation
   - Upload the HTML coverage report
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
- Coding style
   - More secure coding practices
   - Remove the usage of eval, exec, and pickle
- Miscellaneous
   - Rename save and load to write_json and read_json
   - Fix PW spin-polarized functional
   - Align Chachiyo functional with Libxc
   - Add a germanium solid example
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
