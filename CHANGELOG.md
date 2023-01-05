Changelog
=========

dev
---
- New features
   - Rewritten save and load functions to use JSON
   - Add a bunch of tests
   - Add a small ASCII logo in the info function
- Miscellaneous
   - Rename save and load to write_json and read_json
   - Fix PW spin-polarized functional
   - Align Chachiyo functional with Libxc
   - Add a germanium solid example
   - More secure coding practices
   - Removed the usage of eval, exec, and pickle
   - Simplify PyCOM interface

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
	- Unified read, write, and view functions
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
	- Improved logging in some places
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
    - Improved displaying of examples in the documentation
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
   - Improved setup.py
   - More tests
   - Improve readability
   - Fix various bugs

v1.0.1 - Nov 23, 2021
---------------------
- Add branding
- Fix GTH files not included in PyPI build

v1.0.0 - Nov 17, 2021
---------------------
- Initial release
