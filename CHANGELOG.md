Changelog
=========

dev
---
- Miscellaneous
	- Test different platforms and more Python versions
	- Rework the addons/extras functionality inclusion
	- Rename filehandler to io

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
    - Fix some bugs (e.g. the LibXC interface for spin-polarized systems)

v2.0.0 - May 20, 2022
---------------------
- Performance improved by 10-30%
- New features
   - SCF class
   - Domains
   - LibXC interface
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
