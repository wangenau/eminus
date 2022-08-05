Changelog
=========

dev
---

- New features
    - Support for spin polarized-systems!
    - Rewritten GTH parser to use the CP2K file format
    - This adds support for the elements Ac to Lr
- Updated docs
    - Improved displaying of examples in documentation
    - Convert notebooks to HTML pages
    - New overview image
    - Minify pages
- Misc
    - Minimal versions for dependencies
    - GUI option for viewer and better examples
    - Rename Ns to Nstate to avoid confusion with Nspin
    - Set charge directly in atom when calculating single atoms
    - CI tests for the minimal Python version
    - Some code style improvements (e.g. using pathlib over os.path)
    - Misc performance improvements (e.g. in Ylm_real and get_Eewald)
    - Fix some bugs (e.g. the LibXC interface for spin-polarized systems)

v2.0.0 - May 20, 2020
---------------------
- Performance improved by 10-30%
- New features
   - SCF class
   - Domains
   - LibXC interface
   - Examples
   - cg minimizer
   - Simplify and optimize operators
- Updated docs
   - New theme with dark mode
   - Add examples, changelog and license pages
   - Add dev information
   - Enable compression
- Coding style
   - Improve comments and references
   - A lot of refactoring and renaming
   - Google style docstrings
   - Use loggers
   - Unify coding style
   - Remove legacy code
- Misc
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
