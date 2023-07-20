# examples

Examples that act as a tutorial for the most important functionalities of the code.

| Folder                      | Description |
| :-------------------------: | :---------: |
| 00_installation_test        | Test installation and display versions |
| 01_minimal_example          | Minimal DFT calculation |
| 03_atoms_objects            | Atoms object introduction |
| 04_dft_calculations         | Advanced DFT calculations |
| 05_input_output             | In-/Output functionalities |
| 06_advanced_functionalities | Advanced functionalities and workflows |
| 07_fod_extra                | FOD and FLO generation |
| 08_visualizer_extra         | Visualization features |
| 09_sic_calculations         | Calculate SIC energies |
| 10_domain_generation        | Generate domains and use them |
| 11_simpledft_examples       | Comparison to SimpleDFT codes |
| 12_germanium_solid          | Calculate a solid using periodicity |
| 13_wannier_localization     | Wannier orbital localization |
| 14_geometry_optimization    | Optimization of the H2 bond distance |
| 15_custom_functionals       | Usage of customized functionals |
| 16_custom_functionals_spin  | Extension of example 15 for the spin-polarized case |

This folder will automatically be parsed to generate the documentation.
The following structure has to be kept
* Each example has its own subfolder
* Each subfolder contains a README.rst with a title
* Each subfolder can contain one Python script or one Jupyter notebook with the same name as the subfolder
* For other files a download button will be generated
* Comments that should be displayed as text start with # #
* Comments that should be displayed in code blocks start with #
* Images should be saved as pngs with unique names
* Each image should be referenced in the example code in a comment starting with # #
