#!/usr/bin/env python3
"""SCF class definition."""
import copy
import logging
import time

import numpy as np

from .dft import get_epsilon, guess_pseudo, guess_random
from .energies import Energy, get_Edisp, get_Eewald, get_Esic
from .gth import init_gth_loc, init_gth_nonloc
from .io import read_gth
from .logger import create_logger, get_level
from .minimizer import IMPLEMENTED as all_minimizer
from .potentials import IMPLEMENTED as all_potentials
from .potentials import init_pot
from .tools import center_of_mass, get_spin_squared
from .version import info
from .xc import parse_functionals, parse_xc_type


class SCF:
    """Perform direct minimizations.

    Args:
        atoms: Atoms object.

    Keyword Args:
        xc (str): Comma-separated exchange-correlation functional description (case insensitive).

            Adding 'libxc:' before a functional will try to use the Libxc interface.

            Example: 'lda,pw'; 'lda,'; ',vwn'; ','; 'libxc:LDA_X,libxc:7',
            Default: 'lda,vwn'
        pot (str): Type of pseudopotential (case insensitive).

            Example: 'GTH'; 'harmonic'; 'Coulomb'; 'Ge',
            Default: 'gth'
        guess (str): Initial guess method for the basis functions (case insensitive).

            Example: 'random'; 'rand'; 'pseudo',
            Default: 'random'
        etol (float): Convergence tolerance of the total energy.

            Default: 1e-7
        gradtol (float): Convergence tolerance of the gradient norm.

            Disabled by default. This tolerance will only be used in conjugate gradient methods.

            Default: None
        cgform (int): Conjugated-gradient form for the pccg minimization.

            1 for Fletcher-Reeves, 2 for Polak-Ribiere, 3 for Hestenes-Stiefel, and 4 for Dai-Yuan

            Default: 1
        sic (bool): Calculate the Kohn-Sham Perdew-Zunger SIC energy at the end of the SCF step.

            Default: False
        disp (bool | dict): Calculate a dispersion correction.

            Example: {'version': 'd3zero', 'atm': False, 'xc': 'scan'}
            Default: False
        symmetric (bool): Weather to use the same initial guess for both spin channels.

            Default: False
        min (dict | None): Dictionary to set the order and number of steps per minimization method.

            Example: {'sd': 10, 'pccg': 100}; {'pccg': 10, 'lm': 25, 'pclm': 50},
            Default: None (will default to {'pccg': 250})
        verbose (int | str): Level of output (case insensitive).

            Can be one of 'CRITICAL', 'ERROR', 'WARNING', 'INFO', or 'DEBUG'.
            An integer value can be used as well, where larger numbers mean more output, starting
            from 0.

            Default: 'info'
    """
    def __init__(self, atoms, xc='lda,vwn', pot='gth', guess='random', etol=1e-7, gradtol=None,
                 cgform=1, sic=False, disp=False, symmetric=False, min=None, verbose=None):
        """Initialize the SCF object."""
        self.atoms = atoms          # Atoms object
        self.xc = xc.lower()        # Exchange-correlation functional
        self.pot = pot              # Used pseudopotential
        self.guess = guess.lower()  # Initial wave functions guess
        self.etol = etol            # Total energy convergence tolerance
        self.gradtol = gradtol      # Gradient norm convergence tolerance
        self.cgform = cgform        # Conjugate gradient form
        self.sic = sic              # Calculate the SIC energy
        self.disp = disp            # Calculate the dispersion correction
        self.symmetric = symmetric  # Use the same initial guess for both spin channels
        self.min = min              # Minimization methods

        # Set min here, better not use mutable data types in signatures
        if self.min is None:
            # For systems with bad convergence throw in some sd steps
            self.min = {'auto': 250}

        # Initialize logger
        self.log = create_logger(self)
        if verbose is None:
            self.verbose = atoms.verbose
        else:
            self.verbose = verbose

        # Set up final and intermediate results
        self.W = None             # Basis functions
        self.n = None             # Electronic density
        self.energies = Energy()  # Energy object that holds energy contributions
        self.clear()

        # Parameters that will be built out of the inputs
        self.GTH = {}              # Dictionary of GTH parameters per atom species
        self.Vloc = None           # Local pseudopotential contribution
        self.NbetaNL = 0           # Number of projector functions for the non-local gth potential
        self.prj2beta = None       # Index matrix to map to the correct projector function
        self.betaNL = None         # Atomic-centered projector functions
        self.xc_type = None        # Type of functional that will be used
        self.psp = None            # Type of GTH pseudopotential that will be used
        self.is_converged = False  # Flag to determine if the object was converged or not
        self.initialize()

    def clear(self):
        """Initialize and clear intermediate results."""
        self.Y = None          # Orthogonal wave functions
        self.n_spin = None     # Electronic densities per spin
        self.dn_spin = None    # Gradient of electronic densities per spin
        self.tau = None        # Kinetic energy densities per spin
        self.phi = None        # Hartree field
        self.exc = None        # Exchange-correlation energy density
        self.vxc = None        # Exchange-correlation potential
        self.vsigma = None     # n times d exc/d |dn|^2
        self.vtau = None       # d exc/d tau
        self.precomputed = {}  # Dictionary of precomputed values not to be saved
        return self

    def initialize(self):
        """Validate inputs, update them and build all necessary parameters."""
        self.xc = parse_functionals(self.xc)
        self.xc_type = parse_xc_type(self.xc)
        # Build the atoms object if necessary and make a copy
        # This way the atoms object in scf is independent but we ensure that both atoms are build
        if not self.atoms.is_built:
            self.atoms = copy.copy(self.atoms.build())
        else:
            self.atoms = copy.copy(self.atoms)
        self._set_potential()
        self._init_W()
        return self

    def run(self, **kwargs):
        """Run the self-consistent field (SCF) calculation."""
        if self.log.level <= logging.DEBUG:
            info()
        self.log.debug(f'\n--- System information ---\n{self.atoms}\n'
                       f'Spin handling: {"un" if self.atoms.Nspin == 2 else ""}restricted\n'
                       f'Number of states: {self.atoms.Nstate}\n'
                       f'Occupation per state:\n{self.atoms.f}\n'
                       f'\n--- Cell information ---\nSide lengths: {self.atoms.a} a0\n'
                       f'Sampling per axis: {self.atoms.s}\n'
                       f'Cut-off energy: {self.atoms.ecut} Eh\n'
                       f'Compression: {len(self.atoms.G2c) / len(self.atoms.G2):.5f}\n'
                       f'\n--- Calculation information ---\n{self}\n\n--- SCF data ---')

        # Calculate Ewald energy that only depends on the system geometry
        self.energies.Eewald = get_Eewald(self.atoms)

        if 'mock_xc' in self.xc:
            self.log.warning('Usage of mock functional detected.')

        # Start minimization procedures
        Etots = []
        minimizer_log = {}
        for imin in self.min:
            try:
                self.log.info(f'Start {all_minimizer[imin].__name__}...')
            except KeyError:
                self.log.exception(f'No minimizer found for "{imin}".')
                raise
            start = time.perf_counter()
            Elist = all_minimizer[imin](self, self.min[imin], **kwargs)  # Call minimizer
            end = time.perf_counter()
            minimizer_log[imin] = {}  # Create an entry for the current minimizer
            minimizer_log[imin]['time'] = end - start  # Save time in dictionary
            minimizer_log[imin]['iter'] = len(Elist)  # Save iterations in dictionary
            Etots += Elist  # Append energies from minimizer
            # Do not start other minimizations if one converged
            if self.is_converged:
                break
        if self.is_converged:
            self.log.info(f'SCF converged after {len(Etots)} iterations.')
        else:
            self.log.warning('SCF not converged!')

        # Print SCF data
        self.log.debug('\n--- SCF results ---')
        t_tot = 0
        for imin in minimizer_log:
            N = minimizer_log[imin]['iter']
            t = minimizer_log[imin]['time']
            t_tot += t
            self.log.debug(f'Minimizer: {imin}'
                           f'\nIterations: {N}'
                           f'\nTime: {t:.5f} s'
                           f'\nTime/Iteration: {t / N:.5f} s')
        self.log.info(f'Total SCF time: {t_tot:.5f} s')

        # Calculate SIC energy if desired
        if self.sic:
            self.energies.Esic = get_Esic(self, self.Y)
        # Calculate dispersion correction energy if desired
        if isinstance(self.disp, dict):
            self.energies.Edisp = get_Edisp(self, **self.disp)
        elif self.disp:
            self.energies.Edisp = get_Edisp(self)

        # Print the S^2 expecation value for unrestricted calculations
        if self.atoms.Nspin == 2:
            self.log.info(f'<S^2> = {get_spin_squared(self):.6e}')
        # Print energy data
        if self.log.level <= logging.DEBUG:
            self.log.debug('\n--- Energy data ---\n'
                           f'Eigenenergies:\n{get_epsilon(self, self.W)}\n'
                           f'\n{self.energies}')
        else:
            self.log.info(f'Etot = {self.energies.Etot:.9f} Eh')
        return self.energies.Etot

    kernel = run

    def recenter(self, center=None):
        """Recenter the system inside the cell.

        Keyword Args:
            center (float | list | tuple | ndarray | None): Point to center the system around.
        """
        # Get the COM before centering the atoms
        com = center_of_mass(self.atoms.X)

        # Run the recenter method of the atoms object
        self.atoms.recenter(center=center)

        if center is None:
            dr = com - self.atoms.a / 2
        else:
            center = np.asarray(center)
            dr = com - center

        # Shift orbitals and density
        self.W = self.atoms.T(self.W, dr=-dr)
        # Transform the density to the reciprocal space, shift, and transform back
        Jn = self.atoms.J(self.n)
        TJn = self.atoms.T(Jn, dr=-dr)
        self.n = np.real(self.atoms.I(TJn))

        # Recalculate the pseudopotential since it depends on the structure factor
        self._set_potential()
        # Clear intermediate results to make sure no one uses the unshifted results
        self.clear()
        return self

    def _set_potential(self):
        """Build the potential."""
        atoms = self.atoms

        # If pot is no supported potential it can be a path to a directory containing GTH files
        if self.pot.lower() != 'gth' and self.pot.lower() not in all_potentials:
            self.psp = self.pot
            self.pot = 'gth'
        else:
            self.pot = self.pot.lower()
            if 'gga' in self.xc_type:
                self.psp = 'pbe'
            else:
                self.psp = 'pade'

        if self.pot == 'gth':
            for ia in range(atoms.Natoms):
                self.GTH[atoms.atom[ia]] = read_gth(atoms.atom[ia], atoms.Z[ia], psp_path=self.psp)
            # Set up the local and non-local part
            self.Vloc = init_gth_loc(self)
            self.NbetaNL, self.prj2beta, self.betaNL = init_gth_nonloc(self)
        else:
            self.Vloc = init_pot(self)
        return

    def _init_W(self):
        """Initialize wave functions."""
        if self.guess in ('rand', 'random'):
            # Start with randomized, complex basis functions with a random seed
            self.W = guess_random(self, symmetric=self.symmetric)
        elif self.guess in ('pseudo', 'pseudo_rand', 'pseudo_random'):
            # Start with pseudo-random numbers, mostly to compare with SimpleDFT
            self.W = guess_pseudo(self, symmetric=self.symmetric)
        else:
            self.log.error(f'No guess found for "{self.guess}".')
        return

    def __repr__(self):
        """Print the parameters stored in the SCF object."""
        # Use chr(10) to create a linebreak since backslashes are not allowed in f-strings
        return f'XC functionals: {self.xc}\n' \
               f'Potential: {self.pot}\n' \
               f'{f"GTH files: {self.psp}" + chr(10) if self.pot == "gth" else ""}' \
               f'Starting guess: {self.guess}\n' \
               f'Symmetric guess: {self.symmetric}\n' \
               f'Energy convergence tolerance: {self.etol} Eh\n' \
               f'Gradient convergence tolerance: {self.gradtol}\n' \
               f'Non-local contribution: {self.NbetaNL > 0}'

    @property
    def verbose(self):
        """Verbosity level."""
        return self._verbose

    @verbose.setter
    def verbose(self, level):
        self._verbose = get_level(level)
        self.log.verbose = self._verbose
        return


class RSCF(SCF):
    """SCF class for spin-paired systems.

    Inherited from :class:`eminus.scf.SCF`.

    In difference to the SCF class, this class will not build the original Atoms object, only the
    one attributed to the class.
    """
    def initialize(self):
        """Validate inputs, update them and build all necessary parameters."""
        self.atoms = copy.copy(self.atoms)
        self.atoms._set_states(Nspin=1)
        super().initialize()
        return self


class USCF(SCF):
    """SCF class for spin-polarized systems.

    Inherited from :class:`eminus.scf.SCF`.

    In difference to the SCF class, this class will not build the original Atoms object, only the
    one attributed to the class.
    """
    def initialize(self):
        """Validate inputs, update them and build all necessary parameters."""
        self.atoms = copy.copy(self.atoms)
        self.atoms._set_states(Nspin=2)
        super().initialize()
        return self
