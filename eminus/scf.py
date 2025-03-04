# SPDX-FileCopyrightText: 2021 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""SCF class definition."""

import copy
import logging
import time

import numpy as np

from . import config
from .band_minimizer import get_grad_unocc, scf_step_unocc
from .band_minimizer import IMPLEMENTED as BAND_MINIMIZER
from .dft import (
    get_epsilon,
    get_n_spin,
    get_n_total,
    get_phi,
    guess_pseudo,
    guess_random,
    orth,
)
from .energies import Energy, get_Edisp, get_Eewald, get_Esic
from .gga import get_grad_field, get_tau
from .gth import GTH
from .logger import create_logger, get_level
from .minimizer import IMPLEMENTED as ALL_MINIMIZER
from .potentials import get_pot_defaults, init_pot
from .potentials import IMPLEMENTED as ALL_POTENTIALS
from .tools import center_of_mass, get_spin_squared
from .utils import BaseObject
from .version import info
from .xc import get_xc, get_xc_defaults, parse_functionals, parse_xc_type


class SCF(BaseObject):
    """Perform direct minimizations.

    Args:
        atoms: Atoms object.

    Keyword Args:
        xc: Comma-separated exchange-correlation functional description.

            Adding "libxc:" before a functional will use the Libxc interface for that functional,
            e.g., with :code:`libxc:mgga_x_scan,libxc:mgga_c_scan`.
        pot: Type of potential.

            Can be one of "GTH" (default), "Coulomb", "Harmonic", or "Ge". Alternatively, a path to
            a directory containing custom GTH pseudopotential files can be given.
        guess: Initial guess for the wave functions.

            Can be one of "random" (default) or "pseudo". Adding "symm" to the string will use the
            same coefficients for both spin channels, e.g., :code:`symm-rand`.
        etol: Convergence tolerance of the total energy.
        gradtol: Convergence tolerance of the gradient norm.

            This tolerance will only be used in conjugate gradient methods.
        sic: Calculate the Kohn-Sham Perdew-Zunger SIC energy at the end of the SCF.
        disp: Calculate a dispersion correction.

            A dictionary can be used to pass arguments to the respective
            function, e.g., with :code:`{"version": "d3zero", "atm": False, "xc": "scan"}`.
        opt: Dictionary to customize the minimization methods.

            The keys can be chosen out of "sd", "lm", "pclm", "cg", "pccg", and "auto". Defaults to
            :code:`{"auto": 250}`.
        verbose: Level of output.

            Can be one of "critical", "error", "warning", "info" (default), or "debug". An integer
            value can be used as well, where larger numbers mean more output, starting from 0.
            Defaults to the verbosity level of the Atoms object.
    """

    def __init__(
        self,
        atoms,
        xc="lda,vwn",
        pot="gth",
        guess="random",
        etol=1e-7,
        gradtol=None,
        sic=False,
        disp=False,
        opt=None,
        verbose=None,
    ):
        """Initialize the SCF object."""
        # Set opt here, better to not use mutable data types in signatures
        if opt is None:
            opt = {"auto": 250}

        # Set the input parameters (the ordering is important)
        self.atoms = atoms  #: Atoms object.
        self._log = create_logger(self)  #: Logger object.
        self.verbose = verbose  #: Verbosity level.
        self.xc = xc  #: Exchange-correlation functional.
        self.pot_params = {}  #: Potential parameters.
        self.pot = pot  #: Used potential.
        self.guess = guess  #: Initial wave functions guess.
        self.etol = etol  #: Total energy convergence tolerance.
        self.gradtol = gradtol  #: Gradient norm convergence tolerance.
        self.sic = sic  #: Enables the SIC energy calculation.
        self.disp = disp  #: Enables the dispersion correction calculation.
        self.opt = opt  #: Minimization methods.
        self.smear_update = 2  #: Steps after the smeared occupations are recalculated.

        # Initialize other attributes
        self.energies = Energy()  #: Energy object holding energy contributions.
        self.is_converged = False  #: Determines the SCF object convergence.
        self.W = None  #: Unconstrained wave functions.
        self.xc_params = {}  #: Exchange-correlation functional parameters.
        self.clear()

    # ### Class properties ###

    @property
    def atoms(self):
        """Atoms object."""
        return self._atoms

    @atoms.setter
    def atoms(self, value):
        # Build the Atoms object if necessary and make a copy
        # This way the Atoms objects inside and outside the class are independent but both are build
        if not value.is_built:
            self._atoms = copy.deepcopy(value.build())
        else:
            self._atoms = copy.deepcopy(value)

    @property
    def xc(self):
        """Exchange-correlation functional."""
        return self._xc

    @xc.setter
    def xc(self, value):
        self._xc = parse_functionals(value.lower())
        # Determine the type of functional combinations
        self._xc_type = parse_xc_type(self._xc)
        if "mock_xc" in self._xc and "_xc_" not in "".join(self._xc).lower():
            self._log.warning("Usage of mock functional detected.")

    @property
    def xc_params(self):
        """Exchange-correlation functional parameters."""
        return self._xc_params

    @xc_params.setter
    def xc_params(self, value):
        # Check if some parameters are unused in the functionals
        # This also ensures that we print a warning in case of overlapping parameters in x and c
        if value != {} and value is not None:
            not_used = value.keys() - self.xc_params_defaults.keys()
            if len(not_used) > 0:
                self._log.warning(f"Some xc_params are unused, namely: {', '.join(not_used)}.")
        self._xc_params = value

    @property
    def pot(self):
        """Used potential."""
        return self._pot

    @pot.setter
    def pot(self, value):
        if value.lower() in ALL_POTENTIALS:
            self._pot = value.lower()
            # Only set the pseudopotential type for GTH pseudopotentials
            if self._pot == "gth":
                if "gga" in self._xc_type:
                    self._psp = "pbe"
                else:
                    self._psp = "pade"
        # If pot is no supported potential treat it as a path to a directory containing GTH files
        else:
            self._log.info(f'Use the path "{value}" to search for GTH pseudopotential files.')
            self._psp = value
            self._pot = "gth"
        # Build the potential
        if self._pot == "gth":
            self.gth = GTH(self)
        self.Vloc = init_pot(self, self.pot_params)

    @property
    def pot_params(self):
        """Potential parameters."""
        return self._pot_params

    @pot_params.setter
    def pot_params(self, value):
        # Check if some parameters are unused in the potential
        if value != {} and value is not None:
            not_used = value.keys() - self.pot_params_defaults.keys()
            if len(not_used) > 0:
                self._log.warning(f"Some pot_params are unused, namely: {', '.join(not_used)}.")
        self._pot_params = value
        # Update the local potential for the new parameters
        if hasattr(self, "pot"):
            self.Vloc = init_pot(self, self.pot_params)

    @property
    def guess(self):
        """Initial wave functions guess."""
        return self._guess

    @guess.setter
    def guess(self, value):
        # Set the guess method
        value = value.lower()
        if "rand" in value:
            self._guess = "random"
        elif "pseudo" in value:
            self._guess = "pseudo"
        else:
            msg = f"{value} is no valid initial guess."
            raise ValueError(msg)
        # Check if a symmetric or unsymmetric guess is selected
        if "sym" in value and "unsym" not in value:
            self._symmetric = True
        else:
            self._symmetric = False

    @property
    def opt(self):
        """Minimization methods."""
        return self._opt

    @opt.setter
    def opt(self, value):
        # Set lowercase to all keys
        value = {k.lower(): v for k, v in value.items()}
        for opt in value:
            if opt not in ALL_MINIMIZER:
                msg = f'No minimizer found for "{opt}".'
                raise KeyError(msg)
        self._opt = value

    @property
    def verbose(self):
        """Verbosity level."""
        return self._verbose

    @verbose.setter
    def verbose(self, value):
        # If no verbosity is given use the one from the Atoms object
        if value is None:
            value = self.atoms.verbose
        self._verbose = get_level(value)
        self._log.verbose = self._verbose

    # ### Read-only properties ###

    @property
    def kpts(self):
        """Pass-through to the KPoints object of the Atoms object."""
        return self.atoms.kpts

    @property
    def pot_params_defaults(self):
        """Get the default potential parameters."""
        return get_pot_defaults(self.pot)

    @property
    def psp(self):
        """Pseudopotential path."""
        return self._psp

    @property
    def symmetric(self):
        """Determines if the initial guess is the same for both spin channels."""
        return self._symmetric

    @property
    def xc_type(self):
        """Determines the exchange-correlation family."""
        return self._xc_type

    @property
    def xc_params_defaults(self):
        """Get the default exchange-correlation functional parameters."""
        return get_xc_defaults(self.xc)

    # ### Class methods ###

    def run(self, **kwargs):
        """Run the self-consistent field (SCF) calculation.

        Keyword Args:
            **kwargs: Pass-through keyword arguments.

        Returns:
            Total energy.
        """
        # Print some information about the calculation
        if self._log.level <= logging.DEBUG:
            info()
        if ":" in "".join(self._xc) and config.use_pylibxc:
            self._log.info(
                "This calculation employs Libxc to evaluate density functionals. When using Libxc,"
                "\nplease cite SoftwareX 7, 1 (2018) (doi:10.1016/j.softx.2017.11.002)."
            )
        self._log.debug(
            f"\n--- Atoms information ---\n{self.atoms}\n"
            f"\n--- Cell information ---\nCell vectors:\n{self.atoms.a} a0\n"
            f"Sampling per axis: {self.atoms.s}\n"
            f"Cut-off energy: {self.atoms.ecut} Eh\n"
            f"\n--- State information ---\n{self.atoms.occ}\n"
            f"\n--- Calculation information ---\n{self}\n\n--- SCF data ---"
        )

        # Calculate Ewald energy that only depends on the system geometry
        self.energies.Eewald = get_Eewald(self.atoms)
        # Build the initial wave function if there is no W to start from
        if self.W is None:
            if "random" in self.guess:
                self.W = guess_random(self, symmetric=self.symmetric)
            elif "pseudo" in self.guess:
                self.W = guess_pseudo(self, symmetric=self.symmetric)

        # Start the minimization procedures
        self.clear()
        Etots = []
        for imin in self.opt:
            # Call the minimizer
            self._log.info(f"Start {ALL_MINIMIZER[imin].__name__}...")
            start = time.perf_counter()
            Elist = ALL_MINIMIZER[imin](self, self.opt[imin], **kwargs)
            end = time.perf_counter()
            # Save the minimizer results
            self._opt_log[imin] = {}
            self._opt_log[imin]["iter"] = len(Elist)
            self._opt_log[imin]["time"] = end - start
            Etots += Elist
            # Do not start other minimizations if one converged
            if self.is_converged:
                break
        if self.is_converged:
            self._log.info(f"SCF converged after {len(Etots)} iterations.")
        else:
            self._log.warning("SCF not converged!")

        # Calculate SIC energy if needed
        if self.sic:
            self.energies.Esic = get_Esic(self, self.Y)
        # Calculate dispersion correction energy if needed
        if isinstance(self.disp, dict):
            self.energies.Edisp = get_Edisp(self, **self.disp)
        elif self.disp:
            self.energies.Edisp = get_Edisp(self)

        # Print minimizer timings
        self._log.debug("\n--- SCF results ---")
        t_tot = 0
        for imin in self._opt_log:
            N = self._opt_log[imin]["iter"]
            t = self._opt_log[imin]["time"]
            t_tot += t
            self._log.debug(
                f"Minimizer: {imin}"
                f"\nIterations: {N}"
                f"\nTime: {t:.5f} s"
                f"\nTime/Iteration: {t / N:.5f} s"
            )
        self._log.info(f"Total SCF time: {t_tot:.5f} s")
        # Print the S^2 expectation value for unrestricted calculations
        if self.atoms.unrestricted:
            self._log.info(f"<S^2> = {get_spin_squared(self):.6e}")
        # Print energy data
        if self._log.level <= logging.DEBUG:
            self._log.debug(
                "\n--- Energy data ---\n"
                f"Eigenenergies:\n{get_epsilon(self, self.W)}\n\n{self.energies}"
            )
        else:
            self._log.info(f"Etot = {self.energies.Etot:.9f} Eh")
        return self.energies.Etot

    kernel = run

    def converge_bands(self, **kwargs):
        """Converge occupied bands after an SCF calculation.

        Keyword Args:
            **kwargs: Pass-through keyword arguments.
        """
        if not self.is_converged:
            self._log.warning("The previous calculation has not been converged.")

        # If new k-points have been set rebuild the atoms object and the potential
        if not self.atoms.kpts.is_built or (
            self.W is not None and len(self.W) != self.atoms.kpts.Nk
        ):
            self.atoms.build()
            self.pot = self.pot
            self.is_converged = False

        # Build the initial wave function if there is no W to start from
        if self.W is None or len(self.W) != self.atoms.kpts.Nk:
            if "random" in self.guess:
                self.W = guess_random(self, symmetric=self.symmetric)
            elif "pseudo" in self.guess:
                self.W = guess_pseudo(self, symmetric=self.symmetric)

        self._log.info("Minimize occupied band energies...")
        # Start the minimization procedures
        Etots = []
        for imin in self.opt:
            # Call the minimizer
            self._log.info(f"Start {BAND_MINIMIZER[imin].__name__}...")
            start = time.perf_counter()
            Elist, self.W = BAND_MINIMIZER[imin](self, self.W, self.opt[imin], **kwargs)
            end = time.perf_counter()
            # Save the minimizer results
            self._opt_log[imin] = {}
            self._opt_log[imin]["iter"] = len(Elist)
            self._opt_log[imin]["time"] = end - start
            Etots += Elist
            # Do not start other minimizations if one converged
            if self.is_converged:
                break
        if self.is_converged:
            self._log.info(f"Band minimization converged after {len(Etots)} iterations.")
        else:
            self._log.warning("Band minimization not converged!")

        # Print minimizer timings
        self._log.debug("\n--- Band minimization results ---")
        t_tot = 0
        for imin in self._opt_log:
            N = self._opt_log[imin]["iter"]
            t = self._opt_log[imin]["time"]
            t_tot += t
            self._log.debug(
                f"Minimizer: {imin}"
                f"\nIterations: {N}"
                f"\nTime: {t:.5f} s"
                f"\nTime/Iteration: {t / N:.5f} s"
            )
        self._log.info(f"Total band minimization time: {t_tot:.5f} s")

        # Converge empty bands automatically if needed
        if self.atoms.occ.Nempty > 0:
            self.converge_empty_bands(**kwargs)
        return self

    def converge_empty_bands(self, Nempty=None, **kwargs):
        """Converge unoccupied bands after converging occ. bands.

        Keyword Args:
            Nempty: Number of empty states.
            **kwargs: Pass-through keyword arguments.
        """
        if not self.is_converged:
            self._log.warning("The previous calculation has not been converged.")
        self.is_converged = False

        if Nempty is None:
            Nempty = self.atoms.occ.Nempty

        # Build the initial wave functions
        if self.Z is None:
            if "random" in self.guess:
                self.Z = guess_random(self, Nempty, symmetric=self.symmetric)
            elif "pseudo" in self.guess:
                self.Z = guess_pseudo(self, Nempty, symmetric=self.symmetric)

        self._log.info("Minimize unoccupied band energies...")
        # Start the minimization procedures
        Etots = []
        for imin in self.opt:
            # Call the minimizer
            self._log.info(f"Start {BAND_MINIMIZER[imin].__name__}...")
            start = time.perf_counter()
            Elist, self.Z = BAND_MINIMIZER[imin](
                self, self.Z, self.opt[imin], cost=scf_step_unocc, grad=get_grad_unocc, **kwargs
            )
            end = time.perf_counter()
            # Save the minimizer results
            self._opt_log[imin] = {}
            self._opt_log[imin]["iter"] = len(Elist)
            self._opt_log[imin]["time"] = end - start
            Etots += Elist
            # Do not start other minimizations if one converged
            if self.is_converged:
                break
        if self.is_converged:
            self._log.info(f"Band minimization converged after {len(Etots)} iterations.")
        else:
            self._log.warning("Band minimization not converged!")

        # Print minimizer timings
        self._log.debug("\n--- Band minimization results ---")
        t_tot = 0
        for imin in self._opt_log:
            N = self._opt_log[imin]["iter"]
            t = self._opt_log[imin]["time"]
            t_tot += t
            self._log.debug(
                f"Minimizer: {imin}"
                f"\nIterations: {N}"
                f"\nTime: {t:.5f} s"
                f"\nTime/Iteration: {t / N:.5f} s"
            )
        self._log.info(f"Total band minimization time: {t_tot:.5f} s")
        return self

    def recenter(self, center=None):
        """Recenter the system inside the cell.

        Keyword Args:
            center: Point to center the system around.
        """
        atoms = self.atoms
        # Get the COM before centering the atoms
        com = center_of_mass(atoms.pos)
        # Run the recenter method of the atoms object
        self.atoms.recenter(center=center)
        if center is None:
            dr = com - np.sum(atoms.a, axis=0) / 2
        else:
            center = np.asarray(center)
            dr = com - center

        # Shift orbitals and density
        if self.W is not None:
            self.W = atoms.T(self.W, dr=-dr)
        # Transform the density to the reciprocal space, shift, and transform back
        n = None
        if self.n is not None:
            Jn = atoms.J(self.n)
            TJn = atoms.T(Jn, dr=-dr)
            n = np.real(atoms.I(TJn))

        # Recalculate the potential since it depends on the structure factor
        self.pot = self.pot
        # Clear intermediate results to make sure no one uses the unshifted results
        self.clear()
        # Set the shifted density after calling the clearing function
        self.n = n
        return self

    def clear(self):
        """Initialize or clear intermediate results."""
        self.Y = None  # Orthogonal wave functions
        self.Z = None  # Unconstrained wave functions of unoccupied states
        self.D = None  # Orthogonal wave functions of unoccupied states
        self.n = None  #: Electronic density
        self.n_spin = None  # Electronic densities per spin
        self.dn_spin = None  # Gradient of electronic densities per spin
        self.tau = None  # Kinetic energy densities per spin
        self.phi = None  # Hartree field
        self.exc = None  # Exchange-correlation energy density
        self.vxc = None  # Exchange-correlation potential
        self.vsigma = None  # n times d exc/d |dn|^2
        self.vtau = None  # d exc/d tau
        self._precomputed = {}  # Dictionary of pre-computed values not to be saved
        self._opt_log = {}  # Log of the optimization procedure
        return self

    @staticmethod
    def callback(scf, step):
        """Callback function that will get called every SCF iteration.

        This is just an empty function users can overwrite with their implementation.
        Remember that the "auto" minimization can call the callback function twice when the pccg
        step is not preferred.

        Args:
            scf: SCF object.
            step: Optimization step.
        """

    def _precompute(self):
        """Precompute fields stored in the SCF object."""
        atoms = self.atoms
        self.Y = orth(atoms, self.W)
        self.n_spin = get_n_spin(atoms, self.Y)
        self.n = get_n_total(atoms, self.Y, self.n_spin)
        if "gga" in self.xc_type:
            self.dn_spin = get_grad_field(atoms, self.n_spin)
        if self.xc_type == "meta-gga":
            self.tau = get_tau(atoms, self.Y)
        self.phi = get_phi(atoms, self.n)
        self.exc, self.vxc, self.vsigma, self.vtau = get_xc(
            self.xc, self.n_spin, atoms.occ.Nspin, self.dn_spin, self.tau, self.xc_params
        )
        self._precomputed = {
            "dn_spin": self.dn_spin,
            "phi": self.phi,
            "vxc": self.vxc,
            "vsigma": self.vsigma,
            "vtau": self.vtau,
        }
        return self

    def __repr__(self):
        """Print the most important parameters stored in the SCF object."""
        # Use chr(10) to create a linebreak since backslashes are not allowed in f-strings
        return (
            f"XC functionals: {self.xc}\n"
            f"Potential: {self.pot}\n"
            f"{f'GTH files: {self.psp}' + chr(10) if self.pot == 'gth' else ''}"
            f"Starting guess: {self.guess}\n"
            f"Symmetric guess: {self.symmetric}\n"
            f"Energy convergence tolerance: {self.etol} Eh\n"
            f"Gradient convergence tolerance: {self.gradtol}\n"
            f"Non-local potential: {self.gth.NbetaNL > 0 if self.pot == 'gth' else 'false'}\n"
            f"Smearing: {self.atoms.occ.smearing > 0}\n"
            f"Smearing update cycle: {self.smear_update}"
        )


class RSCF(SCF):
    """SCF class for spin-paired systems.

    Inherited from :class:`eminus.scf.SCF`.

    In difference to the SCF class, this class will not build the original Atoms object, only the
    one attributed to the class. Customized fillings could be overwritten when using this class.
    """

    @property
    def atoms(self):
        """Atoms object."""
        return self._atoms

    @atoms.setter
    def atoms(self, value):
        self._atoms = copy.deepcopy(value)
        self._atoms.unrestricted = False
        self._atoms.occ.fill()
        if not self._atoms.is_built:
            self._atoms = self._atoms.build()


class USCF(SCF):
    """SCF class for spin-polarized systems.

    Inherited from :class:`eminus.scf.SCF`.

    In difference to the SCF class, this class will not build the original Atoms object, only the
    one attributed to the class. Customized fillings could be overwritten when using this class.
    """

    @property
    def atoms(self):
        """Atoms object."""
        return self._atoms

    @atoms.setter
    def atoms(self, value):
        self._atoms = copy.deepcopy(value)
        self._atoms.unrestricted = True
        self._atoms.occ.fill()
        if not self._atoms.is_built:
            self._atoms = self._atoms.build()
