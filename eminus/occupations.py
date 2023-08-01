#!/usr/bin/env python3
"""Determine occupations for atomic systems from simple inputs."""
import dataclasses
import numbers

import numpy as np

from .logger import log


@dataclasses.dataclass
class Occupations:
    """Occupations class to save electronic state information in one place.

    The attribute Nelec has to be given first after instantiation.
    """
    # Set the private variable for the attributes that are properties.
    _Nelec: int = 0          #: Number of electrons.
    _Nspin: int = 1          #: Number of spin states.
    _spin: int = 0           #: Number of unpaired electrons.
    _charge: int = 0         #: System charge.
    _Nstate: int = 0         #: Number of states.
    is_filled: bool = False  #: Determines the Occupations object fill status.

    # ### Class properties ###

    @property
    def Nelec(self):
        """Number of electrons."""
        return self._Nelec

    @Nelec.setter
    def Nelec(self, value):
        # Only update Nelec if it actually gets updated
        if self._Nelec != int(value):
            self._Nelec = int(value)
            self.is_filled = False

    @property
    def Nspin(self):
        """Number of spin states."""
        return self._Nspin

    @Nspin.setter
    def Nspin(self, value):
        # Use a spin-paired calculation for an even number of electrons
        if value is None:
            if self.Nelec % 2 == 0:
                value = 1
            else:
                value = 2
        # Only update if needed
        if self._Nspin != int(value):
            self._Nspin = int(value)
            self.spin = self.spin
            self.is_filled = False
            if hasattr(self, 'f'):
                log.warning('Reset previously set fillings.')

    @property
    def spin(self):
        """Number of unpaired electrons."""
        return self._spin

    @spin.setter
    def spin(self, value):
        # If no spin is given try to determine it from the number of electrons
        if value is None:
            if self.Nelec % 2 == 0:
                value = 0
            else:
                value = 1
        if self._spin != int(value):
            self.is_filled = False
        # We have no spin in the spin-paired case
        if self.Nspin == 1:
            self._spin = 0
        else:
            self._spin = int(value)

    @property
    def charge(self):
        """System charge."""
        return self._charge

    @charge.setter
    def charge(self, value):
        # If we set a charge via this setter update the number of electrons
        if hasattr(self, 'Nelec'):
            self.Nelec += self._charge - int(value)
        if self._charge != int(value):
            self._charge = int(value)
            self.is_filled = False

    @property
    def f(self):
        """Occupation numbers per state."""
        return self._f

    @f.setter
    def f(self, value):
        # Make sure the occupations are in a two-dimensional array
        if isinstance(value, (list, tuple, np.ndarray)):
            value = np.atleast_2d(value)
        self.is_filled = False
        # This setter will only be called when explicitly setting f
        # Call the fill function in that case
        self.fill(value)

    # ### Read-only properties ###

    @property
    def Nstate(self):
        """Number of states."""
        return self._Nstate

    @property
    def F(self):
        """Diagonal matrices of f per spin."""
        return [np.diag(f) for f in self.f]

    # ### Class methods ###

    def fill(self, f=None):
        """Fill the states of the object.

        Keyword Args:
            f: (float | ndarray | None): Fillings.
        """
        # Do nothing if the object is already filled
        if self.is_filled:
            return self
        # If no f is given just use the standard fillings: 2 for restricted and 1 for unrestricted
        if f is None:
            f = 2 / self.Nspin
        self._update_from_fillings(f)
        # Assure that no electrons have been lost
        if np.sum(self.f) != self.Nelec:
            ValueError(f'Sum of fillings ({np.sum(self.f)}) differs from Nelec ({self.Nelec}).')
        self.is_filled = True
        return self

    kernel = fill

    def _update_from_fillings(self, value):
        """Update fillings."""
        # Do not use the setter methods in this place to not trigger the setter effects
        # If f is a number use this occupation for all states
        if isinstance(value, numbers.Real):
            # Do not leave the states array empty when no electrons are present
            if self.Nelec <= 0:
                self._Nstate = 1
                self._f = np.zeros((self.Nspin, 1))
            elif self.Nspin == 1 or self.Nelec % 2 == self.spin % 2:
                self._integer_fillings(value)
            else:
                log.warning('Using fractional occupations.')
                self._fractional_fillings(value)
        # If f is an array redetermine all attributes from it
        else:
            self._f = value
            self._Nspin = len(value)
            self._Nstate = len(value[0])
            self._charge += self.Nelec - int(np.sum(value))
            self.Nelec = int(np.sum(value))
            if self.Nspin == 1:
                self.spin = 0
            else:
                self.spin = abs(np.sum(self.f[0] - self.f[1]))

    def _integer_fillings(self, f):
        """Update fillings while maintaining integer occupations numbers.

        Keyword Args:
            f: (int | ndarray): Fillings.
        """
        # Determine the electrons per spin channel
        Nup = self.Nelec // self.Nspin + self.spin // self.Nspin + self.Nelec % self.Nspin
        Ndw = self.Nelec // self.Nspin - self.spin // self.Nspin
        elecs = np.array([Nup, Ndw])
        # Get the number of states
        self._Nstate = int(np.ceil(max(elecs / f)))
        # Simply build the occupations array
        self._f = f * np.ones((self.Nspin, self._Nstate), dtype=int)
        # If we have filled too much correct it in the second spin channel
        # The following procedure ensures that we have no negative fillings
        rest = np.sum(self._f) - self.Nelec
        i = 1
        while rest > 0:
            # If the occupation is greater than the rest we are done with removing electrons
            if self._f[-1, -i] >= rest:
                self._f[-1, -i] -= rest
                break
            # Otherwise zero out the state, update the rest, and move to the next state
            rest -= self._f[-1, -i]
            self._f[-1, -i] = 0
            i += 1

    def _fractional_fillings(self, f):
        """Update fillings while allowing fractional occupation numbers.

        Keyword Args:
            f: (float | ndarray): Fillings.
        """
        # Determine the electrons per spin channel
        Nup = self.Nelec / self.Nspin + self.spin / self.Nspin
        Ndw = self.Nelec / self.Nspin - self.spin / self.Nspin
        elecs = np.array([Nup, Ndw])
        # Get the number of states
        self._Nstate = int(np.ceil(max(elecs / f)))
        # Simply build the occupations array
        self._f = f * np.ones((self.Nspin, self._Nstate))
        # If we have filled too much correct it in both spin channels
        # The following procedure ensures that we have no negative fillings
        rest = np.sum(self._f, axis=1) - elecs
        for s in range(self.Nspin):
            i = 1
            while rest[s] > 0:
                # If the occupation is greater than the rest we are done with removing electrons
                if self._f[s, -i] >= rest[s]:
                    self._f[s, -i] -= rest[s]
                    break
                # Otherwise zero out the state, update the rest, and move to the next state
                rest[s] -= self._f[s, -i]
                self._f[s, -i] = 0
                i += 1

    def __repr__(self):
        """Print the parameters stored in the Occupations object."""
        return f'Spin handling: {"un" if self.Nspin == 2 else ""}restricted\n' \
               f'Number of electrons: {self.Nelec}\n' \
               f'Spin: {self.spin}\n' \
               f'Charge: {self.charge}\n' \
               f'Number of states: {self._Nstate}\n' \
               f'Fillings: \n{self.f if self.is_filled else "Not filled"}'
