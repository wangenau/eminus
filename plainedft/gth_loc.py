#!/usr/bin/env python3
'''
Calculate the local potential with GTH pseudopotentials. Phys. Rev. B 54, 1703
'''
import numpy as np


def calc_Vloc(atoms):
    '''Local contribution for the GTH pseudopotential.'''
    G2 = atoms.G2
    atom = atoms.atom
    species = list(set(atom))
    Nspecies = len(species)

    Vsp = np.zeros(len(G2))
    Vloc = np.zeros(len(G2))

    for isp in range(Nspecies):
        psp = atoms.GTH[species[isp]]

        rloc = psp['rlocal']
        Zion = psp['Zval']
        C1 = psp['C'][0]
        C2 = psp['C'][1]
        C3 = psp['C'][2]
        C4 = psp['C'][3]

        omega = 1  # det(atoms.R)  # FIXME: is this correct?
        rlocG2 = G2[1:] * rloc**2

        Vsp[1:] = -4 * np.pi * Zion / omega * np.exp(-0.5 * rlocG2) / G2[1:] + \
                  np.sqrt((2 * np.pi)**3) * rloc**3 / omega * np.exp(-0.5 * rlocG2) * \
                  (C1 + C2 * (3 - rlocG2) + C3 * (15 - 10 * rlocG2 + rlocG2**2) +
                  C4 * (105 - 105 * rlocG2 + 21 * rlocG2**2 - rlocG2**3))
        # TODO: Apply to all elements with sqrt(G2) <1e-8
        Vsp[0] = 2 * np.pi * Zion * rloc**2 + \
                 (2 * np.pi)**1.5 * rloc**3 * (C1 + 3 * C2 + 15 * C3 + 105 * C4)

        # Sum up the structure factor for every species
        Sf = np.zeros(len(atoms.Sf[0]), dtype=complex)
        for ia in range(len(atom)):
            if atom[ia] == species[isp]:
                Sf += atoms.Sf[ia]
        Vloc += np.real(atoms.J(Vsp * Sf))
    return Vloc


def init_gth_loc(atoms):
    '''Goedecker, Teter, Hutter pseudopotentials.'''
    return calc_Vloc(atoms)
