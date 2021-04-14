#!/usr/bin/env python3
'''
Plot different properties.
'''
import matplotlib.pyplot as plt
import numpy as np
from .atoms import Atoms
from .potentials import init_pot
from .gth_loc import init_gth_loc


def plot_pot(atoms, rmax=2):
    '''Plot the GTH pseudopotential along with the coulomb potential.'''
    atom = atoms.atom
    lattice = rmax
    X = np.array([[0, 0, 0]])
    Z = atoms.Z
    Ns = atoms.Ns
    S = np.array([100, 1, 1])
    f = atoms.f
    ecut = atoms.ecut
    verbose = 0
    pot = 'gth'
    # Set up a dummy atoms object
    tmp = Atoms(atom, lattice, X, Z, Ns, S, f, ecut, verbose, pot)

    if len(atom) == 1:
        rloc = tmp.GTH[atom[0]]['rlocal']
    r = tmp.r[:, 0]  # Only use x coordinates, y and z are zero
    max = len(r) // 2  # Only plot half of the cell
    Vdual = init_gth_loc(tmp)
    GTH = np.real(Vdual)

    tmp.pot = 'COULOMB'  # Switch to coulomb potential so we dont get a key error
    COUL = np.real(init_pot(tmp))
    plt.plot(r[1:max], GTH[1:max], label=f'GTH for {tmp.atom}')
    plt.plot(r[1:max], COUL[1:max], label='Coulomb')
    if len(atom) == 1:
        plt.axvline(rloc, label='$r_{loc}$', c='grey', ls='--')
    plt.xlabel('Core distance [$a_0$]')
    plt.ylabel('Potential')
    plt.legend()
    plt.show()
    return


def plot_den(atoms):
    '''Plot the electronic density in real-space.'''
    # Plot over x- and y-axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mask = atoms.r[:, 2] == atoms.a / 2  # We only want to look at values in the middle of z
    ax.plot_trisurf(atoms.r[:, 0][mask], atoms.r[:, 1][mask], atoms.n[mask])
    ax.set_xlabel('x-axis', fontsize=12)
    ax.set_ylabel('y-axis', fontsize=12)
    ax.set_zlabel('Density', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Plot over z- and x-axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mask = atoms.r[:, 1] == atoms.a / 2
    ax.plot_trisurf(atoms.r[:, 2][mask], atoms.r[:, 0][mask], atoms.n[mask])
    ax.set_xlabel('z-axis', fontsize=12)
    ax.set_ylabel('x-axis', fontsize=12)
    ax.set_zlabel('Density', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Plot over y- and z-axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mask = atoms.r[:, 0] == atoms.a / 2
    ax.plot_trisurf(atoms.r[:, 1][mask], atoms.r[:, 2][mask], atoms.n[mask])
    ax.set_xlabel('y-axis', fontsize=12)
    ax.set_ylabel('z-axis', fontsize=12)
    ax.set_zlabel('Density', fontsize=12)
    plt.tight_layout()
    plt.show()
    return


def plot_den_iso(atoms, iso_max, iso_min=0):
    '''Plot the electronic density in real-space for isosurface values.'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mask = (atoms.n < iso_max) & (atoms.n > iso_min)
    ax.scatter(atoms.r[:, 0][mask], atoms.r[:, 1][mask], atoms.r[:, 2][mask])
    ax.scatter(atoms.X[:, 0], atoms.X[:, 1], atoms.X[:, 2], c='r', s=100)
    ax.set_xlabel('x-axis', fontsize=12)
    ax.set_ylabel('y-axis', fontsize=12)
    ax.set_zlabel('z-axis', fontsize=12)
    plt.tight_layout()
    plt.show()
    return
