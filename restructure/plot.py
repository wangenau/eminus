import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf
from gth import GTH_LDA_Abinit


def Coulomb_real(a, r):
    '''Coulomb potential in real space.'''
    return -a.Z / np.abs(r)


def Vloc_real(a, r):
    '''Local contribution for the GTH pseudopotential in real space.'''
    rloc = GTH_LDA_Abinit[a.atom]['rloc']
    Zion = GTH_LDA_Abinit[a.atom]['Zion']
    C1 = GTH_LDA_Abinit[a.atom]['C1']
    C2 = GTH_LDA_Abinit[a.atom]['C2']
    C3 = GTH_LDA_Abinit[a.atom]['C3']
    C4 = GTH_LDA_Abinit[a.atom]['C4']

    V = -a.Z / r * erf(r / np.sqrt(2) / rloc) + \
        np.exp(-0.5 * (r / rloc)**2) * \
        (C1 + C2 * (r / rloc)**2 + C3 * (r / rloc)**4 + C4 *(r / rloc)**6)
    return V


def plot_pot(a, rmin=0.05, rmax=2):
    '''Plot the GTH pseudopotential along with the coulomb potential.'''
    rloc = GTH_LDA_Abinit[a.atom]['rloc']
    r = np.arange(rmin, rmax, 0.001)
    plt.plot(r, Vloc_real(a, r), label=f'GTH for {a.atom}')
    plt.plot(r, Coulomb_real(a, r), label='Coulomb')
    plt.axvline(rloc, label='$r_{loc}$', c='grey', ls='--')
    plt.legend()
    plt.show()
    return


def plot_n(a):
    '''Plot the electronic density in real-space.'''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mask = a.r[:,2] == a.a / 2
    ax.plot_trisurf(a.r[:,0][mask], a.r[:,1][mask], a.n[mask])
    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mask = a.r[:,1] == a.a / 2
    ax.plot_trisurf(a.r[:,2][mask], a.r[:,0][mask], a.n[mask])
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    mask = a.r[:,0] == a.a / 2
    ax.plot_trisurf(a.r[:,1][mask], a.r[:,2][mask], a.n[mask])
    plt.show()
    return
