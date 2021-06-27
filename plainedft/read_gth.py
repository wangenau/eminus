#!/usr/bin/env python3
'''
Get constants from GTH files.
'''
import numpy as np
from plainedft import __path__
from glob import glob
from os.path import basename

PSP_PATH = __path__[0] + '/pade_gth/'


def read_gth(system, charge=None):
    '''Read GTH files for a given system.'''
    if charge is not None:
        f_psp = PSP_PATH + str(system) + '-q' + str(charge) + '.gth'
    else:
        files = glob(PSP_PATH + str(system) + '-*')
        try:
            f_psp = files[0]
        except IndexError:
            print(f'ERROR: There is no GTH pseudopotential for \'{system}\'')
        if len(files) > 1:
            print(f'INFO: Multiple pseudopotentials found for \'{system}\'. '
                  f'Continue with \'{basename(f_psp)}\'.')

    psp = {}
    C = np.zeros(4)
    rc = np.zeros(4)
    Nproj_l = np.zeros(4, dtype=int)
    h = np.zeros([4, 3, 3])
    try:
        with open(f_psp, 'r') as fh:
            psp['symbol'] = fh.readline().split()[0]
            N_all = fh.readline().split()
            N_s, N_p, N_d, N_f = int(N_all[0]), int(N_all[1]), int(N_all[2]), int(N_all[3])
            psp['Zval'] = N_s + N_p + N_d + N_f
            rlocal, n_c_local = fh.readline().split()
            psp['rlocal'] = float(rlocal)
            psp['n_c_local'] = int(n_c_local)
            for i, val in enumerate(fh.readline().split()):
                C[i] = float(val)
            psp['C'] = C
            lmax = int(fh.readline().split()[0])
            psp['lmax'] = lmax
            for iiter in range(lmax):
                rc[iiter], Nproj_l[iiter] = [float(i) for i in fh.readline().split()]
                for jiter in range(Nproj_l[iiter]):
                    tmp = fh.readline().split()
                    iread = 0
                    for kiter in range(jiter, Nproj_l[iiter]):
                        h[iiter, jiter, kiter] = float(tmp[iread])
                        iread += 1
            psp['rc'] = rc
            psp['Nproj_l'] = Nproj_l
            for k in range(3):
                for i in range(2):
                    for j in range(i + 1, 2):
                        h[k, j, i] = h[k, i, j]
            psp['h'] = h
    except FileNotFoundError:
        print(f'ERROR: There is no GTH pseudopotential for \'{basename(f_psp)}\'')
    return psp
