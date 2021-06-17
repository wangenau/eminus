#!/usr/bin/env python3
'''
Get constants from GTH files.
'''
import numpy as np
import plainedft
from glob import glob
from os.path import basename

PSP_PATH = plainedft.__path__[0] + '/pade_gth/'


def read_GTH(system, charge=None):
    '''Read GTH files for a given system.'''
    if charge is not None:
        f_psp = PSP_PATH + str(system) + '-q' + str(charge) + '.gth'
    else:
        files = glob(PSP_PATH + str(system) + '-*')
        try:
            f_psp = files[0]
        except IndexError:
            print(f'ERROR: There is no pseudopotential for \'{system}\'')
        if len(files) > 1:
            print(f'INFO: Multiple pseudopotentials found for \'{system}\'. '
                  f'Continue with \'{basename(f_psp)}\'.')

    try:
        with open(f_psp, 'r') as fh:
            psp = {}
            C = np.zeros(4)
            rc = np.zeros(4)
            h = np.zeros([4, 3, 3])
            Nproj_l = np.zeros(4, dtype=int)
            symbol = fh.readline().split()[0]
            psp['symbol'] = symbol
            N_all = fh.readline().split()
            N_s, N_p, N_d, N_f = int(N_all[0]), int(N_all[1]), int(N_all[2]), int(N_all[3])
            Zval = N_s + N_p + N_d + N_f
            psp['Zval'] = Zval
            rlocal, n_c_local = fh.readline().split()
            rlocal, n_c_local = float(rlocal), int(n_c_local)
            psp['rlocal'] = rlocal
            psp['n_c_local'] = n_c_local
            for i, val in enumerate(fh.readline().split()):
                C[i] = float(val)
            psp['C'] = C
            lmax = int(fh.readline().split()[0])
            psp['lmax'] = lmax
            for iiter in range(lmax):
                rc[iiter], Nproj_l[iiter] = [float(i) for i in fh.readline().split()]
                for jiter in range(int(Nproj_l[iiter])):
                    tmp = fh.readline().split()
                    iread = 0
                    for kiter in range(jiter, int(Nproj_l[iiter])):
                        h[iiter, jiter, kiter] = float(tmp[iread])
                        iread += 1
            for k in range(3):
                for i in range(2):
                    for j in range(i + 1, 2):
                        h[k, j, i] = h[k, i, j]
            psp['rc'] = rc
            psp['Nproj_l'] = Nproj_l
            psp['h'] = h
    except FileNotFoundError:
        print(f'ERROR: There is no pseudopotential for \'{basename(f_psp)}\'')
    return psp, Zval
