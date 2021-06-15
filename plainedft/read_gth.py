#!/usr/bin/env python3
'''
Get constants from GTH files.
'''
import numpy as np
import plainedft

PSP_PATH = plainedft.__path__[0] + '/pade_gth/'


# TODO: Rewrite???
def read_GTH(f_file):
    '''Read HGH/GTH files.'''
    f_file = PSP_PATH + f_file
    psp = {}
    C = np.zeros(4)
    rc = np.zeros(4)
    h = np.zeros([4, 3, 3])
    Nproj_l = np.zeros(4, dtype=int)
    try:
        f = open(f_file, 'r')
    except FileNotFoundError:
        print('ERROR: Can\'t find file %s' % f_file)

    symbol = f.readline().split()[0]
    psp['symbol'] = symbol
    N_all = f.readline().split()
    N_s, N_p, N_d, N_f = int(N_all[0]), int(N_all[1]), int(N_all[2]), int(N_all[3])
    Zval = N_s + N_p + N_d + N_f
    psp['Zval'] = Zval
    rlocal, n_c_local = f.readline().split()
    rlocal, n_c_local = float(rlocal), int(n_c_local)
    psp['rlocal'] = rlocal
    psp['n_c_local'] = n_c_local
    for i, val in enumerate(f.readline().split()):
        C[i] = float(val)
    psp['C'] = C
    lmax = int(f.readline().split()[0])
    psp['lmax'] = lmax
    for iiter in range(lmax):
        rc[iiter], Nproj_l[iiter] = [float(i) for i in f.readline().split()]
        for jiter in range(int(Nproj_l[iiter])):
            tmp = f.readline().split()
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
    return psp
