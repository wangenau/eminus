import time

from eminus import Atoms, SCF


def calculate(atoms):
    start = time.perf_counter()
    etot = SCF(atoms, xc='lda,chachiyo', pot='coulomb', guess='pseudo', etol=1e-6,
               min={'sd': 1001}).run(betat=1e-5)
    print('Etot({}) = {:.6f} Eh'.format(atoms.atom, etot))
    print(' {:.6f} seconds'.format(time.perf_counter() - start))


H = Atoms(['H'], [[0, 0, 0]], 16, 16, [1], [60, 60, 60], f=[1],
          Nspin=1, verbose='warning')
calculate(H)
# # `Output:  Etot(['H']) = -0.438413 Eh`

He = Atoms(['He'], [[0, 0, 0]], 16, 16, [2], [60, 60, 60], f=[2],
           Nspin=1, verbose='warning')
calculate(He)
# # `Output:  Etot(['He']) = -2.632034 Eh`

H2 = Atoms(['H', 'H'], [[0, 0, 0], [1.4, 0, 0]], 16, 16, [1, 1], [60, 60, 60], f=[2],
           Nspin=1, verbose='warning')
calculate(H2)
# # `Output:  Etot(['H', 'H']) = -1.113968 Eh`
