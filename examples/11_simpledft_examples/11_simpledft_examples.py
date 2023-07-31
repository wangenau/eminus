import time

from eminus import Atoms, SCF


def calculate(atoms):
    start = time.perf_counter()
    atoms.s = 60
    etot = SCF(atoms, xc='lda,chachiyo', pot='coulomb', guess='pseudo',
               etol=1e-6, opt={'sd': 1001}).run(betat=1e-5)
    print(f'Etot({atoms.atom}) = {etot:.6f} Eh')
    print(f' {time.perf_counter() - start:.6f} seconds')


H = Atoms(['H'], [[0, 0, 0]], 16, 16, unrestricted=False, verbose='warning')
calculate(H)
# # `Output:  Etot(['H']) = -0.438418 Eh`

He = Atoms(['He'], [[0, 0, 0]], 16, 16, unrestricted=False, verbose='warning')
calculate(He)
# # `Output:  Etot(['He']) = -2.632035 Eh`

H2 = Atoms(['H', 'H'], [[0, 0, 0], [1.4, 0, 0]], 16, 16, unrestricted=False, verbose='warning')
calculate(H2)
# # `Output:  Etot(['H', 'H']) = -1.113968 Eh`
