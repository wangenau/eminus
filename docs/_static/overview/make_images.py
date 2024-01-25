#!/usr/bin/env python3
"""Create images for the overview documentation page."""
from eminus import Cell, SCF
from eminus.extras import plot_bandstructure, view

cell = Cell('Si', 'diamond', ecut=10, a=10.2631, bands=8)
fig = view(cell)
fig.update_layout(scene_camera={'eye': {'x': 2, 'y': 0, 'z': 1}},
                  margin={'l': 0, 'r': 0, 'b': 0, 't': 0})
fig.write_image('cell.png')
scf = SCF(cell)
scf.run()
scf.kpts.path = 'LGXU,KG'
scf.kpts.Nk = 25
fig = view(scf.kpts.build())
fig.update_layout(margin={'l': 20, 'r': 0, 'b': 20, 't': 0})
fig.write_image('bz.png')
scf.converge_bands()
fig = plot_bandstructure(scf)
margin = {'l': 60, 'r': 20, 'b': 60, 't': 40}
fig.update_layout(margin=margin)
fig.write_image('band_structure.png')
