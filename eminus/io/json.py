# SPDX-FileCopyrightText: 2023 The eminus developers
# SPDX-License-Identifier: Apache-2.0
"""JSON file handling."""

import base64
import copy
import json

import numpy as np


def _custom_object_hook(dct):  # noqa: PLR0911
    """Custom JSON object hook to create eminus classes after deserialization."""
    import eminus

    def set_attrs(obj, dct):
        """Set attributes of an object using a given dictionary."""
        for attr in dct:
            if attr == "_log":
                continue
            setattr(obj, attr, copy.deepcopy(dct[attr]))
        return obj

    # ndarrays are base64 encoded, decode and recreate
    if isinstance(dct, dict) and "__ndarray__" in dct:
        data = base64.b64decode(dct["__ndarray__"])
        return np.frombuffer(data, dct["dtype"]).reshape(dct["shape"])

    # Create simple eminus objects and set all attributes afterwards
    # Explicitly call objects with verbosity since the logger is created at instantiation

    # Atoms objects
    if isinstance(dct, dict) and "_atom" in dct:
        atoms = eminus.Atoms(dct["_atom"], dct["_pos"], verbose=dct["_verbose"])
        atoms = set_attrs(atoms, dct)
        # The tuple type is not preserved when serializing, manually cast the only important one
        if not isinstance(atoms._active, tuple):
            atoms._active = [tuple(i) for i in atoms._active]
        return atoms
    # SCF objects
    if isinstance(dct, dict) and "_atoms" in dct:
        scf = eminus.SCF(dct["_atoms"], verbose=dct["_verbose"])
        return set_attrs(scf, dct)
    # Energy objects
    if isinstance(dct, dict) and "Ekin" in dct:
        energies = eminus.energies.Energy()
        return set_attrs(energies, dct)
    # GTH objects
    if isinstance(dct, dict) and "NbetaNL" in dct:
        gth = eminus.gth.GTH()
        return set_attrs(gth, dct)
    # Occupations objects
    if isinstance(dct, dict) and "_Nelec" in dct:
        occ = eminus.occupations.Occupations()
        return set_attrs(occ, dct)
    # KPoints objects
    if isinstance(dct, dict) and "_kmesh" in dct:
        kpts = eminus.kpoints.KPoints(dct["lattice"])
        return set_attrs(kpts, dct)
    return dct


def read_json(filename):
    """Load objects from a JSON file.

    Args:
        filename: JSON input file path/name.

    Returns:
        Class object.
    """
    if not filename.endswith(".json"):
        filename += ".json"

    with open(filename, encoding="utf-8") as fh:
        return json.load(fh, object_hook=_custom_object_hook)


def write_json(obj, filename):
    """Save objects in a JSON file.

    Args:
        obj: Class object.
        filename: JSON output file path/name.
    """
    import eminus

    class _CustomEncoder(json.JSONEncoder):
        """Custom JSON encoder class to serialize eminus classes."""

        def default(self, obj):
            """Overwrite the default function to handle eminus objects."""
            # ndarrays are not JSON serializable, encode them as base64 to save them
            if isinstance(obj, np.ndarray):
                data = base64.b64encode(obj.copy(order="C")).decode("utf-8")
                return {"__ndarray__": data, "dtype": str(obj.dtype), "shape": obj.shape}

            # If obj is an eminus class dump them as a dictionary
            if isinstance(
                obj,
                (
                    eminus.Atoms,
                    eminus.SCF,
                    eminus.energies.Energy,
                    eminus.gth.GTH,
                    eminus.kpoints.KPoints,
                    eminus.occupations.Occupations,
                ),
            ):
                # Only dumping the dict would result in a string, so do one dump and one load
                data = json.dumps(obj.__dict__, cls=_CustomEncoder)
                return dict(json.loads(data))
            # The logger class is not serializable, just ignore it
            if isinstance(obj, eminus.logger.CustomLogger):
                return None
            return json.JSONEncoder.default(self, obj)

    if not filename.endswith(".json"):
        filename += ".json"

    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(obj, fp, cls=_CustomEncoder)
