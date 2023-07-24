#!/usr/bin/env python3
"""JSON file handling."""
import base64
import copy
import json

import numpy as np


def read_json(filename):
    """Load objects from a JSON file.

    Args:
        filename (str): json input file path/name.

    Returns:
        Class object.
    """
    import eminus

    def custom_object_hook(dct):
        """Custom JSON object hook to create eminus classes after deserialization."""
        # ndarrays are base64 encoded, decode and recreate
        if isinstance(dct, dict) and '__ndarray__' in dct:
            data = base64.b64decode(dct['__ndarray__'])
            return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])

        # Create a simple eminus objects and set all attributes afterwards
        # Explicitly call objects with verbosity since the logger is created at instantiation
        if isinstance(dct, dict) and 'atom' in dct:
            atoms = eminus.Atoms(dct['atom'], dct['X'], verbose=dct['_verbose'])
            atoms._set_operators()
            for attr in dct:
                if attr == 'log':
                    continue
                setattr(atoms, attr, copy.deepcopy(dct[attr]))
            # The tuple type is not preserved when serializing, manually cast the only important one
            if isinstance(atoms.active, list):
                atoms.active = tuple(atoms.active)
            return atoms
        if isinstance(dct, dict) and '_atoms' in dct:
            scf = eminus.SCF(dct['_atoms'], verbose=dct['_verbose'])
            for attr in dct:
                if attr == 'log':
                    continue
                setattr(scf, attr, copy.deepcopy(dct[attr]))
            return scf
        if isinstance(dct, dict) and 'Ekin' in dct:
            energies = eminus.energies.Energy()
            for attr in dct:
                setattr(energies, attr, dct[attr])
            return energies
        if isinstance(dct, dict) and 'NbetaNL' in dct:
            gth = eminus.gth.GTH()
            for attr in dct:
                setattr(gth, attr, dct[attr])
            return gth
        return dct

    if not filename.endswith('.json'):
        filename += '.json'

    with open(filename, 'r') as fh:
        return json.load(fh, object_hook=custom_object_hook)


def write_json(obj, filename):
    """Save objects in a JSON file.

    Args:
        obj: Class object.
        filename (str): json output file path/name.
    """
    import eminus

    class CustomEncoder(json.JSONEncoder):
        """Custom JSON encoder class to serialize eminus classes."""
        def default(self, obj):
            # ndarrays are not json serializable, encode them as base64 to save them
            if isinstance(obj, np.ndarray):
                data = base64.b64encode(obj.copy(order='C')).decode('utf-8')
                return {'__ndarray__': data, 'dtype': str(obj.dtype), 'shape': obj.shape}

            # If obj is a Atoms or SCF class dump them as a dictionary
            if isinstance(obj, (eminus.Atoms, eminus.SCF, eminus.energies.Energy, eminus.gth.GTH)):
                # Only dumping the dict would result in a string, so do one dump and one load
                data = json.dumps(obj.__dict__, cls=CustomEncoder)
                return dict(json.loads(data))
            # The logger class is not serializable, just ignore it
            if isinstance(obj, eminus.logger.CustomLogger):
                return None
            return json.JSONEncoder.default(self, obj)

    if not filename.endswith('.json'):
        filename += '.json'

    with open(filename, 'w') as fp:
        json.dump(obj, fp, cls=CustomEncoder)
