"""Store trajectories and solve results in versioned NPZ and JSON files.

NPZ files contain arrays without pickle data. JSON files contain metadata.
Saved trajectories preserve their native interpolation. Before saving a
callable source, convert it with ``TrajectoryData.materialize``. Files do not
contain callbacks or live solver objects.
"""
from collections.abc import Mapping
from dataclasses import fields
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
from trajopt.trajectories import TrajectoryData, TrajectoryField, TrajectorySet

FORMAT = 'trajopt-data'
VERSION = 3


def _classes():
    from trajopt.outputs.results import SolveResult, AttemptRecord
    from trajopt.formulations.specs import ProblemSpec, SegmentSpec, CouplingSpec
    from trajopt.methods.specs import MethodSpec
    return {kind.__name__: kind for kind in (
        SolveResult, AttemptRecord, ProblemSpec, SegmentSpec, CouplingSpec,
        MethodSpec, TrajectoryData, TrajectoryField)}


class _Encoder:
    def __init__(self):
        self.arrays = {}
        self.classes = _classes()

    def encode(self, value, location='root'):
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else {'type': 'float', 'value': str(value)}
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject:
                raise TypeError(f'{location}: object arrays are not portable')
            key = f'array_{len(self.arrays)}'
            self.arrays[key] = value
            return {'type': 'array', 'key': key}
        if isinstance(value, TrajectorySet):
            return {'type': 'TrajectorySet', 'segments': self.encode(dict(value), location+'.segments')}
        kind = type(value).__name__
        if kind in self.classes and type(value) is self.classes[kind]:
            return {'type': kind, 'fields': {
                item.name: self.encode(getattr(value, item.name), f'{location}.{item.name}')
                for item in fields(value) if item.init}}
        if isinstance(value, Mapping):
            return {'type': 'mapping', 'items': [
                [self.encode(key, location+'.key'), self.encode(item, f'{location}.{key}')]
                for key, item in value.items()]}
        if isinstance(value, (tuple, list)):
            return {'type': 'tuple' if isinstance(value, tuple) else 'list',
                    'items': [self.encode(item, f'{location}[{i}]') for i, item in enumerate(value)]}
        if isinstance(value, Path):
            return {'type': 'path', 'value': str(value)}
        if callable(value):
            raise TypeError(f'{location}: callable sources are not portable; materialize the trajectory '
                            'explicitly, or store a declarative function reference in configuration')
        raise TypeError(f'{location}: unsupported portable value {type(value).__name__}; '
                        'only owned data, not live numerical resources, can be saved')


class _Decoder:
    def __init__(self, arrays):
        self.arrays = arrays
        self.classes = _classes()

    def decode(self, value):
        if not isinstance(value, dict):
            return value
        kind = value.get('type')
        if kind == 'array':
            return self.arrays[value['key']].copy()
        if kind == 'mapping':
            return {self.decode(key): self.decode(item) for key, item in value['items']}
        if kind in ('tuple', 'list'):
            items = [self.decode(item) for item in value['items']]
            return tuple(items) if kind == 'tuple' else items
        if kind == 'float':
            if value['value'] not in ('inf', '-inf', 'nan'):
                raise ValueError('Invalid nonfinite float in metadata')
            return float(value['value'])
        if kind == 'path':
            return Path(value['value'])
        if kind == 'TrajectorySet':
            return TrajectorySet(self.decode(value['segments']))
        if kind in self.classes:
            expected = {item.name for item in fields(self.classes[kind]) if item.init}
            if set(value['fields']) != expected:
                raise ValueError(f'Invalid stored fields for {kind}')
            return self.classes[kind](**{key: self.decode(item) for key, item in value['fields'].items()})
        raise ValueError(f'Unsupported stored data type {kind!r}')


def _paths(path):
    path = Path(path)
    if path.suffix in ('.npz', '.json'):
        return path.with_suffix('.npz'), path.with_suffix('.json')
    return Path(str(path)+'.npz'), Path(str(path)+'.json')


def _save(value, path, kind):
    encoder = _Encoder()
    payload = encoder.encode(value)
    array_path, metadata_path = _paths(path)
    if not array_path.parent.is_dir():
        raise FileNotFoundError(f'Persistence directory does not exist: {array_path.parent}')
    temp_paths = []
    try:
        with tempfile.NamedTemporaryFile(dir=array_path.parent, suffix='.npz', delete=False) as handle:
            temporary_arrays = Path(handle.name)
            temp_paths.append(temporary_arrays)
            np.savez_compressed(handle, **encoder.arrays)
        digest = hashlib.sha256(temporary_arrays.read_bytes()).hexdigest()
        manifest = {'format': FORMAT, 'version': VERSION, 'kind': kind,
                    'arrays_sha256': digest, 'payload': payload}
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf8', dir=array_path.parent,
                                         suffix='.json', delete=False) as handle:
            temporary_metadata = Path(handle.name)
            temp_paths.append(temporary_metadata)
            json.dump(manifest, handle, allow_nan=False, indent=2)
            handle.write('\n')
        os.replace(temporary_arrays, array_path)
        os.replace(temporary_metadata, metadata_path)
    finally:
        for temporary in temp_paths:
            temporary.unlink(missing_ok=True)
    return array_path


def _load(path, kind):
    array_path, metadata_path = _paths(path)
    manifest = json.loads(metadata_path.read_text(encoding='utf8'))
    if manifest.get('format') != FORMAT or manifest.get('version') != VERSION:
        raise ValueError(f'Unsupported trajectory data format/version: '
                         f'{manifest.get("format")!r}/{manifest.get("version")!r}')
    if manifest.get('kind') != kind:
        raise ValueError(f'Expected saved {kind}, found {manifest.get("kind")!r}')
    if hashlib.sha256(array_path.read_bytes()).hexdigest() != manifest.get('arrays_sha256'):
        raise ValueError('Array file and metadata do not match (incomplete or modified snapshot)')
    with np.load(array_path, allow_pickle=False) as arrays:
        return _Decoder(arrays).decode(manifest['payload'])


def save_trajectory(trajectory, path):
    """Save a ``TrajectoryData`` or ``TrajectorySet`` and return the NPZ path."""
    if not isinstance(trajectory, (TrajectoryData, TrajectorySet)):
        raise TypeError('Save requires owned trajectory data; materialize callable sources explicitly')
    return _save(trajectory, path, 'trajectory')


def load_trajectory(path):
    """Load a trajectory from its base path, NPZ path, or JSON path."""
    value = _load(path, 'trajectory')
    if not isinstance(value, (TrajectoryData, TrajectorySet)):
        raise ValueError('Saved trajectory payload has the wrong type')
    return value


def save_result(result, path):
    """Save a ``SolveResult`` with its physical trajectories and numerical metadata."""
    from trajopt.outputs.results import SolveResult
    if not isinstance(result, SolveResult):
        raise TypeError('save_result requires a SolveResult')
    return _save(result, path, 'result')


def load_result(path):
    """Load a ``SolveResult`` with immutable arrays without constructing a solver."""
    from trajopt.outputs.results import SolveResult
    value = _load(path, 'result')
    if not isinstance(value, SolveResult):
        raise ValueError('Saved result payload has the wrong type')
    return value
