"""Convert between physical trajectories and solver arrays."""
import hashlib
import json

import numpy as np

from trajopt.utils._snapshot import freeze
from trajopt.trajectories import TrajectoryData, TrajectorySet


def segment_layout(phase):
    config = phase.config
    layout = {}
    for kind in ('state', 'control'):
        groups = phase.index_map.components[kind]
        def channel(key, indices):
            cfg = config[kind][key]
            record = (key, tuple(int(i) for i in indices), cfg.get('units', cfg.get('unit')))
            frame = cfg.get('reference_frame', cfg.get('frame'))
            return record if frame is None else (*record, frame)
        layout[kind] = (
            getattr(phase.index_map.n, kind),
            tuple(sorted(channel(key, indices) for key, indices in groups.items())),
        )
    time = config.get('time', {})
    layout['time_unit'] = time.get('units', time.get('unit'))
    frame = config.get('reference_frame', config.get('frame'))
    if frame is not None:
        layout['reference_frame'] = frame
    return freeze(layout)


def auxiliary_metadata(phase):
    def plain(value):
        if hasattr(value, 'items'):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [plain(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if callable(value):
            return f'{value.__module__}:{value.__qualname__}'
        return value
    config = phase.config
    definitions = dict(
        constraints={name: c for name, c in config.constraints.items()
                     if c['type'] == 'ctcs_nonconvex_inequality'},
        costs={name: c for name, c in config.get('costs', {}).items()
               if c['type'] == 'nonconvex_running'},
        params=phase.params, state=config.state, control=config.control,
        time=config.time, fcns=config.fcns,
    )
    digest = hashlib.sha256(json.dumps(plain(definitions), sort_keys=True).encode()).hexdigest()
    return freeze(dict(definition=digest, units='destination_nondimensional',
                       ctcs=phase.index_map.n.ctcs, running_cost=phase.index_map.n.running_cost))


def export_trajectory(segment, *, data=None, include_aux=False):
    from trajopt.methods.initial_guess.preparation import dimensional_samples
    data = segment.current_iter_data if data is None else data
    z = data.get('z_opt', data.get('z'))
    nu = data.get('nu_opt', data.get('nu'))
    values = dimensional_samples(segment.problem, z, nu)
    mesh = segment.mesh
    return TrajectoryData.from_samples(
        t=values.t, x=values.x, u=values.u, s=values.s,
        aux=values.aux if include_aux else None, tau=mesh.tau,
        interpolation=({'t': 'polynomial', 'x': 'polynomial', 'u': 'linear',
                        's': 'constant', 'aux': 'polynomial'}
                       if mesh.discretize == 'ps' else 'linear'),
        segments=mesh.segments, segment_id=segment.name,
        layout=segment_layout(segment.problem),
        aux_metadata=auxiliary_metadata(segment.problem) if include_aux else {},
    )


def export_set(segments, *, initial=False, include_aux=False):
    return TrajectorySet({name: export_trajectory(segment,
        data=segment.initial_guess if initial else None, include_aux=include_aux)
        for name, segment in segments.items()})
