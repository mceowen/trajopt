"""Physical trajectory data for initial guesses, iterates, and results.

Time and model channels use their declared physical units. Numerical
auxiliaries have separate metadata. ``without_aux`` removes them. Each field
stores its node grid and interpolation intervals. Sampling needs no solver,
external mesh, or callback.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import numpy as np
from scipy.optimize import brentq

from trajopt.trajectories.interpolation import TrajectoryField, _nodes, _queries
from trajopt.utils._snapshot import freeze, readonly_array


@dataclass(frozen=True)
class TrajectorySamples:
    """Trajectory samples in physical units.

    ``s`` is ``dt/dtau``. ``aux`` uses its separately declared units.
    """
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    s: np.ndarray | None = None
    aux: np.ndarray | None = None

    def __post_init__(self):
        count = len(np.atleast_1d(self.t))
        for name in ('t', 'x', 'u', 's', 'aux'):
            value = getattr(self, name)
            if value is None:
                if name in ('t', 'x', 'u'):
                    raise ValueError(f'Trajectory samples require {name}')
                continue
            array = np.asarray(value, dtype=float)
            rank = 1 if name in ('t', 's') else 2
            if array.ndim != rank or len(array) != count:
                raise ValueError(f'Trajectory sample {name} must have rank-{rank} shape with {count} rows')
            if not np.isfinite(array).all():
                raise ValueError(f'Trajectory sample {name} contains nonfinite values')
            object.__setattr__(self, name, readonly_array(array))


@runtime_checkable
class TrajectorySource(Protocol):
    @property
    def native_tau(self) -> np.ndarray | None: ...

    def sample(self, tau) -> TrajectorySamples: ...


@dataclass(frozen=True)
class TrajectoryFunction:
    """A trajectory source defined by a callable.

    ``evaluate(tau)`` returns ``TrajectorySamples`` in physical units. If
    supplied, ``native_tau`` adds the source nodes to the grid during import.
    Before saving, convert the source with ``TrajectoryData.materialize``.
    For array data, use ``TrajectoryData`` directly.
    """
    evaluate: Callable
    native_tau: np.ndarray | None = None

    def __post_init__(self):
        if not callable(self.evaluate):
            raise TypeError('Trajectory evaluator must be callable')
        if self.native_tau is not None:
            object.__setattr__(self, 'native_tau', readonly_array(_nodes(self.native_tau)))

    def sample(self, tau):
        q = _queries(tau)
        values = self.evaluate(q.copy())
        if not isinstance(values, TrajectorySamples) or len(values.t) != len(q):
            raise TypeError('Trajectory evaluator must return TrajectorySamples with one row per query')
        return values


def _validate_layout(layout, fields):
    if not isinstance(layout, Mapping):
        raise TypeError('Trajectory layout must be a mapping')
    for kind, key in (('state', 'x'), ('control', 'u')):
        channels = layout.get(kind)
        if channels is None:
            continue
        dimension = (int(channels[0]) if len(channels) == 2
                     and isinstance(channels[0], (int, np.integer)) else len(channels))
        if dimension != fields[key].values[0].shape[1]:
            raise ValueError(f'Layout {kind} channels must match the trajectory dimension')


def _join_interval_arrays(arrays):
    """Join stored rows without interpolation. Keep all boundary values."""
    return arrays[0] if len(arrays) == 1 else readonly_array(np.concatenate(arrays, axis=0))


@dataclass(frozen=True)
class TrajectoryData:
    """Immutable physical trajectory fields for one named segment."""
    fields: Mapping
    segment_id: str = 'main'
    layout: Mapping = field(default_factory=dict)
    aux_metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.segment_id, str) or not self.segment_id:
            raise ValueError('segment_id must be a nonempty string')
        if not {'t', 'x', 'u'}.issubset(self.fields):
            raise ValueError('Trajectory fields require t, x and u')
        unknown = set(self.fields)-{'t', 'x', 'u', 's', 'aux'}
        if unknown:
            raise ValueError(f'Unknown trajectory fields: {sorted(unknown)}')
        for name, value in self.fields.items():
            if not isinstance(value, TrajectoryField):
                raise TypeError(f'Trajectory field {name} must be a TrajectoryField')
            rank = 1 if name in ('t', 's') else 2
            if value.values[0].ndim != rank:
                raise ValueError(f'Trajectory field {name} must have a rank-{rank} shape')
        times = self.fields['t']
        for values in times.values:
            if np.any(np.diff(values) <= 0):
                raise ValueError('Trajectory times must strictly increase')
        for previous, following in zip(times.values, times.values[1:]):
            if not np.isclose(previous[-1], following[0], rtol=0, atol=1e-10):
                raise ValueError('The physical time map must be continuous across intervals')
        dilation = self.fields.get('s')
        if dilation is not None and any(np.any(values <= 0) for values in dilation.values):
            raise ValueError('Trajectory time dilation must be positive')
        _validate_layout(self.layout, self.fields)
        object.__setattr__(self, 'fields', freeze(dict(self.fields)))
        object.__setattr__(self, 'layout', freeze(self.layout))
        object.__setattr__(self, 'aux_metadata', freeze(self.aux_metadata))

    @classmethod
    def from_samples(cls, *, t, x, u, tau=None, s=None, aux=None,
                     interpolation='linear', segments=1, segment_id='main',
                     layout=None, aux_metadata=None, state_tau=None,
                     control_tau=None, dilation_tau=None, aux_tau=None):
        """Copy samples, retaining native interpolation and each field's grid.

        Polynomial ``segments`` is an interval count (equal order), a
        sequence of local degrees, or a mapping from field name to either.
        ``interpolation`` may also map each field to its own interpolation
        kind. Constant fields use one value per segment and closing-knot sides.
        More general discontinuous data uses ``TrajectoryField.from_intervals``
        and the ordinary ``TrajectoryData(fields=...)`` constructor.
        """
        t = np.asarray(t, dtype=float)
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError('Trajectory times must be finite and strictly increasing')
        tau = _nodes((t-t[0])/(t[-1]-t[0]) if tau is None else tau)
        field_nodes = {'t': tau, 'x': state_tau, 'u': control_tau,
                       's': dilation_tau, 'aux': aux_tau}
        result = {}
        for name, raw in (('t', t), ('x', x), ('u', u), ('s', s), ('aux', aux)):
            if raw is None:
                if name in ('t', 'x', 'u'):
                    raise ValueError(f'Trajectory {name} is required')
                continue
            a = np.asarray(raw, dtype=float)
            if name == 's' and a.ndim == 2 and a.shape[1] == 1:
                a = a[:, 0]
            nodes = tau if field_nodes[name] is None else field_nodes[name]
            kind = interpolation.get(name, 'linear') if isinstance(interpolation, Mapping) else interpolation
            local_segments = segments.get(name, 1) if isinstance(segments, Mapping) else segments
            if isinstance(interpolation, Mapping) and kind == 'linear':
                local_segments = 1
            if name == 's' and (not np.isfinite(a).all() or np.any(a <= 0)):
                raise ValueError('Trajectory time dilation must be positive and finite')
            result[name] = TrajectoryField.from_samples(nodes, a, interpolation=kind,
                                                        segments=local_segments)
        return cls(result, segment_id, layout or {}, aux_metadata or {})

    @classmethod
    def from_file(cls, path, *, state_columns, control_columns, time_column=0,
                  delimiter=None, skiprows=0, tau_column=None, dilation_column=None,
                  interpolation='linear', segments=1):
        """Load dimensional samples using zero-based columns in model order.

        Defaults: commas for .csv, whitespace otherwise. Use skiprows for headers.
        Optional tau/dilation columns preserve non-affine time maps.
        """
        path = Path(path)
        if delimiter is None and path.suffix.lower() == '.csv':
            delimiter = ','
        data = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows, ndmin=2)

        def columns(value, name, *, scalar=False):
            indices = np.asarray(value)
            if scalar:
                if indices.ndim != 0:
                    raise ValueError(f"{name} must be a single column index")
                indices = indices.reshape(1)
            if (indices.ndim != 1 or not len(indices)
                    or not np.issubdtype(indices.dtype, np.integer)
                    or np.any(indices < 0) or np.any(indices >= data.shape[1])):
                raise ValueError(f"{name} must contain integer column indices in [0, {data.shape[1]})")
            return indices

        t_idx = columns(time_column, 'time_column', scalar=True)
        x_idx = columns(state_columns, 'state_columns')
        u_idx = columns(control_columns, 'control_columns')
        tau_idx = None if tau_column is None else columns(tau_column, 'tau_column', scalar=True)
        s_idx = None if dilation_column is None else columns(dilation_column, 'dilation_column', scalar=True)
        selected = np.concatenate([idx for idx in (t_idx, x_idx, u_idx, tau_idx, s_idx) if idx is not None])
        if len(np.unique(selected)) != len(selected):
            raise ValueError('Time, state, control, tau and dilation columns must be distinct')
        return cls.from_samples(
            t=data[:, t_idx[0]], x=data[:, x_idx], u=data[:, u_idx],
            tau=None if tau_idx is None else data[:, tau_idx[0]],
            s=None if s_idx is None else data[:, s_idx[0]],
            interpolation=interpolation, segments=segments,
        )

    @classmethod
    def materialize(cls, source, *, tau, interpolation='linear', segments=1,
                    segment_id='main', layout=None, aux_metadata=None):
        """Sample a trajectory source on the supplied grid and store the arrays.

        Choose the grid and interpolation to meet the required approximation
        accuracy. Saving a trajectory does not sample callbacks automatically.
        """
        values = source.sample(_nodes(tau))
        return cls.from_samples(t=values.t, x=values.x, u=values.u, s=values.s,
                                aux=values.aux, tau=tau, interpolation=interpolation,
                                segments=segments, segment_id=segment_id,
                                layout=layout, aux_metadata=aux_metadata)

    @property
    def native_tau(self):
        return readonly_array(np.unique(np.concatenate([value.native_tau for value in self.fields.values()])))

    @property
    def samples(self):
        return self.sample(self.native_tau)

    @property
    def t(self):
        """Return stored physical times at ``t_tau``.

        Use ``x_t`` for state rows and ``u_t`` for control rows.
        """
        return _join_interval_arrays(self.fields['t'].values)

    @property
    def t_tau(self):
        """Return normalized times for ``t`` rows. Include repeated interval boundaries."""
        return _join_interval_arrays(self.fields['t'].nodes)

    @property
    def x(self):
        """Return read-only stored states at ``x_tau``. Keep both interval endpoints."""
        return _join_interval_arrays(self.fields['x'].values)

    @property
    def u(self):
        """Return read-only stored controls at ``u_tau``. Keep both interval endpoints."""
        return _join_interval_arrays(self.fields['u'].values)

    @property
    def x_tau(self):
        """Return normalized times for ``x`` rows. Include repeated interval boundaries."""
        return _join_interval_arrays(self.fields['x'].nodes)

    @property
    def u_tau(self):
        """Return normalized times for ``u`` rows. Include repeated interval boundaries."""
        return _join_interval_arrays(self.fields['u'].nodes)

    @property
    def x_t(self):
        """Return physical times for ``x`` rows. Interpolate only the stored time map."""
        return readonly_array(self.fields['t'].sample(self.x_tau))

    @property
    def u_t(self):
        """Return physical times for ``u`` rows. Interpolate only the stored time map."""
        return readonly_array(self.fields['t'].sample(self.u_tau))

    def _uniform_physical_time_queries(self, num):
        """Invert the stored time map, preserving exact native-knot queries."""
        time = self.fields['t']
        if time.interpolation == 'polynomial':
            # Increasing nodal times alone do not guarantee an invertible polynomial.
            # Check its derivative at endpoints and all interior extrema.
            for nodes, values in zip(time.nodes, time.values):
                polynomial = np.polynomial.Chebyshev.fit(
                    nodes, values - values[0], len(nodes) - 1,
                    domain=[nodes[0], nodes[-1]])
                derivative = polynomial.deriv()
                roots = derivative.deriv().roots()
                critical = roots.real[np.abs(roots.imag) < 1e-10]
                critical = critical[(critical > nodes[0]) & (critical < nodes[-1])]
                slopes = derivative(np.r_[nodes[0], critical, nodes[-1]])
                tolerance = 64 * np.finfo(float).eps * max(1., np.max(np.abs(slopes)))
                if not np.isfinite(slopes).all() or np.min(slopes) < -tolerance:
                    raise ValueError('Physical-time sampling requires an increasing time map')

        nodes = time.native_tau
        times = time.sample(nodes)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError('Physical-time sampling requires an increasing time map')
        targets = np.linspace(times[0], times[-1], num)
        if time.interpolation == 'linear':
            return np.interp(targets, times, nodes)

        queries = np.empty(num)
        queries[0], queries[-1] = 0., 1.
        for i, target in enumerate(targets[1:-1], start=1):
            right = np.searchsorted(times, target)
            if times[right] == target:
                queries[i] = nodes[right]
            else:
                queries[i] = brentq(lambda q: time.sample([q])[0] - target,
                                   nodes[right - 1], nodes[right], xtol=1e-14)
        return queries

    def sample(self, tau=None, *, num=None, spacing='tau', side=None):
        """Sample explicit normalized times, or a uniform grid including both endpoints.

        Use ``sample(num=101, spacing='tau')`` for uniform normalized time,
        or ``spacing='t'`` for uniform physical time. States and controls share
        the returned physical times. Explicit ``tau`` cannot be combined with
        ``num`` or physical-time spacing. ``side`` selects interval-boundary values.
        """
        if spacing not in ('tau', 't'):
            raise ValueError("spacing must be 'tau' or 't'")
        if tau is not None:
            if num is not None or spacing != 'tau':
                raise ValueError('Use explicit tau or num with spacing, not both')
            q = _queries(tau)
        else:
            if (isinstance(num, (bool, np.bool_)) or not isinstance(num, (int, np.integer))
                    or num < 2):
                raise ValueError('num must be an integer of at least 2 when tau is omitted')
            q = (np.linspace(0., 1., num) if spacing == 'tau'
                 else self._uniform_physical_time_queries(num))
        values = {name: value.sample(q, side=side) for name, value in self.fields.items()}
        if any(not np.isfinite(value).all() for value in values.values()):
            raise ValueError('Trajectory interpolation produced nonfinite values')
        if 's' in values and np.any(values['s'] <= 0):
            raise ValueError('Trajectory sampled time dilation must be positive')
        if len(q) > 1:
            order = np.argsort(q)
            distinct = np.diff(q[order]) > 0
            time_steps = np.diff(values['t'][order])
            tolerance = 16*np.finfo(float).eps*max(1., np.max(np.abs(values['t'])))
            if np.any(time_steps[distinct] < -tolerance):
                raise ValueError('Trajectory sampled physical time must increase with tau')
        return TrajectorySamples(**values)

    def without_aux(self):
        return TrajectoryData({name: value for name, value in self.fields.items() if name != 'aux'},
                              self.segment_id, self.layout)


class TrajectorySet(Mapping):
    """An ordered, immutable mapping of segments with separate channel dimensions."""
    __slots__ = ('_segments',)

    def __init__(self, segments):
        if isinstance(segments, TrajectoryData):
            segments = {segments.segment_id: segments}
        if not isinstance(segments, Mapping) or not segments:
            raise ValueError('TrajectorySet requires a nonempty mapping of named segments')
        for name, segment in segments.items():
            if not isinstance(segment, TrajectoryData):
                raise TypeError('TrajectorySet values must be owned TrajectoryData')
            if name != segment.segment_id:
                raise ValueError(f'Segment key {name!r} differs from its identity {segment.segment_id!r}')
        object.__setattr__(self, '_segments', freeze(segments))

    def __setattr__(self, name, value):
        raise TypeError('TrajectorySet is immutable')

    def __getitem__(self, key):
        return self._segments[key]

    def __iter__(self):
        return iter(self._segments)

    def __len__(self):
        return len(self._segments)

    @property
    def single(self):
        if len(self) != 1:
            raise ValueError('Select a named segment from a multisegment trajectory')
        return next(iter(self.values()))

    @property
    def native_tau(self):
        return self.single.native_tau

    @property
    def t(self):
        """Return the only segment's stored physical times at ``t_tau``."""
        return self.single.t

    @property
    def t_tau(self):
        """Return normalized times for the only segment's stored time rows."""
        return self.single.t_tau

    @property
    def x(self):
        """Return stored states for the only segment.

        For multiple segments, select a segment by name.
        """
        return self.single.x

    @property
    def u(self):
        """Return stored controls for the only segment.

        For multiple segments, select a segment by name.
        """
        return self.single.u

    @property
    def x_tau(self):
        """Return normalized times for the only segment's stored state rows."""
        return self.single.x_tau

    @property
    def u_tau(self):
        """Return normalized times for the only segment's stored control rows."""
        return self.single.u_tau

    @property
    def x_t(self):
        """Return physical times for the only segment's stored state rows."""
        return self.single.x_t

    @property
    def u_t(self):
        """Return physical times for the only segment's stored control rows."""
        return self.single.u_t

    def sample(self, tau=None, *, num=None, spacing='tau', side=None):
        """Sample the only segment at explicit tau or uniformly in tau/physical time."""
        return self.single.sample(tau, num=num, spacing=spacing, side=side)

    def without_aux(self):
        return TrajectorySet({name: value.without_aux() for name, value in self.items()})

    def concatenate(self, names=None, *, segment_id='combined'):
        """Join contiguous physical segments and preserve both sides of each endpoint.

        Channel names, order, units, and frames must match. The result excludes
        auxiliary fields. This operation does not apply reset maps or convert
        channels.
        """
        names = list(self if names is None else names)
        if not names or len(set(names)) != len(names):
            raise ValueError('Concatenation needs a nonempty sequence of distinct segment names')
        selected = [self[name].without_aux() for name in names]
        layout = selected[0].layout
        if not layout.get('state') or 'control' not in layout:
            raise ValueError('Concatenation requires explicit state and control channel metadata')
        for value in selected[1:]:
            if value.layout != layout:
                raise ValueError('Concatenation requires identical channel identities, order, units and frames')
        starts = np.array([value.sample([0.]).t[0] for value in selected])
        stops = np.array([value.sample([1.]).t[0] for value in selected])
        if np.any(np.diff(starts) <= 0) or not np.allclose(stops[:-1], starts[1:], rtol=0, atol=1e-10):
            raise ValueError('Concatenation requires contiguous, strictly time-ordered segments')
        duration = stops[-1]-starts[0]
        widths = (stops-starts)/duration
        edges = np.r_[0., np.cumsum(widths)]
        edges[-1] = 1.
        fields = {}
        shared = set.intersection(*(set(value.fields) for value in selected))
        for name in shared:
            representations = [value.fields[name] for value in selected]
            kinds = {value.interpolation for value in representations}
            if len(kinds) != 1:
                raise ValueError('Concatenation requires matching interpolation kinds per field')
            sides = {value.boundary_side for value in representations}
            if len(sides) != 1:
                raise ValueError('Concatenation requires matching boundary conventions per field')
            intervals = []
            for i, source in enumerate(representations):
                for nodes, values in zip(source.nodes, source.values):
                    mapped_nodes = edges[i]+widths[i]*nodes
                    if nodes[0] == 0:
                        mapped_nodes[0] = edges[i]
                    if nodes[-1] == 1:
                        mapped_nodes[-1] = edges[i+1]
                    mapped_values = values/widths[i] if name == 's' else values
                    intervals.append((mapped_nodes, mapped_values))
            fields[name] = TrajectoryField.from_intervals(
                intervals, interpolation=kinds.pop(), boundary_side=sides.pop())
        return TrajectoryData(fields, segment_id, layout)
