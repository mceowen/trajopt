"""Piecewise trajectory fields with linear, constant, or polynomial interpolation."""
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import BarycentricInterpolator

from trajopt.utils._snapshot import readonly_array


def _queries(value):
    q = np.asarray(value, dtype=float)
    if q.ndim != 1 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError('Trajectory evaluation coordinates must lie in [0, 1]')
    return q


def _nodes(value, *, endpoints=True):
    a = np.asarray(value, dtype=float)
    if (a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all()
            or np.any(np.diff(a) <= 0)):
        raise ValueError('Trajectory nodes must be finite and strictly increasing')
    if endpoints and not (np.isclose(a[0], 0, rtol=0, atol=1e-12)
                          and np.isclose(a[-1], 1, rtol=0, atol=1e-12)):
        raise ValueError('Trajectory nodes must span [0, 1]')
    a = a.copy()
    if endpoints:
        a[0], a[-1] = 0., 1.
    return a


def _degrees(segments, count):
    if isinstance(segments, (int, np.integer)) and not isinstance(segments, bool):
        if segments < 1 or (count-1) % segments:
            raise ValueError('segments must divide the number of sample intervals')
        return [(count-1)//segments] * int(segments)
    if isinstance(segments, Sequence):
        degrees = list(segments)
        if (not degrees or any(not isinstance(p, (int, np.integer)) or isinstance(p, bool)
                               or p < 1 for p in degrees) or sum(degrees) != count-1):
            raise ValueError('Interval degrees must be positive integers summing to N-1')
        return degrees
    raise TypeError('segments must be an interval count or a sequence of local degrees')


def _polynomial_interpolator(nodes, values):
    """Build a barycentric interpolator with reproducible weights.

    Compute the weights from scaled products with a fixed random permutation.
    This preserves SciPy's numerical strategy without changing the global
    random number generator state. Repeated calls and saved trajectories use
    the same weights.
    """
    count = len(nodes)
    permutation = np.random.default_rng(0).permutation(count)
    inverse = np.empty(count, dtype=int)
    inverse[permutation] = np.arange(count)
    capacity = 4.0/(nodes[-1]-nodes[0])
    weights = np.empty(count)
    for i in range(count):
        distances = capacity*(nodes[i]-nodes[permutation])
        distances[inverse[i]] = 1.
        with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
            weights[i] = 1./np.prod(distances)
    if not np.isfinite(weights).all() or np.any(weights == 0):
        raise ValueError('Polynomial nodes are too ill-conditioned for barycentric interpolation')
    return BarycentricInterpolator(nodes, values, axis=0, wi=readonly_array(weights))


@dataclass(frozen=True)
class TrajectoryField:
    """Trajectory samples on intervals with different degrees or endpoint jumps.

    ``nodes`` and ``values`` each contain one array per interval. Adjacent
    intervals share an endpoint coordinate but can have different values there.
    At a shared endpoint, ``sample(side='right')`` selects the next interval.
    ``side='left'`` selects the previous interval. If ``side`` is omitted,
    sampling uses the stored boundary convention, including dilation values
    from the preceding interval.
    """
    nodes: tuple
    values: tuple
    interpolation: str = 'linear'
    boundary_side: str = 'right'
    _interpolators: tuple = field(init=False, repr=False, compare=False, default=())

    def __post_init__(self):
        if self.interpolation not in ('linear', 'polynomial', 'constant'):
            raise ValueError("interpolation must be 'linear', 'polynomial' or 'constant'")
        if self.boundary_side not in ('left', 'right'):
            raise ValueError("Boundary side must be 'left' or 'right'")
        if not self.nodes or len(self.nodes) != len(self.values):
            raise ValueError('Each trajectory interval needs nodes and values')
        nodes, values, shape = [], [], None
        for i, (raw_nodes, raw_values) in enumerate(zip(self.nodes, self.values)):
            tau = _nodes(raw_nodes, endpoints=False)
            a = np.asarray(raw_values, dtype=float)
            if a.ndim not in (1, 2) or len(a) != len(tau) or not np.isfinite(a).all():
                raise ValueError('Field values must have one row per node and contain no nonfinite values')
            if shape is not None and a.shape[1:] != shape:
                raise ValueError('Field channel dimensions must agree across intervals')
            if self.interpolation == 'constant' and not np.all(a == a[0]):
                raise ValueError('Constant trajectory intervals must have identical values at every node')
            if i and not np.isclose(tau[0], nodes[-1][-1], rtol=0, atol=1e-12):
                raise ValueError('Trajectory intervals must cover contiguous coordinates')
            if i:
                tau[0] = nodes[-1][-1]
            shape = a.shape[1:]
            nodes.append(readonly_array(tau))
            values.append(readonly_array(a))
        if not (nodes[0][0] == 0 and nodes[-1][-1] == 1):
            raise ValueError('Trajectory intervals must span [0, 1]')
        object.__setattr__(self, 'nodes', tuple(nodes))
        object.__setattr__(self, 'values', tuple(values))
        if self.interpolation == 'polynomial':
            object.__setattr__(self, '_interpolators', tuple(
                _polynomial_interpolator(tau, value) for tau, value in zip(nodes, values)))

    @classmethod
    def from_samples(cls, tau, values, *, interpolation='linear', segments=1):
        tau = _nodes(tau)
        a = np.asarray(values, dtype=float)
        if a.ndim not in (1, 2) or len(a) != len(tau):
            raise ValueError('Field values must have one row per node')
        if not np.isfinite(a).all():
            raise ValueError('Field values must contain no nonfinite values')
        if interpolation == 'linear':
            if not isinstance(segments, (int, np.integer)) or segments != 1:
                raise ValueError('segments is only used for polynomial interpolation')
            return cls((tau,), (a,), interpolation)
        degrees = _degrees(segments, len(tau))
        starts = np.cumsum([0] + degrees[:-1])
        if interpolation == 'constant':
            # The opening knot carries the preceding segment's dilation.
            # The first interior/collocation value belongs to this segment.
            return cls(tuple(tau[[start, start+p]] for start, p in zip(starts, degrees)),
                       tuple(np.repeat(a[start+1:start+2], 2, axis=0) for start in starts),
                       interpolation, boundary_side='left')
        return cls(tuple(tau[start:start+p+1] for start, p in zip(starts, degrees)),
                   tuple(a[start:start+p+1] for start, p in zip(starts, degrees)), interpolation)

    @classmethod
    def from_intervals(cls, intervals, *, interpolation='polynomial', boundary_side='right'):
        """Build a field from ``[(tau0, values0), (tau1, values1), ...]``."""
        intervals = tuple(intervals)
        return cls(tuple(item[0] for item in intervals),
                   tuple(item[1] for item in intervals), interpolation, boundary_side)

    @property
    def native_tau(self):
        return readonly_array(np.unique(np.concatenate(self.nodes)))

    @property
    def degrees(self):
        return tuple(len(tau)-1 for tau in self.nodes)

    def sample(self, tau, *, side=None):
        q = _queries(tau)
        side = self.boundary_side if side is None else side
        if side not in ('left', 'right'):
            raise ValueError("Endpoint side must be 'left' or 'right'")
        ends = np.array([nodes[-1] for nodes in self.nodes])
        interval = np.minimum(np.searchsorted(ends, q, side=side), len(ends)-1)
        output = np.empty((len(q),) + self.values[0].shape[1:])
        for i, (nodes, values) in enumerate(zip(self.nodes, self.values)):
            mask = interval == i
            if not mask.any():
                continue
            if self.interpolation == 'polynomial':
                output[mask] = self._interpolators[i](q[mask])
            elif self.interpolation == 'constant':
                output[mask] = values[0]
            elif values.ndim == 1:
                output[mask] = np.interp(q[mask], nodes, values)
            elif values.shape[1]:
                output[mask] = np.column_stack([np.interp(q[mask], nodes, col) for col in values.T])
        return output
