"""Mesh-independent guesses in dimensional model units."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.interpolate import BarycentricInterpolator


@dataclass(frozen=True)
class GuessSamples:
    """Dimensional t/x/u and optional dt/dtau; aux uses target nondimensional units."""
    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    s: np.ndarray | None = None
    aux: np.ndarray | None = None


@dataclass(frozen=True)
class GuessContext:
    """Phase model, target mesh, merged options, and preceding phase samples."""
    problem: object
    mesh: object
    options: object
    previous: GuessSamples | None = None

    @property
    def params(self):
        return self.problem.params

    @property
    def fcns(self):
        return self.problem.fcns


def _grid(tau):
    tau = np.asarray(tau, dtype=float)
    if (tau.ndim != 1 or len(tau) < 2 or not np.isfinite(tau).all()
            or np.any(np.diff(tau) <= 0)
            or not np.isclose(tau[0], 0, rtol=0, atol=1e-12)
            or not np.isclose(tau[-1], 1, rtol=0, atol=1e-12)):
        raise ValueError("Guess nodes must strictly increase from 0 to 1")
    tau = tau.copy()
    tau[0], tau[-1] = 0.0, 1.0
    return tau


def _samples(value, n):
    if not isinstance(value, GuessSamples):
        raise TypeError("A guess evaluator must return GuessSamples")
    arrays = {}
    for name in ('t', 'x', 'u', 's', 'aux'):
        raw = getattr(value, name)
        if raw is None:
            if name in ('t', 'x', 'u'):
                raise ValueError(f"Guess {name} is required")
            arrays[name] = None
            continue
        a = np.array(raw, dtype=float, copy=True)
        if name in ('t', 's'):
            if a.ndim == 2 and a.shape[1] == 1:
                a = a[:, 0]
            if a.shape != (n,):
                raise ValueError(f"Guess {name} must have shape ({n},)")
        elif a.ndim != 2 or a.shape[0] != n:
            raise ValueError(f"Guess {name} must have shape ({n}, n_{name})")
        if not np.isfinite(a).all():
            raise ValueError(f"Guess {name} contains nonfinite values")
        arrays[name] = a
    return GuessSamples(**arrays)


class Guess:
    """A source trajectory evaluated at normalized coordinates in [0, 1]."""

    def __init__(self, evaluate: Callable, native_tau=None):
        if not callable(evaluate):
            raise TypeError("Guess evaluator must be callable")
        self._evaluate = evaluate
        self.native_tau = None if native_tau is None else _grid(native_tau)

    @classmethod
    def from_callable(cls, evaluate, *, native_tau=None):
        """Wrap evaluate(tau) -> GuessSamples; native_tau adds source sampling nodes."""
        return cls(evaluate, native_tau)

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
    def from_samples(cls, *, t, x, u, tau=None, s=None, aux=None,
                     interpolation='linear', segments=1):
        """Interpolate copied samples; tau defaults to normalized physical time.

        Polynomial segments share endpoints and have equal order. Non-affine
        time maps require s=dt/dtau. Omitted aux is reconstructed on import.
        """
        t = np.asarray(t, dtype=float)
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]
        if t.ndim != 1:
            raise ValueError("Guess t must have shape (N,) or (N, 1)")
        if len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError("Guess times must be finite and strictly increasing")
        tau = _grid((t - t[0]) / (t[-1] - t[0]) if tau is None else tau)
        values = _samples(GuessSamples(t, x, u, s, aux), len(tau))
        if interpolation not in ('linear', 'polynomial'):
            raise ValueError("interpolation must be 'linear' or 'polynomial'")
        if not isinstance(segments, (int, np.integer)) or segments < 1 or (len(tau)-1) % segments:
            raise ValueError("segments must divide the number of sample intervals")
        if interpolation == 'linear' and segments != 1:
            raise ValueError("segments is only used for polynomial interpolation")
        p = (len(tau)-1) // segments
        fields = {}
        for name in ('t', 'x', 'u', 's', 'aux'):
            a = getattr(values, name)
            if a is None:
                fields[name] = None
            elif interpolation == 'polynomial':
                fields[name] = [BarycentricInterpolator(tau[h*p:h*p+p+1], a[h*p:h*p+p+1], axis=0)
                                for h in range(segments)]
            else:
                fields[name] = a

        def evaluate(q):
            result = {}
            for name, source in fields.items():
                if source is None:
                    result[name] = None
                elif interpolation == 'linear':
                    if source.ndim == 1:
                        result[name] = np.interp(q, tau, source)
                    elif source.shape[1]:
                        result[name] = np.column_stack([np.interp(q, tau, col) for col in source.T])
                    else:
                        result[name] = np.zeros((len(q), 0))
                else:
                    a = getattr(values, name)
                    out = np.empty((len(q),) + a.shape[1:])
                    interval = np.clip(np.searchsorted(tau[::p], q, side='right')-1, 0, segments-1)
                    for h, polynomial in enumerate(source):
                        mask = interval == h
                        out[mask] = polynomial(q[mask])
                    result[name] = out
            return GuessSamples(**result)
        return cls(evaluate, tau)

    def sample(self, tau):
        tau = np.asarray(tau, dtype=float)
        if tau.ndim != 1 or not np.isfinite(tau).all() or np.any((tau < 0) | (tau > 1)):
            raise ValueError("Guess evaluation coordinates must lie in [0, 1]")
        return _samples(self._evaluate(tau.copy()), len(tau))
