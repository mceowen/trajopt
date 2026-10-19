"""Mesh-independent guesses, providers, sampling, and nondimensionalization."""
import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import BarycentricInterpolator

from trajopt.methods.common import discretize
from trajopt.utils.tools import AttrDict, deep_merge

CHAIN_FROM_PREVIOUS = 'previous'


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
                  interpolation='linear'):
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
            interpolation=interpolation,
        )

    @classmethod
    def from_samples(cls, *, t, x, u, tau=None, s=None, aux=None,
                     interpolation='linear'):
        """Interpolate copied samples; tau defaults to normalized physical time.

        Polynomial interpolation uses one global polynomial. Non-affine
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
        fields = {}
        for name in ('t', 'x', 'u', 's', 'aux'):
            a = getattr(values, name)
            if a is None:
                fields[name] = None
            elif interpolation == 'polynomial':
                fields[name] = BarycentricInterpolator(tau, a, axis=0)
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
                    result[name] = source(q)
            return GuessSamples(**result)
        return cls(evaluate, tau)

    def sample(self, tau):
        tau = np.asarray(tau, dtype=float)
        if tau.ndim != 1 or not np.isfinite(tau).all() or np.any((tau < 0) | (tau > 1)):
            raise ValueError("Guess evaluation coordinates must lie in [0, 1]")
        return _samples(self._evaluate(tau.copy()), len(tau))


def _constraint_of_type(phase, type_name):
    return next((c for c in phase.constraints.values() if c.type == type_name), None)


def guess_options(phase, method_phase):
    options = method_phase.hyperparams.get('initial_guess', AttrDict())
    defaults = deep_merge(options.get('default', AttrDict()), options.get(phase.name, AttrDict()))
    return deep_merge(copy.deepcopy(defaults), copy.deepcopy(phase.guess))


def guess_field(phase, subproblem, name, default=None):
    context = getattr(subproblem, 'guess_context', None)
    if context is None:
        context = _context(subproblem, getattr(subproblem, '_previous_guess', None))
    return context.options.get(name, default)


def dimensional_samples(phase, z, nu):
    x, t, aux, u, s = phase.index_map.unpack_znu(np.asarray(z), np.asarray(nu))
    return GuessSamples(t[:, 0] * phase.nondim.time_scale,
                        x @ phase.nondim.M.state.nd2d.T,
                        u @ phase.nondim.M.control.nd2d.T,
                        s[:, 0] * phase.nondim.time_scale, aux.copy())


def guess_endpoint(method_phase):
    return dimensional_samples(method_phase.phase, method_phase.initial_guess.z,
                               method_phase.initial_guess.nu).x[-1]


def _context(method_phase, previous):
    phase = method_phase.phase
    options = guess_options(phase, method_phase)
    for field, value in options.items():
        if isinstance(value, str) and value.startswith('fcns.') and field != 'fcn':
            options[field] = phase.index_map.call_fcn(method_phase, value[len('fcns.'):])
    for field in ('x_start', 't_start'):
        value = options.get(field)
        if isinstance(value, str):
            if value != CHAIN_FROM_PREVIOUS:
                raise ValueError(f"phase '{phase.name}': unknown guess.{field} value {value!r}")
            if previous is None:
                raise ValueError(f"phase '{phase.name}': guess.{field}='previous' needs a preceding phase")
            options[field] = previous.x[-1].copy() if field == 'x_start' else float(previous.t[-1])
    return GuessContext(phase, method_phase.mesh, options, previous)


def _times(context):
    phase, cfg = context.problem, context.options
    initial = _constraint_of_type(phase, 'initial_time')
    final = _constraint_of_type(phase, 'final_time')
    start = float(initial.value) * phase.nondim.time_scale if initial is not None else float(cfg.get('t_start', 0.))
    if initial is not None and final is not None and final.is_fixed:
        stop = float(final.fixed_value) * phase.nondim.time_scale
    elif 'duration' in cfg:
        stop = start + float(cfg.duration)
    elif 't_stop' in cfg:
        stop = float(cfg.t_stop)
    else:
        raise ValueError(f"phase '{phase.name}': guess needs duration or t_stop")
    if not np.isfinite([start, stop]).all() or stop <= start:
        raise ValueError(f"phase '{phase.name}': guess must have a positive finite duration")
    return start, stop


def _endpoint_states(context):
    phase, cfg = context.problem, context.options
    n_x, d2nd = phase.index_map.n.state, phase.nondim.M.state.d2nd
    if 'x_start' in cfg:
        x0 = d2nd @ np.atleast_1d(cfg.x_start)
    else:
        constraint = _constraint_of_type(phase, 'initial_state')
        if constraint is None:
            raise ValueError(f"phase '{phase.name}': guess needs x_start or an initial_state constraint")
        x0 = np.zeros(n_x)
        x0[np.asarray(constraint.idx, dtype=int)] = constraint.value
    xf = x0.copy()
    if 'x_stop' in cfg:
        xf = d2nd @ np.atleast_1d(cfg.x_stop)
    else:
        constraint = _constraint_of_type(phase, 'final_state')
        if constraint is not None:
            xf[np.asarray(constraint.idx, dtype=int)] = constraint.value
    return x0, xf


def straight_line(context):
    """Linear states/controls with zero accumulators."""
    phase, cfg = context.problem, context.options
    start, stop = _times(context)
    x0, xf = _endpoint_states(context)
    return Guess.from_samples(t=[start, stop],
                              x=np.array([x0, xf]) @ phase.nondim.M.state.nd2d.T,
                              u=np.array([cfg.u_start, cfg.u_stop]),
                              aux=np.zeros((2, len(phase.index_map.indices.z.augmented))))


def propagation(context, propagate_rk4=discretize.propagate_rk4):
    """RK4 with ramped controls and constant dt/dtau."""
    phase, cfg = context.problem, context.options
    index, idx = phase.index_map, phase.index_map.indices
    start, stop = _times(context)
    x0, _ = _endpoint_states(context)
    u0 = phase.nondim.M.control.d2nd @ np.atleast_1d(cfg.u_start)
    uf = phase.nondim.M.control.d2nd @ np.atleast_1d(cfg.u_stop)
    N = index.N.all
    z0 = np.zeros(index.n.z)
    z0[idx.z.state], z0[idx.z.time] = x0, start / phase.nondim.time_scale
    tau_ref = jnp.linspace(0., 1., N)
    u_ref = jnp.asarray(np.linspace(0, 1, N)[:, None] * (uf-u0) + u0)
    ctrl, dilation = jnp.array(idx.nu.control), jnp.array(idx.nu.dilation_factor)
    sigma = jnp.asarray((stop-start) / phase.nondim.time_scale)

    def nu_fn(z, tau):
        k = jnp.clip(jnp.searchsorted(tau_ref, tau, side='right') - 1, 0, N-2)
        a = (tau-tau_ref[k]) / (tau_ref[k+1]-tau_ref[k])
        return jnp.zeros(index.n.nu).at[ctrl].set((1-a)*u_ref[k]+a*u_ref[k+1]).at[dilation].set(sigma)

    dynamics = _constraint_of_type(phase, 'dynamics')
    if dynamics is None:
        raise ValueError(f"phase '{phase.name}': propagation needs dynamics")
    tau, z, nu = propagate_rk4(z0, 0., 1., nu_fn, dynamics.fcn_znu,
                                         phase.params, n_steps=10*(N-1))
    values = dimensional_samples(phase, z, nu)
    # Restore exact affine time after integration.
    return Guess.from_samples(t=start+(stop-start)*tau, x=values.x, u=values.u,
                              tau=tau, s=values.s, aux=values.aux)


PROVIDERS = {'straight_line': straight_line, 'propagation': propagation}


def straight_line_initial_guess(phase, subproblem):
    return straight_line(subproblem.guess_context)


def nonlinear_initial_guess(phase, subproblem):
    return propagation(subproblem.guess_context, subproblem.fcns.discretize.propagate_rk4)


def _configured_guess(method_phase, context):
    phase = method_phase.phase
    kind = context.options.get('type')
    if kind == 'custom':
        name = context.options.get('fcn')
        provider = phase.fcns.get(name.removeprefix('fcns.')) if isinstance(name, str) else None
        if not callable(provider):
            raise ValueError(f"phase '{phase.name}': unknown guess function {name!r}")
        return provider(context)
    if kind is not None:
        if kind not in PROVIDERS:
            raise ValueError(f"phase '{phase.name}': unknown guess type {kind!r}")
        return (propagation(context, method_phase.fcns.discretize.propagate_rk4)
                if kind == 'propagation' else straight_line(context))

    fcns = method_phase.fcns.initial_guess
    provider = fcns.get(phase.name, fcns.get('default', AttrDict())).get('guess')
    if not callable(provider):
        raise ValueError(f"phase '{phase.name}': configure fcns.initial_guess.<phase>.guess or default.guess")
    scratch = copy.copy(method_phase)
    scratch.guess_context = context
    scratch.initial_guess = AttrDict()
    if 't_stop' in context.options or 'duration' in context.options or not scratch.free_final_time:
        start, stop = _times(context)
        t = (start + context.mesh.tau * (stop-start)) / phase.nondim.time_scale
        scratch.initial_guess.update(t=t, dt=np.diff(t[:, None], axis=0))
        scratch.Ts_init = t[-1]-t[0]
    result = provider(phase, scratch)
    if result is None and 'z' in scratch.initial_guess and 'nu' in scratch.initial_guess:
        values = dimensional_samples(phase, scratch.initial_guess.z, scratch.initial_guess.nu)
        result = Guess.from_samples(t=values.t, x=values.x, u=values.u, s=values.s,
                                    aux=values.aux, tau=context.mesh.tau)
    return result


def _dense_grid(target, native):
    nodes = np.sort(np.concatenate((target, native if native is not None else [],
                                    np.linspace(0, 1, 10*(len(target)-1)+1))))
    # Merge roundoff-separated nodes, retaining exact target coordinates.
    nodes = nodes[np.r_[True, np.diff(nodes) > 1e-12]]
    for t in target:
        nodes[np.argmin(np.abs(nodes-t))] = t
    return nodes


def prepare_initial_guess(method_phase, supplied=None, previous=None):
    """Prepare a guess without modifying the current iterate."""
    phase = method_phase.phase
    if supplied is None:
        context = _context(method_phase, previous)
        supplied = _configured_guess(method_phase, context)
    if not isinstance(supplied, Guess):
        raise TypeError(f"phase '{phase.name}': initial guess must be a Guess")

    target = method_phase.mesh.tau
    dense_tau = _dense_grid(target, supplied.native_tau)
    values = supplied.sample(dense_tau)
    n = phase.index_map.n
    if values.x.shape[1] != n.state or values.u.shape[1] != n.control:
        raise ValueError(f"phase '{phase.name}': expected {n.state} states and {n.control} controls")
    if np.any(np.diff(values.t) <= 0):
        raise ValueError(f"phase '{phase.name}': guess times must strictly increase")
    duration = values.t[-1]-values.t[0]
    s = values.s
    if s is None:
        if not np.allclose(values.t, values.t[0]+dense_tau*duration, rtol=1e-8, atol=1e-9):
            raise ValueError("A non-affine time map requires explicit s=dt/dtau")
        s = np.full(len(dense_tau), duration)
    if np.any(s <= 0):
        raise ValueError("Guess time dilation must be positive")
    initial_time = method_phase.find_constraint('initial_time')
    if initial_time is not None:
        start = float(initial_time.value) * phase.nondim.time_scale
        # The anchored dt[0] prevents correcting a mismatched start epoch.
        if not np.isclose(values.t[0], start, rtol=1e-10, atol=1e-10):
            raise ValueError("Guess start time must match the initial_time constraint")
    if not method_phase.free_final_time:
        start = float(method_phase.find_constraint('initial_time').value) * phase.nondim.time_scale
        stop = float(method_phase.find_constraint('final_time').fixed_value) * phase.nondim.time_scale
        if not (np.allclose(values.t, start+dense_tau*(stop-start)) and np.allclose(s, stop-start)):
            raise ValueError("A fixed-time problem requires its prescribed affine time map")
    n_aux = len(phase.index_map.indices.z.augmented)
    aux = values.aux
    if aux is None:
        aux = np.zeros((len(dense_tau), n_aux))
    if aux.shape != (len(dense_tau), n_aux):
        raise ValueError(f"phase '{phase.name}': expected {n_aux} auxiliary accumulator columns")
    z, nu = phase.index_map.pack_znu(values.x @ phase.nondim.M.state.d2nd.T,
                                    values.t[:, None]/phase.nondim.time_scale, aux,
                                    values.u @ phase.nondim.M.control.d2nd.T,
                                    s[:, None]/phase.nondim.time_scale)
    if values.aux is None and n_aux:
        dynamics = _constraint_of_type(phase, 'dynamics')
        derivatives = np.asarray(jax.jit(jax.vmap(dynamics.fcn_znu, in_axes=(0, 0, None)))(z, nu, phase.params))
        z[:, phase.index_map.indices.z.augmented] = cumulative_trapezoid(
            derivatives[:, phase.index_map.indices.z.augmented], dense_tau, axis=0, initial=0)
    if not np.isfinite(z).all() or not np.isfinite(nu).all():
        raise ValueError(f"phase '{phase.name}': imported guess contains nonfinite values")
    indices = np.searchsorted(dense_tau, target)
    z_nodes, nu_nodes = z[indices].copy(), nu[indices].copy()
    t = z_nodes[:, phase.index_map.indices.z.time].ravel()
    return AttrDict(t=t, dt=np.diff(t[:, None], axis=0), z=z_nodes, nu=nu_nodes,
                    z_dense=z, nu_dense=nu)


def set_initial_guess(phase, method_phase):
    """Initialize an SCP or SQP guess."""
    method_phase.initial_guess = prepare_initial_guess(
        method_phase, getattr(method_phase, '_supplied_guess', None),
        getattr(method_phase, '_previous_guess', None))
    method_phase.Ts_init = method_phase.initial_guess.t[-1]-method_phase.initial_guess.t[0]
    method_phase.cost_init = method_phase.cost_type_module.compute_nonconvex_terminal_costs(
        method_phase.initial_guess.z, method_phase.initial_guess.nu, phase, method_phase)


def export_guess(method_phase, *, include_aux=True):
    """Snapshot native interpolation; omit aux for destination reconstruction."""
    values = dimensional_samples(method_phase.phase, method_phase.current_iter_data.z_opt,
                                 method_phase.current_iter_data.nu_opt)
    mesh = method_phase.mesh
    return Guess.from_samples(t=values.t, x=values.x, u=values.u, s=values.s,
                              aux=values.aux if include_aux else None,
                              tau=mesh.tau, interpolation='polynomial' if mesh.discretize == 'ps' else 'linear')


def phase_guesses(supplied, names):
    names = list(names)
    if supplied is None:
        return {}
    if isinstance(supplied, Guess) and len(names) == 1:
        return {names[0]: supplied}
    if not isinstance(supplied, Mapping):
        raise TypeError('A multiphase initial_guess must map phase names to Guess objects')
    unknown = set(supplied)-set(names)
    if unknown:
        raise ValueError(f'Unknown initial-guess phases: {sorted(unknown)}')
    if any(not isinstance(value, Guess) for value in supplied.values()):
        raise TypeError('Each supplied phase initial_guess must be a Guess')
    return supplied
