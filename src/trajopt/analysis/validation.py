"""Evaluate physical trajectories from solve results.

Evaluation uses copies of the result data and new physical model functions.
It does not read the solver state or CVXPY variables.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
import time

import jax
import numpy as np

from trajopt.utils._snapshot import fingerprint, freeze, thaw
from trajopt.outputs.results import SolveResult
from trajopt.utils.tools import AttrDict, resolve_function_from_string


@dataclass(frozen=True)
class EvaluatedIterate:
    """One physical iterate with independently propagated states and outputs."""

    iter_num: int
    t_opt: np.ndarray
    x_opt: np.ndarray
    u_opt: np.ndarray
    t_nl: np.ndarray
    x_nl: np.ndarray
    u_nl: np.ndarray
    t_init_nl: np.ndarray
    x_init_nl: np.ndarray
    u_init_nl: np.ndarray
    outputs: Mapping

    def __post_init__(self):
        for item in fields(self):
            object.__setattr__(self, item.name, freeze(getattr(self, item.name)))

    @property
    def channels(self):
        return freeze({name: (out.opt[:, 0] if out.opt.ndim == 2 and out.opt.shape[1] == 1
                              else out.opt) for name, out in self.outputs.items()})


@dataclass(frozen=True)
class SegmentEvaluation:
    """Propagated trajectories and outputs in one segment's channel layout."""

    name: str
    iterates: tuple[EvaluatedIterate, ...]
    diagnostics: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if any(not isinstance(it, EvaluatedIterate) for it in self.iterates):
            raise TypeError("Segment evaluations require EvaluatedIterate values")
        object.__setattr__(self, "iterates", tuple(freeze(it) for it in self.iterates))
        object.__setattr__(self, "diagnostics", freeze(self.diagnostics))

    @property
    def final(self):
        if not self.iterates:
            raise ValueError(f"Segment {self.name!r} has no evaluated accepted iterate")
        return self.iterates[-1]


@dataclass(frozen=True)
class EvaluationResult:
    """Propagated physical trajectories with a separate layout for each segment."""

    solve_result: SolveResult
    segments: Mapping[str, SegmentEvaluation]
    protocol: Mapping = field(default_factory=dict)
    diagnostics: Mapping = field(default_factory=dict)
    timings: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.solve_result, SolveResult):
            raise TypeError("EvaluationResult requires a SolveResult")
        for name in ("segments", "protocol", "diagnostics", "timings"):
            object.__setattr__(self, name, freeze(getattr(self, name)))

    def concatenate(self, names):
        """Join compatible sequential segments in physical time order."""
        from trajopt.analysis.analysis import concat

        names = tuple(names)
        if not names or len(set(names)) != len(names):
            raise ValueError("Select distinct named segments to concatenate")
        source = self.solve_result
        if len({source.problem.segments[name].agent for name in names}) > 1:
            raise ValueError("Cannot concatenate trajectories belonging to different agents")
        source.trajectory.concatenate(names)
        histories = [self.segments[name].iterates for name in names]
        identities = [[it.iter_num for it in history] for history in histories]
        if any(ids != identities[0] for ids in identities[1:]):
            raise ValueError("Segment evaluation histories have different iteration identities")
        joined = []
        for per_iteration in zip(*histories):
            outputs = [it.outputs for it in per_iteration]
            if any(out.keys() != outputs[0].keys() for out in outputs[1:]):
                raise ValueError("Segment evaluations have different output identities")
            for name in outputs[0]:
                reference = outputs[0][name]
                if any(fingerprint(out[name].meta) != fingerprint(reference.meta) for out in outputs[1:]):
                    raise ValueError(f"Output {name!r} has different metadata across segments")
            values = {item.name: concat([getattr(it, item.name) for it in per_iteration])
                      for item in fields(EvaluatedIterate) if item.name != "outputs"}
            values["outputs"] = {
                name: {key: (output[key] if key == "meta" else
                             concat([out[name][key] for out in outputs]))
                       for key in output}
                for name, output in outputs[0].items()
            }
            joined.append(EvaluatedIterate(**values))
        return SegmentEvaluation("/".join(names), tuple(joined))


def _discretization_functions(config):
    from trajopt.methods.common import discretize

    defaults = ("make_node_propagation_solver", "make_trajectory_solver",
                "propagate_from_nodes", "propagate_trajectory")
    configured = config.get("fcns", {}).get("discretize", {})
    functions = AttrDict()
    for name in defaults:
        value = configured.get(name)
        functions[name] = (resolve_function_from_string(value) if isinstance(value, str)
                           else value if callable(value) else getattr(discretize, name))
    return functions


def detached_segments(result):
    """Build the physical models and copy the saved arrays for evaluation."""
    adapters = {}
    for name, snapshot in result.segment_data.items():
        phase = result.problem.segments[name].build()
        data = thaw(snapshot)
        method_config = data.get("method_config", result.method.segment_config(name))
        mesh = data.mesh
        adapters[name] = AttrDict(
            problem=phase, mesh=mesh, initial_guess=data.initial_guess,
            iter_data_list=data.get("iter_data_list", []),
            fcns=AttrDict(discretize=_discretization_functions(method_config)),
        )
    return adapters


def _physical_diagnostics(phase, iterate):
    """Evaluate boundary and path constraints on the propagated trajectory.

    Residuals use the physical units returned by each model function.
    Constraints that require derivatives or solver data are listed as
    unavailable.
    """
    residuals, unavailable = {}, []
    x, u, t = iterate.x_nl, iterate.u_nl, iterate.t_nl
    # Node rollouts insert all-NaN rows to separate shooting intervals in plots.
    separators = np.isnan(t) & np.isnan(x).all(axis=1) & np.isnan(u).all(axis=1)
    x, u, t = x[~separators], u[~separators], t[~separators]
    for name, constraint in phase.constraints.items():
        kind = constraint.type
        if kind == "dynamics":
            continue
        xx, uu, tt = x, u, t
        if kind.startswith("initial_"):
            xx, uu, tt = x[:1], u[:1], t[:1]
        elif kind.startswith("final_"):
            xx, uu, tt = x[-1:], u[-1:], t[-1:]
        values = None
        if kind in ("initial_state", "final_state"):
            values = np.abs(xx[:, constraint.idx] - constraint._value_dim)
        elif kind in ("initial_control", "final_control"):
            values = np.abs(uu[:, constraint.idx] - constraint._value_dim)
        elif kind.endswith("state_limits") or kind.endswith("control_limits"):
            source = xx if kind.endswith("state_limits") else uu
            pieces = []
            if constraint.lower_idx:
                pieces.append(constraint._lower_dim - source[:, constraint.lower_idx])
            if constraint.upper_idx:
                pieces.append(source[:, constraint.upper_idx] - constraint._upper_dim)
            values = np.maximum(np.concatenate(pieces, axis=1), 0.) if pieces else np.empty((len(tt), 0))
        elif "nonconvex_" in kind:
            values = np.asarray(jax.vmap(constraint.fcn_txu_dim, in_axes=(0, 0, 0, None))(
                xx, uu, tt, phase.params))
            if values.ndim == 1:
                values = values[:, None]
            if kind.endswith("_equality"):
                values = np.abs(values)
            else:
                pieces = []
                if constraint._lower_dim is not None:
                    pieces.append(np.atleast_1d(constraint._lower_dim) - values)
                if constraint._upper_dim is not None:
                    pieces.append(values - np.atleast_1d(constraint._upper_dim))
                values = np.maximum(np.concatenate(pieces, axis=1) if pieces else values, 0.)
        elif "convex_inequality" in kind:
            raw = constraint.fcn_xu_dim(xx, uu, phase.params)
            values = np.asarray(raw.value if hasattr(raw, "value") else raw)
            if values.ndim == 1:
                values = values[:, None]
            pieces = []
            if constraint._lower_dim is not None:
                pieces.append(np.atleast_1d(constraint._lower_dim) - values)
            if constraint._upper_dim is not None:
                pieces.append(values - np.atleast_1d(constraint._upper_dim))
            values = np.maximum(np.concatenate(pieces, axis=1) if pieces else values, 0.)
        elif kind == "initial_time":
            values = np.abs(tt - constraint._value_dim)
        elif kind == "final_time":
            pieces = []
            if constraint._lower_dim is not None:
                pieces.append(constraint._lower_dim - tt)
            if constraint._upper_dim is not None:
                pieces.append(tt - constraint._upper_dim)
            values = np.maximum(np.concatenate(pieces), 0.) if pieces else np.empty(0)
        else:
            unavailable.append(name)
        if values is not None:
            residuals[name] = AttrDict(values=np.asarray(values), maximum=float(np.max(values, initial=0.)))
    return AttrDict(
        endpoint_state_error=iterate.x_nl[-1] - iterate.x_opt[-1],
        endpoint_time_error=float(iterate.t_nl[-1] - iterate.t_opt[-1]),
        physical_constraint_violations=residuals,
        unevaluated_constraints=tuple(unavailable),
    )


def _coupling_diagnostics(problem, phases, segments):
    """Evaluate the segment couplings in physical units from trajectory endpoints."""
    from trajopt.formulations.common import constraint_types

    residuals = {}
    for spec in problem.couplings:
        source, target = phases[spec.source], phases[spec.target]
        coupling = getattr(constraint_types, spec.kind)(spec.config, target)
        values = {}
        for representation, suffix in (("optimized", "opt"), ("rollout", "nl")):
            groups = {}
            left, right = segments[spec.source].final, segments[spec.target].final
            for kind in coupling.groups:
                conversion = np.asarray(coupling._conversion(kind), dtype=float)
                if kind == "time":
                    groups[kind] = np.atleast_1d(
                        getattr(right, "t_" + suffix)[0] - conversion * getattr(left, "t_" + suffix)[-1])
                else:
                    source_sel, target_sel, _ = coupling._selections(kind)
                    si = constraint_types._continuity_indices(source, kind, source_sel)
                    ti = constraint_types._continuity_indices(target, kind, target_sel)
                    if len(si) != len(ti):
                        raise ValueError(f"Coupling {spec.name!r} has unequal channel mapping dimensions")
                    field = ("x_" if kind == "state" else "u_") + suffix
                    groups[kind] = getattr(right, field)[0, ti] - conversion * getattr(left, field)[-1, si]
            values[representation] = groups
        residuals[f"{spec.target}/{spec.name}"] = values
    return residuals


def evaluate(result, analysis_config=None):
    """Evaluate a solve result without constructing a solver.

    The result records the evaluation settings, physical residuals, and
    unevaluated constraints. It does not certify feasibility or optimality.
    """
    from trajopt.analysis.analysis import analyze_segment
    if not isinstance(result, SolveResult):
        raise TypeError("evaluate requires a SolveResult")

    start = time.perf_counter()
    protocol = AttrDict(compute_iters=False, propagate_from_nodes=False,
                        trajectory_steps=500, node_steps=50,
                        control_interpolation='lagrange')
    protocol.update(thaw(analysis_config or {}))
    for key in ("trajectory_steps", "node_steps"):
        if int(protocol[key]) < 1:
            raise ValueError(f"Evaluation {key} must be positive")
    if (protocol.compute_iters and result.iterations > 1
            and result.method.config.get("history", "scalars") != "full"):
        raise ValueError("All iterate trajectories were not retained; solve with method.history='full'")
    if result.trajectory is None:
        return EvaluationResult(result, {}, protocol,
                                {"unavailable": "No accepted trajectory", "solve_status": result.status},
                                {"evaluation_s": time.perf_counter() - start})
    segments, phases = {}, {}
    for name, adapter in detached_segments(result).items():
        phases[name] = adapter.problem
        mappings = analyze_segment(adapter, AttrDict(analysis=protocol))
        iterates = tuple(EvaluatedIterate(**mapping) for mapping in mappings)
        diagnostics = _physical_diagnostics(adapter.problem, iterates[-1]) if iterates else {}
        segments[name] = SegmentEvaluation(name, iterates, diagnostics)
    return EvaluationResult(
        result, segments, protocol,
        {"solve_status": result.status,
         "coupling_residuals": _coupling_diagnostics(result.problem, phases, segments)},
        {"evaluation_s": time.perf_counter() - start},
    )
