"""Prepare each segment and assemble the joint convex subproblem."""
import time
from collections.abc import Mapping
from dataclasses import dataclass

import cvxpy as cp
from trajopt.utils._snapshot import fingerprint, thaw
from trajopt.methods.initial_guess import preparation as guesses
from trajopt.methods.common.scp.subproblem import Subproblem
from trajopt.methods.specs import MethodSpec
from trajopt.formulations.specs import ProblemSpec
from trajopt.utils.tools import AttrDict


@dataclass
class PreparedCoupling:
    """A coupling specification, its constraint, and the connected subproblems."""

    spec: object
    constraint: object
    source: Subproblem
    target: Subproblem

    def residual(self):
        """Return the scaled physical residual at the accepted reference trajectories."""
        return self.constraint.constraint.accepted_residual(self.source, self.target)


@dataclass
class PreparedSolve:
    """Prepared segments and couplings with their joint CVXPY problem."""

    problem: ProblemSpec
    method: MethodSpec
    segments: Mapping
    couplings: tuple[PreparedCoupling, ...]
    physical_segments: Mapping
    cp_cost: object
    cp_constraints: list
    cp_subproblem: cp.Problem
    preparation_s: float
    signature: str

    @property
    def subproblem(self):
        if len(self.segments) != 1:
            raise ValueError("Use segments[name] for a problem with multiple segments")
        return next(iter(self.segments.values()))

    def update_inputs(self, updates):
        """Update supported numeric inputs and restore previous values if the update fails."""
        from trajopt.methods.bindings import update_inputs
        return update_inputs(self, updates)


def _prepare_segments(physical, method, initial_guess=None):
    """Initialize the segment subproblems in declaration order.

    A guess that uses the previous segment follows declaration order.
    Couplings are assembled after all segment variables exist. A coupling's
    source segment can therefore follow its target in declaration order.
    """
    method = MethodSpec.from_config(method)
    unknown = set(method.inputs.get("segments", {})) - set(physical)
    if unknown:
        raise ValueError(f"Method overrides unknown segments: {sorted(unknown)}")
    supplied = guesses.segment_guesses(initial_guess, physical)
    segments = AttrDict()
    previous = None
    for name, phase in physical.items():
        block = Subproblem(phase, method.segment_config(name),
                           initial_guess=supplied.get(name), previous_guess=previous)
        segments[name] = block
        initialized = block.initial_guess
        previous = guesses.dimensional_samples(phase, initialized.z, initialized.nu)
    return segments


def _prepare_couplings(problem, segments):
    couplings = []
    for spec in problem.couplings:
        target = segments[spec.target]
        constraint = target.constraints[spec.name]
        constraint.attach_coupling(segments)
        couplings.append(PreparedCoupling(spec, constraint, segments[spec.source], target))
    return tuple(couplings)


def _without_display_flags(config):
    """Copy the configuration and remove display settings from compatibility checks."""
    config = thaw(config)
    hyper = config.get('hyperparams', {})
    flags = hyper.get('flags', {})
    flags.pop('quiet', None)
    if not flags:
        hyper.pop('flags', None)
    if not hyper:
        config.pop('hyperparams', None)
    overrides = config.get('segments', {})
    for name, override in list(overrides.items()):
        override = _without_display_flags(override)
        if override:
            overrides[name] = override
        else:
            overrides.pop(name)
    if not overrides:
        config.pop('segments', None)
    return config


def _signature(problem, method, segments):
    # This is deliberately conservative: recipes/scales/constraint definitions
    # participate, while runtime values and a guess's sampling history do not.
    structure = []
    for name, block in segments.items():
        config = problem.segments[name].config
        config.pop("guess", None)
        config.pop("params", None)
        for constraint in config.get("constraints", {}).values():
            if constraint.get("type") == "initial_state":
                value = constraint.get("value", ())
                constraint["value"] = [None if x is None else "numeric" for x in value]
        mesh = block.mesh
        structure.append((name, config,
                          (mesh.discretize, mesh.segments, mesh.tau,
                           mesh.differentiation, mesh.weights),
                          block.index_map.n.z, block.index_map.n.nu))
    method_config = _without_display_flags(method.config)
    method_config.pop("solver_opts", None)
    for section in ("initial_guess",):
        method_config.get("hyperparams", {}).pop(section, None)
        method_config.get("fcns", {}).pop(section, None)
    payload = (structure, tuple((c.name, c.kind, c.source, c.target, c.inputs)
                                for c in problem.couplings), method_config)
    return fingerprint(payload)


def prepare(problem, method, initial_guess=None):
    """Build the physical models and their joint convex subproblem."""
    started = time.perf_counter()
    method_spec = MethodSpec.from_config(method)
    spec = ProblemSpec.from_config(problem)
    physical = spec.build_segments()
    segments = _prepare_segments(physical, method_spec, initial_guess)
    couplings = _prepare_couplings(spec, segments)
    cost = sum(segment.cp_cost for segment in segments.values())
    constraints = [c for segment in segments.values() for c in segment.cp_constraints]
    backend = cp.Problem(cp.Minimize(cost), constraints)
    return PreparedSolve(spec, method_spec, segments, couplings, physical, cost, constraints,
                         backend, time.perf_counter() - started,
                         _signature(spec, method_spec, segments))
