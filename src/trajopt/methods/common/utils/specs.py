"""Immutable numerical method specifications and global SCP settings."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral

import numpy as np

from trajopt.common._snapshot import freeze, thaw
from trajopt.common.tools import (
    AttrDict,
    deep_merge,
    recursive_attrdict,
    resolve_function_from_string,
)
from trajopt.methods.common.convergence import read_iteration_limit
from trajopt.methods.common.troubleshoot import read_recovery_settings
from trajopt.methods.common.trust_region import read_step_settings


def _policy_values(config):
    explicit = config.get("policies") or {}
    if not isinstance(explicit, Mapping):
        raise TypeError("Method policies must be a mapping")
    unknown = set(explicit) - {"on_error", "on_step", "line_search", "alpha_min_ls", "iter_max", "adaptive"}
    if unknown:
        raise ValueError(f"Unknown global method policies: {sorted(unknown)}")
    return {
        **read_recovery_settings(config, explicit),
        **read_step_settings(config, explicit),
        "iter_max": read_iteration_limit(config, explicit),
    }


@dataclass(frozen=True)
class MethodSpec:
    """Immutable method configuration, separate from the algorithm state."""

    inputs: Mapping
    policies: Mapping = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "inputs", freeze(self.inputs))
        object.__setattr__(self, "policies", freeze(_policy_values(self.inputs)))
        base = self.config
        base.pop("phases", None)
        for name, override in self.inputs.get("phases", {}).items():
            effective = deep_merge(base, recursive_attrdict(thaw(override)))
            if freeze(_policy_values(effective)) != self.policies:
                raise ValueError(f"Phase {name!r} overrides a global policy; put it in method.policies")

    @classmethod
    def from_config(cls, config):
        return config if isinstance(config, cls) else cls(config.get("method", config))

    @property
    def config(self):
        return recursive_attrdict(thaw(self.inputs))

    def phase_config(self, name):
        config = self.config
        overrides = config.pop("phases", {})
        config = deep_merge(config, overrides.get(name, AttrDict()))
        if 'adaptive' in config.get('policies', {}):
            config.setdefault('hyperparams', AttrDict()).setdefault('trust_region', AttrDict())['adaptive'] = (
                recursive_attrdict(thaw(self.policies['adaptive'])))
        return config

    @property
    def on_error(self):
        value = self.policies["on_error"]
        return resolve_function_from_string(value) if isinstance(value, str) else value

    @property
    def on_step(self):
        value = self.policies["on_step"]
        return resolve_function_from_string(value) if isinstance(value, str) else value

    @property
    def line_search(self):
        return self.policies["line_search"]

    @property
    def alpha_min_ls(self):
        return self.policies["alpha_min_ls"]

    @property
    def iter_max(self):
        return self.policies["iter_max"]

    @property
    def max_rejects(self):
        adaptive = self.policies['adaptive']
        return None if adaptive is None else int(adaptive['max_rejects'])


@dataclass(frozen=True)
class SolveLimits:
    max_attempts: int
    wall_time_s: float


def prepare_solve(solver, *, verbose=None, initial_guess=None, budget=None):
    """Check the limits and settings before starting the solve."""
    from trajopt.methods.common.utils.solverstate import check_reference_inputs

    solver.method_spec = MethodSpec.from_config(solver.method_config)
    budget = {} if budget is None else dict(budget)
    if set(budget) - {'max_attempts', 'wall_time_s'}:
        raise ValueError('Budget supports max_attempts and wall_time_s')
    limit = budget.get('max_attempts', solver.method_spec.iter_max + 1)
    if isinstance(limit, (bool, np.bool_)) or not isinstance(limit, Integral) or limit < 0:
        raise ValueError('max_attempts must be a nonnegative integer')
    deadline = float(budget.get('wall_time_s', np.inf))
    if deadline < 0 or np.isnan(deadline):
        raise ValueError('Solve budgets must be nonnegative and not NaN')
    retention = solver.method_config.get('history', 'scalars')
    if retention not in {'none', 'scalars', 'full'}:
        raise ValueError('history must be none, scalars, or full')
    if initial_guess is not None:
        solver.reset(initial_guess=initial_guess)
    check_reference_inputs(solver)
    if verbose is not None:
        solver.reporter.quiet = not verbose
    for phase in solver.phases.values():
        phase.method_config.history = retention
    return SolveLimits(int(limit), deadline)
