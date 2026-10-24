"""Immutable numerical method specifications and global SCP settings."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral
import math

from trajopt.utils._snapshot import freeze, thaw
from trajopt.utils.tools import AttrDict, deep_merge, recursive_attrdict, resolve_function_from_string


def _adaptive_values(value):
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("Adaptive trust-region settings must be a mapping")
    value = dict(value)
    limit = value.get('max_rejects')
    if isinstance(limit, bool) or not isinstance(limit, Integral) or limit < 1:
        raise ValueError("adaptive.max_rejects must be a positive integer")
    lower, upper = float(value['s_mult_min']), float(value['s_mult_max'])
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= upper):
        raise ValueError("adaptive s_mult bounds must be finite, positive, and ordered")
    if not (math.isfinite(float(value['on_solve_fail'])) and float(value['on_solve_fail']) > 0):
        raise ValueError("adaptive.on_solve_fail must be finite and positive")
    if not (math.isfinite(float(value['pred_tol'])) and float(value['pred_tol']) >= 0):
        raise ValueError("adaptive.pred_tol must be finite and nonnegative")
    previous = -math.inf
    bands = value.get('bands', ())
    for band in bands:
        below, factor = float(band['below']), float(band['s_factor'])
        if not below > previous or not (math.isfinite(factor) and factor > 0):
            raise ValueError("adaptive bands must have increasing thresholds and finite positive s_factor")
        if not isinstance(band['accept'], bool):
            raise ValueError("adaptive band accept must be a boolean")
        previous = below
    if not bands or previous != math.inf:
        raise ValueError("adaptive bands must end with below: .inf")
    return value


def _policy_values(config):
    hyper = config.get("hyperparams", {})
    trust = hyper.get("trust_region", {})
    recovery = config.get("fcns", {}).get("troubleshoot", {}) or {}
    explicit = config.get("policies") or {}
    if not isinstance(explicit, Mapping):
        raise TypeError("Method policies must be a mapping")
    unknown = set(explicit) - {"on_error", "on_step", "line_search", "alpha_min_ls", "iter_max", "adaptive"}
    if unknown:
        raise ValueError(f"Unknown global method policies: {sorted(unknown)}")
    return {
        "on_error": explicit.get("on_error", recovery.get("on_error")),
        "on_step": explicit.get("on_step", recovery.get("on_step")),
        "line_search": bool(explicit.get("line_search", trust.get("line_search", True))),
        "alpha_min_ls": float(explicit.get("alpha_min_ls", trust.get("alpha_min_ls", 1e-7))),
        "iter_max": int(explicit.get("iter_max", hyper.get("convergence", {}).get("iter_max", 100))),
        "adaptive": _adaptive_values(explicit.get("adaptive", trust.get("adaptive"))),
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
        base.pop("segments", None)
        for name, override in self.inputs.get("segments", {}).items():
            effective = deep_merge(base, recursive_attrdict(thaw(override)))
            if freeze(_policy_values(effective)) != self.policies:
                raise ValueError(f"Segment {name!r} overrides a global policy; put it in method.policies")

    @classmethod
    def from_config(cls, config):
        return config if isinstance(config, cls) else cls(config.get("method", config))

    @property
    def config(self):
        return recursive_attrdict(thaw(self.inputs))

    def segment_config(self, name):
        config = self.config
        overrides = config.pop("segments", {})
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
