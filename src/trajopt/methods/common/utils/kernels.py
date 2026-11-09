"""Keys for reusing built-in numerical kernels within one preparation."""
from collections.abc import Mapping
from functools import partial

from trajopt.common._snapshot import fingerprint
from trajopt.formulations.stable import constraint_types


def _callbacks(value):
    if isinstance(value, partial):
        return (value.func, _callbacks(value.args), _callbacks(value.keywords))
    if isinstance(value, Mapping):
        return {key: _callbacks(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_callbacks(item) for item in value)
    return value


def constraint_key(constraint):
    supported = (constraint_types.dynamics, constraint_types.nonconvex_inequality,
                 constraint_types.initial_nonconvex_inequality,
                 constraint_types.final_nonconvex_inequality,
                 constraint_types.nonconvex_equality,
                 constraint_types.initial_nonconvex_equality,
                 constraint_types.final_nonconvex_equality)
    if type(constraint) not in supported:
        return None
    if getattr(constraint, 'ctcs_constraints', ()) or getattr(constraint, 'running_costs', ()):
        return None
    return fingerprint((type(constraint), _callbacks(constraint.fcn_txu_dim),
                        constraint.index_map.indices, constraint.index_map.n,
                        constraint.M_out_d2nd, constraint.M_state_nd2d,
                        constraint.M_ctrl_nd2d, constraint.time_scale,
                        getattr(constraint, '_lower_dim', None),
                        getattr(constraint, '_upper_dim', None)))
