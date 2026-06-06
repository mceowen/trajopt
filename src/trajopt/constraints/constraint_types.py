from typing import Any

import cvxpy as cp
import jax.numpy as jnp
import numpy as np

from trajopt.utils import tools
from trajopt.constraints.constraint import Constraint


class initial_state(Constraint):
    def __init__(self, cnstr_config: dict, segment) -> None:
        self.name = cnstr_config.name
        self.type = "initial_state"

        nondim = segment.nondim
        raw_value = cnstr_config["value"]

        if "idx" in cnstr_config:
            self.idx = cnstr_config.idx
            self._value_dim = np.atleast_1d(raw_value)
        else:
            self.idx = [i for i, v in enumerate(raw_value) if v is not None]
            self._value_dim = np.atleast_1d([v for v in raw_value if v is not None])

        self.dimension = len(self.idx)
        self._nondim = nondim

    @property
    def value(self):
        return self._nondim.M.state.d2nd[np.ix_(self.idx, self.idx)] @ self._value_dim


class final_state(Constraint):
    def __init__(self, cnstr_config: dict, segment) -> None:
        self.name = cnstr_config.name
        self.type = "final_state"

        nondim = segment.nondim
        raw_value = cnstr_config.value

        if "idx" in cnstr_config:
            self.idx = cnstr_config["idx"]
            self._value_dim = np.atleast_1d(raw_value)
        else:
            self.idx = [i for i, v in enumerate(raw_value) if v is not None]
            self._value_dim = np.atleast_1d([v for v in raw_value if v is not None])

        self.dimension = len(self.idx)
        self._nondim = nondim

    @property
    def value(self):
        return self._nondim.M.state.d2nd[np.ix_(self.idx, self.idx)] @ self._value_dim


class state_limits(Constraint):
    def __init__(self, cnstr_config: dict, segment) -> None:
        index_map = segment.index_map
        nondim = segment.nondim

        self.name = cnstr_config.name
        self.type = "state_limits"

        raw_lower = cnstr_config.get("lower", [])
        raw_upper = cnstr_config.get("upper", [])

        self.lower_idx = [i for i, v in enumerate(raw_lower) if v is not None]
        self.upper_idx = [i for i, v in enumerate(raw_upper) if v is not None]
        self._lower_dim = np.atleast_1d([v for v in raw_lower if v is not None]) if self.lower_idx else np.array([])
        self._upper_dim = np.atleast_1d([v for v in raw_upper if v is not None]) if self.upper_idx else np.array([])

        self.dimension = len(self.lower_idx) + len(self.upper_idx)

        n_elem = index_map.n.state
        parts = []
        if self.lower_idx:
            parts.append(-np.eye(n_elem)[self.lower_idx, :])
        if self.upper_idx:
            parts.append(np.eye(n_elem)[self.upper_idx, :])
        self.M_select = np.vstack(parts) if parts else np.zeros((0, n_elem))
        self._nondim = nondim

    def _d2nd(self, idx):
        return self._nondim.M.state.d2nd[np.ix_(idx, idx)]

    @property
    def lower_value(self):
        return self._d2nd(self.lower_idx) @ self._lower_dim if self.lower_idx else self._lower_dim

    @property
    def upper_value(self):
        return self._d2nd(self.upper_idx) @ self._upper_dim if self.upper_idx else self._upper_dim

    @property
    def rhs(self):
        parts = [v for v in (self.lower_value if self.lower_idx else None,
                             self.upper_value if self.upper_idx else None) if v is not None]
        return np.concatenate(parts) if parts else np.array([])


class initial_state_limits(state_limits):
    def __init__(self, cnstr_config, segment):
        super().__init__(cnstr_config, segment)
        self.type = "initial_state_limits"


class final_state_limits(state_limits):
    def __init__(self, cnstr_config, segment):
        super().__init__(cnstr_config, segment)
        self.type = "final_state_limits"


class control_limits(Constraint):
    def __init__(self, cnstr_config: dict, segment) -> None:
        index_map = segment.index_map
        nondim = segment.nondim

        self.name = cnstr_config.name
        self.type = "control_limits"

        raw_lower = cnstr_config.get("lower", [])
        raw_upper = cnstr_config.get("upper", [])

        self.lower_idx = [i for i, v in enumerate(raw_lower) if v is not None]
        self.upper_idx = [i for i, v in enumerate(raw_upper) if v is not None]
        self._lower_dim = np.atleast_1d([v for v in raw_lower if v is not None]) if self.lower_idx else np.array([])
        self._upper_dim = np.atleast_1d([v for v in raw_upper if v is not None]) if self.upper_idx else np.array([])

        self.dimension = len(self.lower_idx) + len(self.upper_idx)

        n_elem = index_map.n.control
        parts = []
        if self.lower_idx:
            parts.append(-np.eye(n_elem)[self.lower_idx, :])
        if self.upper_idx:
            parts.append(np.eye(n_elem)[self.upper_idx, :])
        self.M_select = np.vstack(parts) if parts else np.zeros((0, n_elem))
        self._nondim = nondim

    @property
    def lower_value(self):
        idx = self.lower_idx
        return self._nondim.M.control.d2nd[np.ix_(idx, idx)] @ self._lower_dim

    @property
    def upper_value(self):
        idx = self.upper_idx
        return self._nondim.M.control.d2nd[np.ix_(idx, idx)] @ self._upper_dim

    @property
    def rhs(self):
        parts = [v for v in (self.lower_value if self.lower_idx else None,
                             self.upper_value if self.upper_idx else None) if v is not None]
        return np.concatenate(parts) if parts else np.array([])


class initial_control_limits(control_limits):
    def __init__(self, cnstr_config, segment):
        super().__init__(cnstr_config, segment)
        self.type = "initial_control_limits"


class final_control_limits(control_limits):
    def __init__(self, cnstr_config, segment):
        super().__init__(cnstr_config, segment)
        self.type = "final_control_limits"


class control_rate_limit(Constraint):
    def __init__(self, cnstr_config: dict, segment) -> None:
        index_map = segment.index_map
        nondim = segment.nondim

        self.type = "control_rate_limit"
        self.name = cnstr_config.name
        self._value_dim = np.atleast_1d(cnstr_config["value"])
        self.idx = cnstr_config.idx
        self.dimension = len(cnstr_config.idx)

        n_elem = index_map.n.control
        M_min = -np.eye(n_elem)[self.idx, :]
        M_max = np.eye(n_elem)[self.idx, :]
        self.M_select = np.vstack([M_min, M_max])
        self._nondim = nondim

    @property
    def value(self):
        nondim = self._nondim
        return nondim.time_scale * nondim.M.control.d2nd[np.ix_(self.idx, self.idx)] @ self._value_dim


class final_time(Constraint):
    def __init__(self, cnstr_config: dict, segment) -> None:
        index_map = segment.index_map
        nondim = segment.nondim

        self.type = "final_time"
        self.name = cnstr_config.name
        self.dimension = 1

        self._lower_dim = cnstr_config.get("lower", None)
        self._upper_dim = cnstr_config.get("upper", None)

        self._N_all = index_map.N.all
        self._nondim = nondim

    @property
    def lower(self):
        return self._lower_dim / self._nondim.time_scale if self._lower_dim is not None else None

    @property
    def upper(self):
        return self._upper_dim / self._nondim.time_scale if self._upper_dim is not None else None

    @property
    def dt_min(self):
        return self.lower / (self._N_all - 1) if self.lower is not None else None

    @property
    def dt_max(self):
        return self.upper / (self._N_all - 1) if self.upper is not None else None


class convex_inequality(Constraint):
    def __init__(self, cnstr_config: dict, segment) -> None:
        index_map = segment.index_map
        nondim = segment.nondim
        fcns = segment.fcns
        params = segment.params

        self.type      = "convex_inequality"
        self.name      = cnstr_config.name
        self._upper_dim = cnstr_config.get("upper", None)
        self._lower_dim = cnstr_config.get("lower", None)
        self.index_map = index_map

        self.fcn_string = cnstr_config.fcn
        self.fcn_xu_dim = tools.resolve_function_from_string(self.fcn_string, fcns)

        out = self.fcn_xu_dim(np.ones((1, index_map.n.state)), np.ones((1, index_map.n.control)), params)
        self.dimension = 1 if out.ndim == 1 else out.shape[1]

        self.scale = np.atleast_1d(cnstr_config.get("scale", np.ones(self.dimension)))
        if self._upper_dim is not None and self._lower_dim is not None:
            self.dimension = 2 * self.dimension

        self.M_state_nd2d = np.asarray(nondim.M.state.nd2d)
        self.M_ctrl_nd2d  = np.asarray(nondim.M.control.nd2d)

        self.M_out_d2nd = np.diag(1.0 / np.abs(self.scale))

    @property
    def upper(self):
        return self.M_out_d2nd @ np.atleast_1d(self._upper_dim) if self._upper_dim is not None else None

    @property
    def lower(self):
        return self.M_out_d2nd @ np.atleast_1d(self._lower_dim) if self._lower_dim is not None else None

    def fcn_txu_nd(self, x, u, t, params):
        M_out_diag = np.diag(self.M_out_d2nd)
        g = cp.multiply(M_out_diag, self.fcn_xu_dim(x @ self.M_state_nd2d.T, u @ self.M_ctrl_nd2d.T, params))
        pieces = []
        if self.lower is not None:
            pieces.append(self.lower - g)
        if self.upper is not None:
            pieces.append(g - self.upper)
        if not pieces:
            pieces.append(g)
        return cp.hstack(pieces)

    def fcn_znu(self, z, nu, params):
        x, t, _, u, _ = self.index_map.unpack_znu(z, nu)
        return self.fcn_txu_nd(x, u, t, params)


class initial_convex_inequality(convex_inequality):
    def __init__(self, cnstr_config, segment):
        super().__init__(cnstr_config, segment)
        self.type = "initial_convex_inequality"


class final_convex_inequality(convex_inequality):
    def __init__(self, cnstr_config, segment):
        super().__init__(cnstr_config, segment)
        self.type = "final_convex_inequality"


class nonconvex_inequality(Constraint):
    def __init__(self, cnstr_config: dict, segment) -> None:
        index_map = segment.index_map
        nondim = segment.nondim
        fcns = segment.fcns
        params = segment.params

        self.type      = "nonconvex_inequality"
        self.name      = cnstr_config.name
        self._upper_dim = cnstr_config.get("upper", None)
        self._lower_dim = cnstr_config.get("lower", None)
        self.index_map = index_map

        self.fcn_string = cnstr_config.fcn
        self.fcn_txu_dim = tools.resolve_function_from_string(self.fcn_string, fcns)

        out = self.fcn_txu_dim(np.ones(index_map.n.state), np.ones(index_map.n.control), 0.0, params)
        self.dimension = jnp.atleast_1d(out).shape[0]

        self.scale = np.atleast_1d(cnstr_config.get("scale", np.ones(self.dimension)))
        if self._upper_dim is not None and self._lower_dim is not None:
            self.dimension = 2 * self.dimension

        self.M_out_d2nd   = jnp.diag(1.0 / jnp.abs(jnp.asarray(self.scale)))
        self.M_state_nd2d = jnp.asarray(nondim.M.state.nd2d)
        self.M_ctrl_nd2d  = jnp.asarray(nondim.M.control.nd2d)

    @property
    def upper(self):
        return self.M_out_d2nd @ jnp.atleast_1d(self._upper_dim) if self._upper_dim is not None else None

    @property
    def lower(self):
        return self.M_out_d2nd @ jnp.atleast_1d(self._lower_dim) if self._lower_dim is not None else None

    def fcn_txu_nd(self, x, u, t, params):
        g = self.M_out_d2nd @ self.fcn_txu_dim(self.M_state_nd2d @ x, self.M_ctrl_nd2d @ u, t, params)
        pieces = []
        if self.lower is not None:
            pieces.append(self.lower - g)
        if self.upper is not None:
            pieces.append(g - self.upper)
        if not pieces:
            pieces.append(g)
        return jnp.concatenate(pieces)

    def fcn_znu(self, z, nu, params):
        x, t, _, u, _ = self.index_map.unpack_znu(z, nu)
        return self.fcn_txu_nd(x, u, t, params)


class initial_nonconvex_inequality(nonconvex_inequality):
    def __init__(self, cnstr_config, segment):
        super().__init__(cnstr_config, segment)
        self.type = "initial_nonconvex_inequality"


class final_nonconvex_inequality(nonconvex_inequality):
    def __init__(self, cnstr_config, segment):
        super().__init__(cnstr_config, segment)
        self.type = "final_nonconvex_inequality"


class ctcs_nonconvex_inequality(nonconvex_inequality):
    def __init__(self, cnstr_config, segment):
        super().__init__(cnstr_config, segment)
        self.type = "ctcs_nonconvex_inequality"


class dynamics(Constraint):
    def __init__(self, cnstr_config: dict, segment) -> None:
        index_map = segment.index_map
        nondim = segment.nondim
        fcns = segment.fcns

        self.type       = "dynamics"
        self.name       = cnstr_config.name
        self.index_map  = index_map
        self.dimension  = index_map.n.z

        self.fcn_string  = cnstr_config.fcn
        self.fcn_txu_dim = tools.resolve_function_from_string(self.fcn_string, fcns)

        self.M_out_d2nd   = jnp.asarray(nondim.M.state.d2nd * nondim.time_scale)
        self.M_state_nd2d = jnp.asarray(nondim.M.state.nd2d)
        self.M_ctrl_nd2d  = jnp.asarray(nondim.M.control.nd2d)

        self.ctcs_constraints = ()

    def fcn_txu_nd(self, x, u, t, params):
        return self.M_out_d2nd @ self.fcn_txu_dim(self.M_state_nd2d @ x, self.M_ctrl_nd2d @ u, t, params)

    def fcn_znu(self, z, nu, params):
        x, t, beta, u, s = self.index_map.unpack_znu(z, nu)

        dx_dt = self.fcn_txu_nd(x, u, t, params)
        dt_dt = jnp.asarray([1.0], dtype=z.dtype)

        if self.ctcs_constraints:
            ctcs_values = jnp.concatenate(
                [jnp.atleast_1d(c.fcn_znu(z, nu, params)) for c in self.ctcs_constraints],
            )
            dbeta_dt = jnp.maximum(ctcs_values, 0.0)
        else:
            dbeta_dt = jnp.zeros_like(beta)

        return s * jnp.concatenate([dx_dt, dt_dt, dbeta_dt])
