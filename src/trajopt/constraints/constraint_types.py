from typing import Any

import cvxpy as cp
import jax.numpy as jnp
import numpy as np

from trajopt.utils import tools

# ===============================================================
# CONVEX CONSTRAINTS
# ===============================================================

constraint_types = [
    "initial_state",
    "final_state",
    "state_limits",
    "control_limits",
    "final_time",
    "convex_inequality",
    "nonconvex_inequality",
    "dynamics",
]

class initial_state:
    def __init__(self, cnstr_config: dict, index_map: Any, **kwargs: Any) -> None:
        """Initial state boundary constraint. Null entries in value are unconstrained."""
        self.name = cnstr_config["name"]
        self.type = "initial_state"
        self.set = cnstr_config.get("set", "state")
        self.node = cnstr_config.get("node", 0)
        self.nodes = cnstr_config.get("nodes", np.array([self.node]))
        self.ct = cnstr_config.get("ct", 0)
        self.group = cnstr_config.get("group")

        raw_value = cnstr_config["value"]

        if "idx" in cnstr_config:
            self.idx = cnstr_config["idx"]
            self.value = np.atleast_1d(raw_value)
        else:
            self.idx = [i for i, v in enumerate(raw_value) if v is not None]
            self.value = np.atleast_1d([v for v in raw_value if v is not None])

        self.dimension = len(self.idx)
        self.eps = np.atleast_1d(cnstr_config.get("eps", np.full(self.dimension, 0.0001)))

    def nondim_constraint(self, nondim: Any) -> None:
        if self.set == "state":
            self.value = nondim.M.state.d2nd[np.ix_(self.idx, self.idx)] @ self.value
            self.eps = nondim.M.state.d2nd[np.ix_(self.idx, self.idx)] @ self.eps
        elif self.set == "control":
            self.value = nondim.M.control.d2nd[np.ix_(self.idx, self.idx)] @ self.value
            self.eps = nondim.M.control.d2nd[np.ix_(self.idx, self.idx)] @ self.eps


class final_state:
    def __init__(self, cnstr_config: dict, index_map: Any, **kwargs: Any) -> None:
        """Final state boundary constraint. Null entries in value are unconstrained."""
        self.name = cnstr_config["name"]
        self.type = "final_state"
        self.set = cnstr_config.get("set", "state")
        self.node = cnstr_config.get("node", -1)
        self.nodes = cnstr_config.get("nodes", np.array([index_map.N.time_grid - 1]))
        self.ct = cnstr_config.get("ct", 0)
        self.group = cnstr_config.get("group")

        raw_value = cnstr_config["value"]

        if "idx" in cnstr_config:
            self.idx = cnstr_config["idx"]
            self.value = np.atleast_1d(raw_value)
        else:
            self.idx = [i for i, v in enumerate(raw_value) if v is not None]
            self.value = np.atleast_1d([v for v in raw_value if v is not None])

        self.dimension = len(self.idx)
        self.eps = np.atleast_1d(cnstr_config.get("eps", np.full(self.dimension, 0.0001)))

    def nondim_constraint(self, nondim: Any) -> None:
        if self.set == "state":
            self.value = nondim.M.state.d2nd[np.ix_(self.idx, self.idx)] @ self.value
            self.eps = nondim.M.state.d2nd[np.ix_(self.idx, self.idx)] @ self.eps
        elif self.set == "control":
            self.value = nondim.M.control.d2nd[np.ix_(self.idx, self.idx)] @ self.value
            self.eps = nondim.M.control.d2nd[np.ix_(self.idx, self.idx)] @ self.eps


class state_limits:
    def __init__(self, cnstr_config: dict, index_map: Any, **kwargs: Any) -> None:
        """State box constraints. Null entries in lower/upper are unconstrained."""
        self.name = cnstr_config["name"]
        self.type = "state_limits"
        self.set = "state"
        self.ct = cnstr_config.get("ct", 0)
        self.group = cnstr_config.get("group")
        self.nodes = cnstr_config.get("nodes", np.arange(0, index_map.N.time_grid))

        raw_lower = cnstr_config.get("lower", [])
        raw_upper = cnstr_config.get("upper", [])

        self.lower_idx = [i for i, v in enumerate(raw_lower) if v is not None]
        self.upper_idx = [i for i, v in enumerate(raw_upper) if v is not None]
        self.lower_value = np.atleast_1d([v for v in raw_lower if v is not None]) if self.lower_idx else np.array([])
        self.upper_value = np.atleast_1d([v for v in raw_upper if v is not None]) if self.upper_idx else np.array([])

        self.dimension = len(self.lower_idx) + len(self.upper_idx)

        n_elem = index_map.n.state
        parts = []
        if self.lower_idx:
            parts.append(-np.eye(n_elem)[self.lower_idx, :])
        if self.upper_idx:
            parts.append(np.eye(n_elem)[self.upper_idx, :])
        self.M_select = np.vstack(parts) if parts else np.zeros((0, n_elem))

        rhs_parts = []
        if self.lower_idx:
            rhs_parts.append(self.lower_value)
        if self.upper_idx:
            rhs_parts.append(self.upper_value)
        self.rhs = np.concatenate(rhs_parts) if rhs_parts else np.array([])

    def nondim_constraint(self, nondim: Any) -> None:
        if self.lower_idx:
            self.lower_value = nondim.M.state.d2nd[np.ix_(self.lower_idx, self.lower_idx)] @ self.lower_value
        if self.upper_idx:
            self.upper_value = nondim.M.state.d2nd[np.ix_(self.upper_idx, self.upper_idx)] @ self.upper_value

        rhs_parts = []
        if self.lower_idx:
            rhs_parts.append(self.lower_value)
        if self.upper_idx:
            rhs_parts.append(self.upper_value)
        self.rhs = np.concatenate(rhs_parts) if rhs_parts else np.array([])


class control_limits:
    def __init__(self, cnstr_config: dict, index_map: Any, **kwargs: Any) -> None:
        """Control box constraints. Null entries in lower/upper are unconstrained."""
        self.name = cnstr_config["name"]
        self.type = "control_limits"
        self.set = "control"
        self.ct = cnstr_config.get("ct", 0)
        self.group = cnstr_config.get("group")
        self.nodes = cnstr_config.get("nodes", np.arange(0, index_map.N.time_grid))

        raw_lower = cnstr_config.get("lower", [])
        raw_upper = cnstr_config.get("upper", [])

        self.lower_idx = [i for i, v in enumerate(raw_lower) if v is not None]
        self.upper_idx = [i for i, v in enumerate(raw_upper) if v is not None]
        self.lower_value = np.atleast_1d([v for v in raw_lower if v is not None]) if self.lower_idx else np.array([])
        self.upper_value = np.atleast_1d([v for v in raw_upper if v is not None]) if self.upper_idx else np.array([])

        self.dimension = len(self.lower_idx) + len(self.upper_idx)

        n_elem = index_map.n.control
        parts = []
        if self.lower_idx:
            parts.append(-np.eye(n_elem)[self.lower_idx, :])
        if self.upper_idx:
            parts.append(np.eye(n_elem)[self.upper_idx, :])
        self.M_select = np.vstack(parts) if parts else np.zeros((0, n_elem))

        rhs_parts = []
        if self.lower_idx:
            rhs_parts.append(self.lower_value)
        if self.upper_idx:
            rhs_parts.append(self.upper_value)
        self.rhs = np.concatenate(rhs_parts) if rhs_parts else np.array([])

    def nondim_constraint(self, nondim: Any) -> None:
        if self.lower_idx:
            self.lower_value = nondim.M.control.d2nd[np.ix_(self.lower_idx, self.lower_idx)] @ self.lower_value
        if self.upper_idx:
            self.upper_value = nondim.M.control.d2nd[np.ix_(self.upper_idx, self.upper_idx)] @ self.upper_value

        rhs_parts = []
        if self.lower_idx:
            rhs_parts.append(self.lower_value)
        if self.upper_idx:
            rhs_parts.append(self.upper_value)
        self.rhs = np.concatenate(rhs_parts) if rhs_parts else np.array([])

# ---------------------------------------------------------------
# rate constraints
# ---------------------------------------------------------------
class control_rate_limit:
    def __init__(self, cnstr_config: dict, index_map: Any, **kwargs: Any) -> None:
        """Rate limit constraint on selected control channels.

        Args:
            cnstr_config: Constraint configuration dictionary.
            index_map: Index map object.
            **kwargs: Additional keyword arguments (unused).

        """
        self.name = cnstr_config["name"]
        self.group = cnstr_config.get("group")
        self.nodes = cnstr_config.get("nodes", np.arange(0, index_map.N.time_grid))
        self.value = np.atleast_1d(cnstr_config["value"])
        self.idx = cnstr_config["idx"]
        self.dimension = len(cnstr_config["idx"])
        self.type = "control_rate_limit"
        self.ct = cnstr_config.get("ct", 0)

        n_elem = index_map.n.control
        M_min = -np.eye(n_elem)[self.idx, :]
        M_max = np.eye(n_elem)[self.idx, :]
        self.M_select = np.vstack([M_min, M_max])

    def nondim_constraint(self, nondim: Any) -> None:
        """Non-dimensionalize the rate limit value.

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.

        """
        self.value = nondim.time_scale * nondim.M.control.d2nd[np.ix_(self.idx, self.idx)] @ self.value

# ---------------------------------------------------------------
# time limits
# ---------------------------------------------------------------
class final_time:
    def __init__(self, cnstr_config: dict, index_map: Any, **kwargs: Any) -> None:
        """Total flight-time box constraint: lower <= T_final <= upper.

        Nondimensionalisation is handled here so the SCP method
        does not need to carry raw dimensional T_min / T_max values.
        """
        self.name = cnstr_config["name"]
        self.type = "final_time"
        self.ct = 0
        self.group = cnstr_config.get("group")
        self.nodes = np.array([index_map.N.time_grid - 1])
        self.dimension = 1

        self.lower = cnstr_config.get("lower", None)
        self.upper = cnstr_config.get("upper", None)

        self._N_time_grid = index_map.N.time_grid

        self.dt_min = None
        self.dt_max = None

    def nondim_constraint(self, nondim: Any) -> None:
        N = self._N_time_grid
        if self.lower is not None:
            self.lower = self.lower / nondim.time_scale
            self.dt_min = self.lower / (N - 1)
        if self.upper is not None:
            self.upper = self.upper / nondim.time_scale
            self.dt_max = self.upper / (N - 1)


# ===============================================================
# CONVEX inequality
# ===============================================================

class convex_inequality:
    def __init__(self, cnstr_config: dict, index_map: Any, fcns: dict | None = None, params: Any = None, **kwargs: Any) -> None:
        """Convex inequality constraint g(t, x, u) <= upper and/or >= lower."""
        self.name      = cnstr_config["name"]
        self.group     = cnstr_config.get("group", None)
        self.nodes     = cnstr_config.get("nodes", np.arange(0, index_map.N.time_grid))
        self.upper     = cnstr_config.get("upper", None)
        self.lower     = cnstr_config.get("lower", None)
        self.type      = "convex_inequality"
        self.ct        = cnstr_config.get("ct", 0)
        self.index_map = index_map

        self.fcn_string = cnstr_config["fcn"]
        self.fcn_dim = tools.resolve_function_from_string(self.fcn_string, fcns)

        out = self.fcn_dim(np.ones((1, index_map.n.state)), np.ones((1, index_map.n.control)), params)
        self.dimension = 1 if out.ndim == 1 else out.shape[1]

        self.scale = np.atleast_1d(cnstr_config.get("scale", np.ones(self.dimension)))
        self.eps   = np.atleast_1d(cnstr_config.get("eps", np.full(self.dimension, 0.0001)))

        if self.upper is not None and self.lower is not None:
            self.dimension = 2 * self.dimension

        self.fcn = None

    def nondim_constraint(self, nondim):
        M_out = np.diag(1.0 / np.abs(self.scale))

        if self.upper is not None:
            self.upper = M_out @ np.atleast_1d(self.upper)
        if self.lower is not None:
            self.lower = M_out @ np.atleast_1d(self.lower)
        self.eps = M_out @ self.eps
        if self.upper is not None and self.lower is not None:
            self.eps = np.concatenate((self.eps, self.eps))

        M_state_T  = np.asarray(nondim.M.state.nd2d).T
        M_ctrl_T   = np.asarray(nondim.M.control.nd2d).T
        M_out_diag = np.diag(M_out)
        raw_fcn    = self.fcn_dim
        upper, lower = self.upper, self.lower

        def fcn(x, u, params):
            g = cp.multiply(M_out_diag, raw_fcn(x @ M_state_T, u @ M_ctrl_T, params))
            pieces = []
            if lower is not None:
                pieces.append(lower - g)
            if upper is not None:
                pieces.append(g - upper)
            if not pieces:
                pieces.append(g)
            return cp.hstack(pieces)

        self.fcn = fcn

# ===============================================================
# NONCONVEX CONSTRAINTS
# ===============================================================

class nonconvex_inequality:
    def __init__(self, cnstr_config: dict, index_map: Any, fcns: dict | None = None, params: Any = None, **kwargs: Any) -> None:
        """Nonconvex inequality constraint g(t, x, u) <= upper and/or >= lower."""
        self.name      = cnstr_config["name"]
        self.group     = cnstr_config.get("group", None)
        self.nodes     = cnstr_config.get("nodes", np.arange(0, index_map.N.time_grid))
        self.ct        = cnstr_config.get("ct", 0)
        self.hard      = cnstr_config.get("hard", 0)
        self.upper     = cnstr_config.get("upper", None)
        self.lower     = cnstr_config.get("lower", None)
        self.type      = "nonconvex_inequality"
        self.index_map = index_map

        self.fcn_string = cnstr_config["fcn"]
        self.fcn_dim = tools.resolve_function_from_string(self.fcn_string, fcns)

        out = self.fcn_dim(0.0, np.ones(index_map.n.state), np.ones(index_map.n.control), params)
        self.dimension = jnp.atleast_1d(out).shape[0]

        self.scale = np.atleast_1d(cnstr_config.get("scale", np.ones(self.dimension)))
        self.eps   = np.atleast_1d(cnstr_config.get("eps", np.full(self.dimension, 0.0001)))

        if self.upper is not None and self.lower is not None:
            self.dimension = 2 * self.dimension

        # fcn_txu_nd(t, x, u, params): non-dimensional constraint, built by nondim_constraint
        self.fcn_txu_nd = None
        # fcn_znu(z, nu, params):      augmented-vector form, built by augment_constraint
        self.fcn_znu = None

    def nondim_constraint(self, nondim):
        """Create the non-dimensional t-x-u constraint function self.fcn_txu_nd(t, x, u, params)."""
        M_out = jnp.diag(1.0 / jnp.abs(jnp.asarray(self.scale)))

        if self.upper is not None:
            self.upper = M_out @ jnp.atleast_1d(self.upper)
        if self.lower is not None:
            self.lower = M_out @ jnp.atleast_1d(self.lower)
        self.eps = M_out @ self.eps
        if self.upper is not None and self.lower is not None:
            self.eps = jnp.concatenate((self.eps, self.eps))

        M_state = jnp.asarray(nondim.M.state.nd2d)
        M_ctrl  = jnp.asarray(nondim.M.control.nd2d)
        fcn_nd  = nondim.nondim_function(self.fcn_dim, M_state, M_ctrl, M_out)
        upper, lower = self.upper, self.lower

        def fcn_txu_nd(t, x, u, params):
            g = fcn_nd(t, x, u, params)
            pieces = []
            if lower is not None:
                pieces.append(lower - g)
            if upper is not None:
                pieces.append(g - upper)
            if not pieces:
                pieces.append(g)
            return jnp.concatenate(pieces)

        self.fcn_txu_nd = fcn_txu_nd

    def augment_constraint(self):
        """Wrap the non-dimensional t-x-u function into augmented-vector form self.fcn_znu(z, nu, params)."""
        self.fcn_znu = self.index_map.wrap_txu_fcn(self.fcn_txu_nd)


class dynamics:
    def __init__(self, cnstr_config: dict, index_map: Any, fcns: dict | None = None, **kwargs: Any) -> None:
        """Dynamics constraint wrapping the physical equations of motion."""
        self.name       = cnstr_config["name"]
        self.type       = "dynamics"
        self.group      = "dynamics"
        self.nodes      = cnstr_config.get("nodes", np.arange(0, index_map.N.time_grid - 1))
        self.index_map  = index_map
        self.dimension  = index_map.n.z

        self.fcn_string = cnstr_config["fcn"]
        self.fcn_dim = tools.resolve_function_from_string(self.fcn_string, fcns)

        # fcn_txu_nd(t, x, u, params): non-dimensional physical dynamics, built by nondim_constraint
        self.fcn_txu_nd = None
        # fcn_znu(z, nu, params):      physical dynamics in augmented-vector form, built by augment_constraint
        self.fcn_znu = None
        # fcn(z, nu, params):          full augmented dynamics zdot = [xdot, 1, dbeta], built by Constraints.augment
        self.fcn = None

    def nondim_constraint(self, nondim: Any) -> None:
        """Create the non-dimensional physical dynamics self.fcn_txu_nd(t, x, u, params)."""
        M_out_d2nd = jnp.asarray(nondim.M.state.d2nd * nondim.time_scale)
        M_state    = jnp.asarray(nondim.M.state.nd2d)
        M_ctrl     = jnp.asarray(nondim.M.control.nd2d)
        self.fcn_txu_nd = nondim.nondim_function(self.fcn_dim, M_state, M_ctrl, M_out_d2nd)

    def augment_constraint(self) -> None:
        """Wrap the non-dimensional physical dynamics into augmented-vector form self.fcn_znu(z, nu, params)."""
        self.fcn_znu = self.index_map.wrap_txu_fcn(self.fcn_txu_nd)