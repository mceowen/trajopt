from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from trajopt.methods.scp import convexify
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
        self.value_guess = cnstr_config.get("value_guess")

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
            if self.value_guess is not None:
                self.value_guess = np.asarray(nondim.M.state.d2nd) @ np.atleast_1d(self.value_guess)
        elif self.set == "control":
            self.value = nondim.M.control.d2nd[np.ix_(self.idx, self.idx)] @ self.value
            self.eps = nondim.M.control.d2nd[np.ix_(self.idx, self.idx)] @ self.eps
            if self.value_guess is not None:
                self.value_guess = np.asarray(nondim.M.control.d2nd) @ np.atleast_1d(self.value_guess)


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
        self.value_guess = cnstr_config.get("value_guess")

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
            if self.value_guess is not None:
                self.value_guess = np.asarray(nondim.M.state.d2nd) @ np.atleast_1d(self.value_guess)
        elif self.set == "control":
            self.value = nondim.M.control.d2nd[np.ix_(self.idx, self.idx)] @ self.value
            self.eps = nondim.M.control.d2nd[np.ix_(self.idx, self.idx)] @ self.eps
            if self.value_guess is not None:
                self.value_guess = np.asarray(nondim.M.control.d2nd) @ np.atleast_1d(self.value_guess)


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
    def __init__(self, cnstr_config: dict, index_map: Any, fcns: dict | None = None, **kwargs: Any) -> None:
        """Convex inequality constraint g(t, x, u) <= upper or >= lower."""
        self.name            = cnstr_config["name"]
        self.group           = cnstr_config.get("group", None)
        self.nodes           = cnstr_config.get("nodes", np.arange(0, index_map.N.time_grid))
        self.upper       = cnstr_config.get("upper", None)
        self.lower       = cnstr_config.get("lower", None)
        self.upper_and_lower = (self.upper is not None) and (self.lower is not None)

        self.fcn_string      = cnstr_config["fcn"]
        self.index_map       = index_map
        self.type            = 'convex_inequality'

        self.fcn_dim = tools.resolve_function_from_string(self.fcn_string, fcns)

        self.dimension = cnstr_config["dimension"]

        self.eps   = np.atleast_1d(cnstr_config.get("eps", np.full(self.dimension, 0.0001)))
        self.scale = cnstr_config.get("scale", None)
        if self.scale is None and self.upper is None and self.lower is None:
            self.scale = np.ones(self.dimension).tolist()

        if self.upper_and_lower:
            self.dimension = 2 * self.dimension
        self.fcn_nd = None

        self.fcn = None

        self.fcn_compiled = None
        self.dfcn_dz_compiled = None
        self.dfcn_du_compiled = None

    def nondim_constraint(self, nondim):
        if self.upper is not None:
            self.upper = jnp.asarray(jnp.atleast_1d(self.upper))

        if self.lower is not None:
            self.lower = jnp.asarray(jnp.atleast_1d(self.lower))

        if self.scale is not None:
            self.M_out_d2nd = jnp.diag(1 / jnp.abs(jnp.asarray(self.scale)))
        elif self.upper is not None:
            self.M_out_d2nd = jnp.diag(1 / jnp.abs(self.upper))
        elif self.lower is not None:
            self.M_out_d2nd = jnp.diag(1 / jnp.abs(self.lower))
        else:
            raise ValueError("At least one of 'scales', 'upper', or 'lower' must be provided for nondimensionalization.")

        self.M_out_nd2d = jnp.diag(1 / jnp.diag(self.M_out_d2nd))

        if self.upper is not None:
            self.upper = self.M_out_d2nd @ self.upper

        if self.lower is not None:
            self.lower = self.M_out_d2nd @ self.lower

        M_state_nd2d_jax = jnp.asarray(nondim.M.state.nd2d)
        M_ctrl_nd2d_jax  = jnp.asarray(nondim.M.control.nd2d)

        import inspect
        sig = inspect.signature(self.fcn_dim)
        if len(sig.parameters) >= 4:
            self.fcn_nd = nondim.nondim_function(self.fcn_dim, M_state_nd2d_jax, M_ctrl_nd2d_jax, self.M_out_d2nd)

        self.eps = self.M_out_d2nd @ self.eps

        if self.upper_and_lower:
            self.eps = jnp.concatenate((self.eps, self.eps))

# ===============================================================
# NONCONVEX CONSTRAINTS
# ===============================================================

class nonconvex_inequality:
    def __init__(self, cnstr_config: dict, index_map: Any, fcns: dict | None = None, **kwargs: Any) -> None:
        """Nonconvex inequality constraint g(t, x, u) <= upper or >= lower."""
        self.name           = cnstr_config["name"]
        self.group          = cnstr_config.get("group", None)
        self.nodes          = cnstr_config.get("nodes", np.arange(0, index_map.N.time_grid))
        self.ct             = cnstr_config.get("ct", 0)
        self.hard           = cnstr_config.get("hard", 0)
        self.backend        = cnstr_config.get("backend", "jax")
        self.upper      = cnstr_config.get("upper", None)
        self.lower      = cnstr_config.get("lower", None)
        self.upper_and_lower = (self.upper is not None) and (self.lower is not None)

        self.fcn_string     = cnstr_config["fcn"]
        self.index_map      = index_map
        self.type           = 'nonconvex_inequality'

        self.fcn_dim = tools.resolve_function_from_string(self.fcn_string, fcns)

        self.dimension = cnstr_config["dimension"]

        self.eps   = np.atleast_1d(cnstr_config.get("eps", np.full(self.dimension, 0.0001)))
        self.scale = cnstr_config.get("scale", None)
        if self.scale is None and self.upper is None and self.lower is None:
            self.scale = np.ones(self.dimension).tolist()

        if self.upper_and_lower:
            self.dimension = 2 * self.dimension
        self.fcn_nd = None

        # this is the symbolic nondimmed version of fcn_fim, it will
        # be provided once the nondim_constraint() function is called
        self.fcn = None

        # the compiled version (jitted for jax / numpy for sympy)
        # will be provided by problem.constraints.convexify_constraints()
        self.fcn_compiled = None
        self.dfcn_dz_compiled = None
        self.dfcn_du_compiled = None

    def nondim_constraint(self, nondim):
        if self.upper is not None:
            self.upper = jnp.asarray(jnp.atleast_1d(self.upper))

        if self.lower is not None:
            self.lower = jnp.asarray(jnp.atleast_1d(self.lower))

        # get provided scales if provided
        if self.scale is not None:
            # TODO (Carlos): revisit, temporary scaling if max value is equal to zero
            self.M_out_d2nd = jnp.diag(1 / jnp.abs(jnp.asarray(self.scale)))
        elif self.upper is not None:
            self.M_out_d2nd = jnp.diag(1 / jnp.abs(self.upper))
        elif self.lower is not None:
            self.M_out_d2nd = jnp.diag(1 / jnp.abs(self.lower))
        else:
            raise ValueError("At least one of 'scales', 'upper', or 'lower' must be provided for nondimensionalization.")

        self.M_out_nd2d = jnp.diag(1 / jnp.diag(self.M_out_d2nd))

        if self.upper is not None:
            self.upper = self.M_out_d2nd @ self.upper

        if self.lower is not None:
            self.lower = self.M_out_d2nd @ self.lower

        M_state_nd2d_jax = jnp.asarray(nondim.M.state.nd2d)
        M_ctrl_nd2d_jax  = jnp.asarray(nondim.M.control.nd2d)

        self.fcn_nd = nondim.nondim_function(self.fcn_dim, M_state_nd2d_jax, M_ctrl_nd2d_jax, self.M_out_d2nd)

        self.eps = self.M_out_d2nd @ self.eps

        if self.upper_and_lower:
            self.eps = jnp.concatenate((self.eps, self.eps))

    def convexify_constraint(self):
        if self.upper_and_lower:
            fcn_lb_txu = lambda t, x, u, params: -self.fcn_nd(t, x, u, params) + self.lower
            fcn_ub_txu = lambda t, x, u, params:  self.fcn_nd(t, x, u, params) - self.upper
            fcn_txu = lambda t, x, u, params: jnp.concatenate([fcn_lb_txu(t, x, u, params), fcn_ub_txu(t, x, u, params)])
        elif (self.upper is not None) and (self.lower is None):
            fcn_txu = lambda t, x, u, params: self.fcn_nd(t, x, u, params) - self.upper
        elif (self.lower is not None) and (self.upper is None):
            fcn_txu = lambda t, x, u, params: -self.fcn_nd(t, x, u, params) + self.lower
        else:
            fcn_txu = self.fcn_nd

        self.fcn = self.index_map.wrap_txu_fcn(fcn_txu)

        vals_function_txu = lambda t, z, nu, params: self.M_out_nd2d @ (self.fcn_nd(t, z, nu, params))
        vals_function = self.index_map.wrap_txu_fcn(vals_function_txu)

        self.fcn_compiled, self.dfcn_dz_compiled, self.dfcn_du_compiled = convexify.linearize_jax(self.fcn)

        self.f_batched       = jax.jit(jax.vmap(vals_function,         in_axes=(0, 0, None)))
        self.fcn_batched     = jax.jit(jax.vmap(self.fcn_compiled,     in_axes=(0, 0, None)))
        self.dfcn_dz_batched = jax.jit(jax.vmap(self.dfcn_dz_compiled, in_axes=(0, 0, None)))
        self.dfcn_du_batched = jax.jit(jax.vmap(self.dfcn_du_compiled, in_axes=(0, 0, None)))

    def g_aff(self, z: Any, nu: Any, params: Any) -> tuple:
        """Evaluate linearized (affine) constraint at a single point.

        Args:
            z: Augmented state vector.
            nu: Augmented control vector.
            params: Problem parameters.

        Returns:
            Tuple of (g, dg_dz, dg_dnu).

        """
        return (
            self.fcn_compiled(z, nu, params),
            self.dfcn_dz_compiled(z, nu, params),
            self.dfcn_du_compiled(z, nu, params),
        )

    def g_aff_batched(self, z: Any, nu: Any, params: Any) -> tuple:
        """Evaluate linearized (affine) constraint over a trajectory batch.

        Args:
            z: Augmented state trajectory array.
            nu: Augmented control trajectory array.
            params: Problem parameters.

        Returns:
            Tuple of batched (g, dg_dz, dg_dnu).

        """
        return (
            self.fcn_batched(z, nu, params),
            self.dfcn_dz_batched(z, nu, params),
            self.dfcn_du_batched(z, nu, params),
        )

class dynamics:
    def __init__(self, cnstr_config: dict, index_map: Any, fcns: dict | None = None, **kwargs: Any) -> None:
        """Dynamics constraint wrapping the physical equations of motion.

        Args:
            cnstr_config: Constraint configuration dictionary.
            index_map: Index map object.
            fcns: Resolved functions dictionary.
            **kwargs: Additional keyword arguments (unused).

        """
        self.name = cnstr_config["name"]
        self.fcn_string = cnstr_config["fcn"]
        self.nodes = cnstr_config.get("nodes", np.arange(0, index_map.N.time_grid))
        self.index_map = index_map
        self.dimension = index_map.n.z
        self.group = "dynamics"
        self.type = "dynamics"

        self.index_map = index_map

        self.fcn_dim = tools.resolve_function_from_string(self.fcn_string, fcns)

        self.backend = cnstr_config.get("backend", "jax")

        self.fcn = None
        self.fcn_base = None
        self.fcn_compiled = None
        self.dfcn_dz_compiled = None
        self.dfcn_du_compiled = None

    def nondim_constraint(self, nondim: Any) -> None:
        """Non-dimensionalize the dynamics function.

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.

        """
        if self.backend == "jax":
            M_out_d2nd_jax = jnp.asarray(nondim.M.state.d2nd * nondim.time_scale)
            M_state_nd2d_jax = jnp.asarray(nondim.M.state.nd2d)
            M_ctrl_nd2d_jax = jnp.asarray(nondim.M.control.nd2d)

        elif self.backend == "sympy":
            pass  # symbolic differentiation

        elif self.backend == "numpy":
            pass  # analytical

        # fcn_dim is already bound with params/fcns by resolve_functions
        # fcn_base is f(t, x, u, params), - > fcn is f(z, nu, params)
        self.fcn_base = nondim.nondim_function(self.fcn_dim, M_state_nd2d_jax, M_ctrl_nd2d_jax, M_out_d2nd_jax)

    def convexify_constraint(self) -> None:
        """Compile and JIT-compile the dynamics constraint and its Jacobians for use in SCP.

        Args:
            None.

        Returns:
            None.

        """
        if self.backend == "jax":
            self.fcn_compiled, self.dfcn_dz_compiled, self.dfcn_du_compiled = convexify.linearize_jax(self.fcn)

        elif self.backend == "sympy":
            pass  # symbolic differentiation

        elif self.backend == "numpy":
            pass  # analytical

    def lin_dyn(self, z: Any, nu: Any, params: Any) -> tuple:
        """Evaluate linearized dynamics at a single point.

        Args:
            z: Augmented state vector.
            nu: Augmented control vector.
            params: Problem parameters.

        Returns:
            Tuple of (f, df_dz, df_dnu).

        """
        return (
            self.fcn_compiled(z, nu, params),
            self.dfcn_dz_compiled(z, nu, params),
            self.dfcn_du_compiled(z, nu, params),
        )