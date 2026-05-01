from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from trajopt.methods.scp import convexify
from trajopt.utils import tools

# ===============================================================
# CONVEX CONSTRAINTS
# ===============================================================

class equality_bc:
    def __init__(self, cnstr_config: dict, index_map: Any, **kwargs: Any) -> None:
        """Equality boundary condition constraint on state or control.

        Args:
            cnstr_config: Constraint configuration dictionary.
            index_map: Index map object.
            **kwargs: Additional keyword arguments (unused).

        """
        # required properties
        self.name = cnstr_config["name"]
        self.group = cnstr_config.get("group")
        self.value_guess = cnstr_config.get("value_guess")
        self.ct = cnstr_config.get("ct", 0)

        # type-specific properties
        self.set = cnstr_config["set"]
        self.value = np.atleast_1d(cnstr_config["value"])

        self.idx = cnstr_config["idx"]
        self.boundary = cnstr_config["boundary"]

        if self.boundary == "init":
            self.boundary_idx = 0
        elif self.boundary == "final":
            self.boundary_idx = -1
        else:
            self.boundary_idx = None

        self.eps = cnstr_config.get("eps", np.zeros(len(self.idx)))
        self.dimension = len(self.idx)

        self.type = "equality_bc"

    def fcn(self, x: np.ndarray) -> np.ndarray:
        """Evaluate equality boundary condition residual.

        Args:
            x: State vector.

        Returns:
            Residual x[idx] - value.

        """
        return x[self.idx] - self.value

    def nondim_constraint(self, nondim: Any) -> None:
        """Non-dimensionalize the boundary condition value and tolerance.

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.

        """
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


class inequality_bc:
    def __init__(self, cnstr_config: dict, index_map: Any, **kwargs: Any) -> None:
        """Inequality boundary condition constraint on state or control.

        Args:
            cnstr_config: Constraint configuration dictionary.
            index_map: Index map object.
            **kwargs: Additional keyword arguments (unused).

        """
        self.name = cnstr_config["name"]
        self.group = cnstr_config.get("group")
        self.set = cnstr_config["set"]
        self.ct = cnstr_config.get("ct", 0)

        self.min_value_dim = np.atleast_1d(cnstr_config["min_value"])
        self.min_value_idx = cnstr_config["min_value_idx"]
        self.max_value_dim = np.atleast_1d(cnstr_config["max_value"])
        self.max_value_idx = cnstr_config["max_value_idx"]

        self.boundary = cnstr_config["boundary"]
        self.idx = 0 if self.boundary == "init" else -1 if self.boundary == "final" else None
        self.eps = cnstr_config["eps"]
        self.dimension = len(self.min_value_idx) + len(self.max_value_idx)
        self.min_value = None
        self.max_value = None
        self.type = "inequality_bc"

    def nondim_constraint(self, nondim: Any) -> None:
        """Non-dimensionalize the boundary inequality bounds.

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.

        """
        if self.set == "state":
            self.min_value = nondim.M.state["d2nd"][np.ix_(self.min_value_idx, self.min_value_idx)] @ self.min_value
            self.max_value = nondim.M.state["d2nd"][np.ix_(self.max_value_idx, self.max_value_idx)] @ self.max_value
        elif self.set == "control":
            self.min_value = nondim.M.control["d2nd"][np.ix_(self.min_value_idx, self.min_value_idx)] @ self.min_value
            self.max_value = nondim.M.control["d2nd"][np.ix_(self.max_value_idx, self.max_value_idx)] @ self.max_value


class box:
    def __init__(self, cnstr_config: dict, index_map: Any, **kwargs: Any) -> None:
        """Box (interval) constraint on state or control at each node.

        Args:
            cnstr_config: Constraint configuration dictionary.
            index_map: Index map object.
            **kwargs: Additional keyword arguments (unused).

        """
        self.name = cnstr_config["name"]
        self.group = cnstr_config.get("group")
        self.set = cnstr_config["set"]
        self.ct = cnstr_config.get("ct", 0)

        self.min_value = cnstr_config["min_value"]
        self.max_value = cnstr_config["max_value"]
        self.min_value_idx = cnstr_config["min_value_idx"]
        self.max_value_idx = cnstr_config["max_value_idx"]
        self.dimension = len(self.min_value_idx) + len(self.max_value_idx)
        self.type = "box"

        self.index_map = index_map

        if self.set == "state":
            n_elem = index_map.n.state
        elif self.set == "control":
            n_elem = index_map.n.control

        M_min = -np.eye(n_elem)[self.min_value_idx, :]
        M_max = np.eye(n_elem)[self.max_value_idx, :]
        self.M_select = np.vstack([M_min, M_max])

    def nondim_constraint(self, nondim: Any) -> None:
        """Non-dimensionalize the box constraint bounds.

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.

        """
        if self.set == "state":
            self.max_value = nondim.M.state["d2nd"][np.ix_(self.max_value_idx, self.max_value_idx)] @ self.max_value
            self.min_value = nondim.M.state["d2nd"][np.ix_(self.min_value_idx, self.min_value_idx)] @ self.min_value
        elif self.set == "control":
            self.min_value = nondim.M.control["d2nd"][np.ix_(self.min_value_idx, self.min_value_idx)] @ self.min_value
            self.max_value = nondim.M.control["d2nd"][np.ix_(self.max_value_idx, self.max_value_idx)] @ self.max_value

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
# Second-order cone constraints
# ---------------------------------------------------------------

class axis_angle_cone:
    def __init__(self, cnstr_config: dict, index_map: Any, **kwargs: Any) -> None:
        """Axis-angle cone constraint bounding the angle between a vector and a reference axis.

        Args:
            cnstr_config: Constraint configuration dictionary.
            index_map: Index map object.
            **kwargs: Additional keyword arguments (unused).

        """
        self.name = cnstr_config["name"]
        self.group = cnstr_config.get("group")
        self.set = cnstr_config["set"]
        self.axis = cnstr_config["axis"] / np.linalg.norm(cnstr_config["axis"])
        self.theta_max = cnstr_config["theta_max"]
        self.cos_theta_max = np.cos(np.deg2rad(self.theta_max))
        self.idx = cnstr_config["idx"]
        self.dimension = 1
        self.ct = cnstr_config.get("ct", 0)
        self.type = "axis_angle_cone"

    def nondim_constraint(self, nondim: Any) -> None:
        """Non-dimensionalize the axis-angle cone constraint (no-op; angles are dimensionless).

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.

        """


class max_norm_cone:
    def __init__(self, cnstr_config: dict, index_map: Any, **kwargs: Any) -> None:
        """Maximum Euclidean norm constraint on selected state or control indices.

        Args:
            cnstr_config: Constraint configuration dictionary.
            index_map: Index map object.
            **kwargs: Additional keyword arguments (unused).

        """
        self.name = cnstr_config["name"]
        self.group = cnstr_config.get("group")
        self.set = cnstr_config["set"]
        self.max_value = cnstr_config["max_value"]
        self.idx = cnstr_config["idx"]
        self.dimension = 1
        self.ct = cnstr_config.get("ct", 0)
        self.type = "max_norm_cone"

    def nondim_constraint(self, nondim: Any) -> None:
        """Non-dimensionalize the maximum norm value.

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.

        """
        if self.set == "state" or self.set == "control":
            self.max_value = self.max_value * nondim.M.state.d2nd[self.idx[0], self.idx[0]]

class quaternion_cone:
    def __init__(self, cnstr_config: dict, index_map: Any, **kwargs: Any) -> None:
        """Quaternion cone constraint bounding the rotation angle about a specified axis.

        Args:
            cnstr_config: Constraint configuration dictionary.
            index_map: Index map object.
            **kwargs: Additional keyword arguments (unused).

        """
        self.name = cnstr_config["name"]
        self.group = None
        self.ct = cnstr_config.get("ct", 0)

        self.quat_start_idx = cnstr_config["quat_start_idx"]
        self.cos_theta_max = np.cos(np.deg2rad(cnstr_config["theta_max"]))
        self.axis_num = cnstr_config["axis_num"]
        self.rhs = np.sqrt((1.0 - self.cos_theta_max) * 0.5)

        self.dimension = 1
        self.type = "quaternion_cone"

    def nondim_constraint(self, nondim: Any) -> None:
        """Non-dimensionalize the quaternion cone constraint (no-op; angles are dimensionless).

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.

        """


# ===============================================================
# NONCONVEX CONSTRAINTS
# ===============================================================

class nonconvex_inequality:
    def __init__(self, cnstr_config: dict, index_map: Any, fcns: dict | None = None, **kwargs: Any) -> None:
        """Nonconvex inequality constraint g(t, x, u) <= max_value or >= min_value."""
        self.name           = cnstr_config["name"]
        self.group          = cnstr_config.get("group", None)
        self.nodes          = cnstr_config.get("nodes", np.arange(0, index_map.N.time_grid))
        self.ct             = cnstr_config.get("ct", 0)
        self.hard           = cnstr_config.get("hard", 0)
        self.backend        = cnstr_config.get("backend", "jax")
        self.max_value      = cnstr_config.get("max_value", None)
        self.min_value      = cnstr_config.get("min_value", None)
        self.upper_and_lower = (self.max_value is not None) and (self.min_value is not None)

        self.fcn_string     = cnstr_config["fcn"]
        self.index_map      = index_map
        self.type           = 'nonconvex_inequality'

        self.fcn_dim = tools.resolve_function_from_string(self.fcn_string, fcns)

        if "dimension" in cnstr_config:
            self.dimension = cnstr_config["dimension"]
        elif self.backend == "jax":
            _params = kwargs.get("params", None)
            _out = self.fcn_dim(jnp.zeros(()), jnp.ones(index_map.n.state), jnp.ones(index_map.n.control), _params)
            self.dimension = jnp.atleast_1d(_out).shape[0]
        else:
            self.dimension = cnstr_config["dimension"]

        self.eps   = np.atleast_1d(cnstr_config.get("eps", np.full(self.dimension, 0.001)))
        self.scale = cnstr_config.get("scale", None)
        if self.scale is None and self.max_value is None and self.min_value is None:
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
        if self.max_value is not None:
            self.max_value = jnp.asarray(jnp.atleast_1d(self.max_value))

        if self.min_value is not None:
            self.min_value = jnp.asarray(jnp.atleast_1d(self.min_value))

        # get provided scales if provided
        if self.scale is not None:
            # TODO (Carlos): revisit, temporary scaling if max value is equal to zero
            self.M_out_d2nd = jnp.diag(1 / jnp.abs(jnp.asarray(self.scale)))
        elif self.max_value is not None:
            self.M_out_d2nd = jnp.diag(1 / jnp.abs(self.max_value))
        elif self.min_value is not None:
            self.M_out_d2nd = jnp.diag(1 / jnp.abs(self.min_value))
        else:
            raise ValueError("At least one of 'scales', 'max_value', or 'min_value' must be provided for nondimensionalization.")

        self.M_out_nd2d = jnp.diag(1 / jnp.diag(self.M_out_d2nd))

        if self.max_value is not None:
            self.max_value = self.M_out_d2nd @ self.max_value

        if self.min_value is not None:
            self.min_value = self.M_out_d2nd @ self.min_value

        M_state_nd2d_jax = jnp.asarray(nondim.M.state.nd2d)
        M_ctrl_nd2d_jax  = jnp.asarray(nondim.M.control.nd2d)

        self.fcn_nd = nondim.nondim_function(self.fcn_dim, M_state_nd2d_jax, M_ctrl_nd2d_jax, self.M_out_d2nd)

        self.eps = self.M_out_d2nd @ self.eps

        if self.upper_and_lower:
            self.eps = jnp.concatenate((self.eps, self.eps))

    def convexify_constraint(self):
        if self.upper_and_lower:
            fcn_lb_txu = lambda t, x, u, params: -self.fcn_nd(t, x, u, params) + self.min_value
            fcn_ub_txu = lambda t, x, u, params:  self.fcn_nd(t, x, u, params) - self.max_value
            fcn_txu = lambda t, x, u, params: jnp.concatenate([fcn_lb_txu(t, x, u, params), fcn_ub_txu(t, x, u, params)])
        elif (self.max_value is not None) and (self.min_value is None):
            fcn_txu = lambda t, x, u, params: self.fcn_nd(t, x, u, params) - self.max_value
        elif (self.min_value is not None) and (self.max_value is None):
            fcn_txu = lambda t, x, u, params: -self.fcn_nd(t, x, u, params) + self.min_value
        else:
            fcn_txu = self.fcn_nd

        self.fcn = self.index_map.problem.constraints.augment_txu_to_znu(fcn_txu)

        vals_function_txu = lambda t, z, nu, params: self.M_out_nd2d @ (self.fcn_nd(t, z, nu, params))
        vals_function = self.index_map.problem.constraints.augment_txu_to_znu(vals_function_txu)

        self.fcn_compiled, self.dfcn_dz_compiled, self.dfcn_du_compiled = convexify.linearize_jax(self.fcn)

        self.f_batched       = jax.jit(jax.vmap(vals_function,          in_axes=(0, 0, None)))
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
        self.fcn = self.index_map.problem.constraints.augment_dynamics_jax(self.fcn_base)

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
