from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from trajopt.methods.scp import convexify
from trajopt.utils import tools


class min_time:
    def __init__(self, cost_config: dict, index_map: Any, **kwargs: Any) -> None:
        """Minimum time cost: penalizes total flight time.

        Args:
            cost_config: Cost configuration dictionary.
            index_map: Index map object.
            **kwargs: Additional keyword arguments (unused).

        """
        self.type = "min_time"
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)

    def nondim_cost(self, nondim: Any) -> None:
        """Non-dimensionalize the cost (no-op for min_time).

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.

        """
        pass


class terminal_state:
    def __init__(self, cost_config: dict, index_map: Any, **kwargs: Any) -> None:
        """Terminal state cost: penalizes selected state components at the final node.

        Args:
            cost_config: Cost configuration dictionary.
            index_map: Index map object.
            **kwargs: Additional keyword arguments (unused).

        """
        self.type = "terminal_state"
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)
        self.idx = cost_config["idx"]
        self.sign = cost_config.get("sign", 1)

    def nondim_cost(self, nondim: Any) -> None:
        """Non-dimensionalize the cost (no-op for terminal_state).

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.

        """
        pass


class min_norm_terminal:
    def __init__(self, cost_config: dict, index_map: Any, **kwargs: Any) -> None:
        """Minimum norm terminal cost: penalizes the norm of selected terminal states.

        Args:
            cost_config: Cost configuration dictionary.
            index_map: Index map object.
            **kwargs: Additional keyword arguments (unused).

        """
        self.type = "min_norm_terminal"
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)
        self.idx = cost_config["idx"]
        self.value = np.array(cost_config["value"]) if "value" in cost_config else None

    def nondim_cost(self, nondim: Any) -> None:
        """Non-dimensionalize the cost (no-op for min_norm_terminal).

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.

        """
        pass


class regularization:
    def __init__(self, cost_config: dict, index_map: Any, **kwargs: Any) -> None:
        """Regularization cost: penalizes the norm of selected state or control components.

        Args:
            cost_config: Cost configuration dictionary.
            index_map: Index map object.
            **kwargs: Additional keyword arguments (unused).

        """
        self.type = "regularization"
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)
        self.set = cost_config["set"]
        self.norm_type = cost_config.get("norm_type", "l2")
        self.w = cost_config["w"]
        self.idx = cost_config.get("idx", np.arange(0, index_map.n.control))

    def nondim_cost(self, nondim: Any) -> None:
        """Non-dimensionalize the cost (no-op for regularization).

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.

        """
        pass


class rate_regularization:
    def __init__(self, cost_config: dict, index_map: Any, **kwargs: Any) -> None:
        """Rate regularization cost: penalizes the norm of finite differences of state or control.

        Args:
            cost_config: Cost configuration dictionary.
            index_map: Index map object.
            **kwargs: Additional keyword arguments (unused).

        """
        self.type = "rate_regularization"
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)
        self.set = cost_config["set"]
        self.norm_type = cost_config.get("norm_type", "l2")
        self.w = cost_config["w"]
        self.idx = cost_config.get("idx", np.arange(0, index_map.n.control))

    def nondim_cost(self, nondim: Any) -> None:
        """Non-dimensionalize the cost (no-op for rate_regularization).

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.

        """
        pass


class nonconvex:
    def __init__(self, cnstr_config: dict, index_map: Any, fcns: dict | None = None, **kwargs: Any) -> None:
        """Nonconvex cost evaluated via a user-supplied function.

        Args:
            cnstr_config: Cost configuration dictionary.
            index_map: Index map object.
            fcns: Resolved functions dictionary.
            **kwargs: Additional keyword arguments (unused).

        """
        # required config
        self.type = "nonconvex"
        self.name = cnstr_config["name"]
        self.group = cnstr_config.get("group", None)
        self.units = cnstr_config.get("units", None)
        self.scale = cnstr_config.get("scale", None)

        self.index_map = index_map

        self.fcn_string = cnstr_config["fcn"]
        self.minimax = cnstr_config.get("minimax", 0)

        # optional configs
        self.ct = cnstr_config.get("ct", 0)
        self.backend = cnstr_config.get("backend", "jax")

        # symbolic function in dimensional units provided by user
        # (jax or sympy)
        self.fcn_dim = tools.resolve_function_from_string(self.fcn_string, fcns)
        self.fcn_nd = None

        # this is the symbolic nondimmed version of fcn_fim, it will
        # be provided once the nondim_constraint() function is called
        self.fcn = None

        # the compiled version (jitted for jax / numpy for sympy)
        # will be provided by problem.constraints.convexify_constraints()
        self.fcn_compiled = None
        self.dfcn_dz_compiled = None
        self.dfcn_du_compiled = None

    def nondim_cost(self, nondim: Any) -> None:
        """Non-dimensionalize the cost function using the provided scaling matrices.

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.

        """
        if self.backend == "jax":

            if self.scale is not None:
                M_out_d2nd = jnp.atleast_1d(1 / self.scale)

            M_state_nd2d = nondim.M.state["nd2d"]
            M_ctrl_nd2d = nondim.M.control["nd2d"]

            self.fcn_txu = nondim.nondim_function(self.fcn_dim, M_state_nd2d, M_ctrl_nd2d, M_out_d2nd)

    def convexify_cost(self) -> None:
        """Compile and JIT-compile the cost and its Jacobians for use in SCP.

        Args:
            None.

        Returns:
            None.

        """
        if self.backend == "jax":
            self.fcn = self.index_map.problem.constraints.augment_txu_to_znu(self.fcn_txu)
            self.fcn_compiled, self.dfcn_dz_compiled, self.dfcn_du_compiled = convexify.linearize_jax(self.fcn)

            self.fcn_batched = jax.jit(jax.vmap(self.fcn_compiled, in_axes=(0, 0, None)))
            self.dfcn_dz_batched = jax.jit(jax.vmap(self.dfcn_dz_compiled, in_axes=(0, 0, None)))
            self.dfcn_du_batched = jax.jit(jax.vmap(self.dfcn_du_compiled, in_axes=(0, 0, None)))

        elif self.backend == "sympy":
            pass

    def g_aff(self, z: Any, nu: Any, params: Any) -> tuple:
        """Evaluate linearized (affine) cost at a single point.

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
        """Evaluate linearized (affine) cost over a trajectory batch.

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
