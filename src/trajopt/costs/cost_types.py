from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from trajopt.utils import tools


class min_time:
    def __init__(self, cost_config: dict, segment) -> None:
        self.type = "min_time"
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)
        self.nodes = cost_config.get("nodes", np.arange(0, segment.index_map.N.all))


class terminal_state:
    def __init__(self, cost_config: dict, segment) -> None:
        self.type = "terminal_state"
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)
        self.nodes = cost_config.get("nodes", np.array([segment.index_map.N.all - 1]))
        self.idx = cost_config["idx"]
        self.sign = cost_config.get("sign", 1)


class min_norm_terminal:
    def __init__(self, cost_config: dict, segment) -> None:
        self.type = "min_norm_terminal"
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)
        self.nodes = cost_config.get("nodes", np.array([segment.index_map.N.all - 1]))
        self.idx = cost_config["idx"]
        self.value = np.array(cost_config["value"]) if "value" in cost_config else None


class regularization:
    def __init__(self, cost_config: dict, segment) -> None:
        self.type = "regularization"
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)
        self.nodes = cost_config.get("nodes", np.arange(0, segment.index_map.N.all))
        self.set = cost_config["set"]
        self.norm_type = cost_config.get("norm_type", "l2")
        self.w = cost_config["w"]
        self.idx = cost_config.get("idx", np.arange(0, segment.index_map.n.control))


class rate_regularization:
    def __init__(self, cost_config: dict, segment) -> None:
        self.type = "rate_regularization"
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)
        self.nodes = cost_config.get("nodes", np.arange(0, segment.index_map.N.all))
        self.set = cost_config["set"]
        self.norm_type = cost_config.get("norm_type", "l2")
        self.w = cost_config["w"]
        self.idx = cost_config.get("idx", np.arange(0, segment.index_map.n.control))


class nonconvex_terminal:
    def __init__(self, cost_config: dict, segment) -> None:
        index_map = segment.index_map
        nondim = segment.nondim
        fcns = segment.fcns

        self.type = "nonconvex_terminal"
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)
        self.nodes = cost_config.get("nodes", np.array([index_map.N.all - 1]))
        self.scale = cost_config.get("scale", None)
        self.backend = cost_config.get("backend", "jax")
        self.ct = cost_config.get("ct", 0)
        self.minimax = cost_config.get("minimax", 0)

        self.index_map = index_map
        self.fcn_string = cost_config["fcn"]
        self.fcn_txu_dim = tools.resolve_function_from_string(self.fcn_string, fcns)

        self.M_out_d2nd   = jnp.atleast_1d(1 / self.scale) if self.scale is not None else jnp.atleast_1d(1.0)
        self.M_state_nd2d = jnp.asarray(nondim.M.state.nd2d)
        self.M_ctrl_nd2d  = jnp.asarray(nondim.M.control.nd2d)

        if self.backend == "jax":
            self.fcn_compiled = jax.jit(self.fcn_znu)
            self.dfcn_dz_compiled = jax.jit(jax.jacfwd(self.fcn_znu, argnums=0))
            self.dfcn_du_compiled = jax.jit(jax.jacfwd(self.fcn_znu, argnums=1))
            self.d2fcn_dz2_compiled = jax.jit(jax.hessian(self.fcn_znu, argnums=0))
            self.d2fcn_dnu2_compiled = jax.jit(jax.hessian(self.fcn_znu, argnums=1))
            self.d2fcn_dzdnu_compiled = jax.jit(jax.jacfwd(jax.jacrev(self.fcn_znu, argnums=1), argnums=0))
            self.fcn_batched = jax.jit(jax.vmap(self.fcn_compiled, in_axes=(0, 0, None)))
            self.dfcn_dz_batched = jax.jit(jax.vmap(self.dfcn_dz_compiled, in_axes=(0, 0, None)))
            self.dfcn_du_batched = jax.jit(jax.vmap(self.dfcn_du_compiled, in_axes=(0, 0, None)))
            self.d2fcn_dz2_batched = jax.jit(jax.vmap(self.d2fcn_dz2_compiled, in_axes=(0, 0, None)))
            self.d2fcn_dnu2_batched = jax.jit(jax.vmap(self.d2fcn_dnu2_compiled, in_axes=(0, 0, None)))
            self.d2fcn_dzdnu_batched = jax.jit(jax.vmap(self.d2fcn_dzdnu_compiled, in_axes=(0, 0, None)))

    def fcn_txu_nd(self, x, u, t, params):
        return self.M_out_d2nd @ jnp.atleast_1d(
            self.fcn_txu_dim(self.M_state_nd2d @ x, self.M_ctrl_nd2d @ u, t, params)
        )

    def fcn_znu(self, z, nu, params):
        x, t, _, u, _ = self.index_map.unpack_znu(z, nu)
        return self.fcn_txu_nd(x, u, t, params)

    def g_aff(self, z: Any, nu: Any, params: Any) -> tuple:
        return (
            self.fcn_compiled(z, nu, params),
            self.dfcn_dz_compiled(z, nu, params),
            self.dfcn_du_compiled(z, nu, params),
        )

    def g_aff_batched(self, z: Any, nu: Any, params: Any) -> tuple:
        return (
            self.fcn_batched(z, nu, params),
            self.dfcn_dz_batched(z, nu, params),
            self.dfcn_du_batched(z, nu, params),
        )


class convex_terminal:
    def __init__(self, cost_config: dict, segment) -> None:
        fcns = segment.fcns

        self.type = "convex_terminal"
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)
        self.nodes = cost_config.get("nodes", np.array([segment.index_map.N.all - 1]))
        self.w = cost_config.get("w", 1.0)

        self.fcn_string = cost_config["fcn"]
        self.fcn_dim = tools.resolve_function_from_string(self.fcn_string, fcns)


class convex_running:
    def __init__(self, cost_config: dict, segment) -> None:
        fcns = segment.fcns

        self.type = "convex_running"
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)
        self.nodes = cost_config.get("nodes", np.arange(0, segment.index_map.N.all))
        self.w = cost_config.get("w", 1.0)

        self.fcn_string = cost_config["fcn"]
        self.fcn_dim = tools.resolve_function_from_string(self.fcn_string, fcns)


class nonconvex_running:
    def __init__(self, cost_config: dict, segment) -> None:
        index_map = segment.index_map
        nondim = segment.nondim
        fcns = segment.fcns

        self.type = "nonconvex_running"
        self.name = cost_config["name"]
        self.group = cost_config.get("group", None)
        self.nodes = cost_config.get("nodes", np.arange(0, index_map.N.all))
        self.scale = cost_config.get("scale", None)
        self.backend = cost_config.get("backend", "jax")
        self.ct = cost_config.get("ct", 0)
        self.minimax = cost_config.get("minimax", 0)
        self.w = cost_config.get("w", 1.0)

        self.index_map = index_map
        self.fcn_string = cost_config["fcn"]
        self.fcn_txu_dim = tools.resolve_function_from_string(self.fcn_string, fcns)

        self.M_out_d2nd   = jnp.atleast_1d(1 / self.scale) if self.scale is not None else jnp.atleast_1d(1.0)
        self.M_state_nd2d = jnp.asarray(nondim.M.state.nd2d)
        self.M_ctrl_nd2d  = jnp.asarray(nondim.M.control.nd2d)

        if self.backend == "jax":
            self.fcn_compiled = jax.jit(self.fcn_znu)
            self.dfcn_dz_compiled = jax.jit(jax.jacfwd(self.fcn_znu, argnums=0))
            self.dfcn_du_compiled = jax.jit(jax.jacfwd(self.fcn_znu, argnums=1))
            self.d2fcn_dz2_compiled = jax.jit(jax.hessian(self.fcn_znu, argnums=0))
            self.d2fcn_dnu2_compiled = jax.jit(jax.hessian(self.fcn_znu, argnums=1))
            self.d2fcn_dzdnu_compiled = jax.jit(jax.jacfwd(jax.jacrev(self.fcn_znu, argnums=1), argnums=0))
            self.fcn_batched = jax.jit(jax.vmap(self.fcn_compiled, in_axes=(0, 0, None)))
            self.dfcn_dz_batched = jax.jit(jax.vmap(self.dfcn_dz_compiled, in_axes=(0, 0, None)))
            self.dfcn_du_batched = jax.jit(jax.vmap(self.dfcn_du_compiled, in_axes=(0, 0, None)))
            self.d2fcn_dz2_batched = jax.jit(jax.vmap(self.d2fcn_dz2_compiled, in_axes=(0, 0, None)))
            self.d2fcn_dnu2_batched = jax.jit(jax.vmap(self.d2fcn_dnu2_compiled, in_axes=(0, 0, None)))
            self.d2fcn_dzdnu_batched = jax.jit(jax.vmap(self.d2fcn_dzdnu_compiled, in_axes=(0, 0, None)))

    def fcn_txu_nd(self, x, u, t, params):
        return self.M_out_d2nd @ jnp.atleast_1d(
            self.fcn_txu_dim(self.M_state_nd2d @ x, self.M_ctrl_nd2d @ u, t, params)
        )

    def fcn_znu(self, z, nu, params):
        x, t, _, u, _ = self.index_map.unpack_znu(z, nu)
        return self.fcn_txu_nd(x, u, t, params)

    def g_aff(self, z: Any, nu: Any, params: Any) -> tuple:
        return (
            self.fcn_compiled(z, nu, params),
            self.dfcn_dz_compiled(z, nu, params),
            self.dfcn_du_compiled(z, nu, params),
        )

    def g_aff_batched(self, z: Any, nu: Any, params: Any) -> tuple:
        return (
            self.fcn_batched(z, nu, params),
            self.dfcn_dz_batched(z, nu, params),
            self.dfcn_du_batched(z, nu, params),
        )
