from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from jax import Array

from trajopt.utils.tools import AttrDict, resolve_function_from_string


class IndexMap:
    """Generate and store structured index maps for states, controls, dynamics, and constraints."""

    def __init__(self, config: AttrDict) -> None:
        """Initialize index map from problem and method configs."""
        # IndexMap should be fully defined from the configs
        self.problem_config = config.problem
        self.method_config  = config.method

        self.n = AttrDict({
            # core model dims
            "state":        max(max(group["idx"]) for group in self.problem_config["state"].values()) + 1,
            "control":      max(max(group["idx"]) for group in self.problem_config["control"].values()) + 1,
        })

        self.N = AttrDict({
            # core method dims
            "time_grid":            self.method_config["time_grid"],
        })

        self.update_indices()

############################################################
# INDEX PACKING / UNPACKING UTILITIES
############################################################

    def unpack_z(self, z: np.ndarray | Array) -> tuple[np.ndarray | Array, np.ndarray | Array, np.ndarray | Array]:
        """Unpack augmented state z = [x, t, beta].

        Supports both z shaped (n_z,) and z shaped (N, n_z).
        """
        z_ndim = getattr(z, "ndim", np.ndim(z))
        if z_ndim == 2:
            x = z[:, self.indices.z.state]
            t = z[:, self.indices.z.time]
            beta = z[:, self.indices.z.ctcs]
        else:
            x = z[self.indices.z.state]
            t = z[self.indices.z.time]
            beta = z[self.indices.z.ctcs]
        return x, t, beta


    def unpack_nu(self, nu: np.ndarray | Array) -> tuple[np.ndarray | Array, np.ndarray | Array]:
        """Unpack augmented control nu = [u, s].

        Supports both nu shaped (n_nu,) and nu shaped (N, n_nu).
        """
        nu_ndim = getattr(nu, "ndim", np.ndim(nu))
        if nu_ndim == 2:
            u = nu[:, self.indices.nu.control]
            s = nu[:, self.indices.nu.dilation_factor]
        else:
            u = nu[self.indices.nu.control]
            s = nu[self.indices.nu.dilation_factor]
        return u, s


    def unpack_znu(self, z: np.ndarray | Array, nu: np.ndarray | Array) -> tuple[np.ndarray | Array, ...]:
        """Unpack augmented state and control into physically meaningful variables."""
        x, t, beta  = self.unpack_z(z)
        u, s        = self.unpack_nu(nu)
        return x, t, beta, u, s


    def pack_z(self, x: np.ndarray | Array, t: np.ndarray | Array, beta: np.ndarray | Array) -> np.ndarray:
        """Pack physical variables into augmented state z = [x, t, beta].

        Supports both per-time-step vectors and time-stacked matrices.
        """
        x_ndim = getattr(x, "ndim", np.ndim(x))
        t_ndim = getattr(t, "ndim", np.ndim(t))
        beta_ndim = getattr(beta, "ndim", np.ndim(beta))

        if x_ndim == 2 or t_ndim == 2 or beta_ndim == 2:
            if x_ndim == 2:
                n_time = x.shape[0]
            elif t_ndim == 2:
                n_time = t.shape[0]
            else:
                n_time = beta.shape[0]

            z = np.zeros((n_time, len(self.indices.z.all)))
            z[:, self.indices.z.state] = x
            z[:, self.indices.z.time] = t
            z[:, self.indices.z.ctcs] = beta
        else:
            z = np.zeros(len(self.indices.z.all))
            z[self.indices.z.state] = x
            z[self.indices.z.time] = t
            z[self.indices.z.ctcs] = beta
        return z


    def pack_nu(self, u: np.ndarray | Array, s: float | np.ndarray | Array) -> np.ndarray:
        """Pack physical variables into augmented control nu = [u, s].

        Supports both per-time-step vectors and time-stacked matrices.
        """
        u_ndim = getattr(u, "ndim", np.ndim(u))
        s_ndim = getattr(s, "ndim", np.ndim(s))

        if u_ndim == 2 or s_ndim == 2:
            n_time = u.shape[0] if u_ndim == 2 else s.shape[0]
            nu = np.zeros((n_time, len(self.indices.nu.all)))
            nu[:, self.indices.nu.control] = u
            nu[:, self.indices.nu.dilation_factor] = s
        else:
            nu = np.zeros(len(self.indices.nu.all))
            nu[self.indices.nu.control] = u
            nu[self.indices.nu.dilation_factor] = s
        return nu

    def pack_znu(
        self,
        x: np.ndarray | Array,
        t: np.ndarray | Array,
        beta: np.ndarray | Array,
        u: np.ndarray | Array,
        s: float | np.ndarray | Array,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pack physical variables into augmented state and control."""
        z   = self.pack_z(x, t, beta)
        nu  = self.pack_nu(u,s)
        return z, nu


    def wrap_txu_fcn(self, f_phys: Callable) -> Callable:
        """Wrap f_phys(t, x, u, params) to be callable as f_alg(z, nu, params)."""
        def f_alg(z, nu, params):
            x, t, beta, u, s = self.unpack_znu(z, nu)
            return f_phys(t, x, u, params)

        return f_alg

    def wrap_znu_fcn(self, f_alg: Callable, beta: np.ndarray | None = None, s: float = 1.0) -> Callable:
        """Wrap f_alg(z, nu, params) to be callable as f_phys(t, x, u, params)."""
        def f_phys(t, x, u, params):
            beta_vec = np.zeros(len(self.indices.z.ctcs)) if beta is None else beta

            z = self.pack_z(x, t, beta_vec)
            nu = self.pack_nu(u, s)
            return f_alg(z, nu, params)

        return f_phys


    def evaluate_f_phys(self, f_phys: Callable, z: np.ndarray | Array, nu: np.ndarray | Array, params: AttrDict) -> Array:
        """Evaluate f_phys(t, x, u, params) from augmented variables (z, nu)."""
        x, t, beta, u, s = self.unpack_znu(z, nu)
        return f_phys(t, x, u, params)


    def evaluate_f_alg(
        self,
        f_alg: Callable,
        t: float,
        x: np.ndarray | Array,
        u: np.ndarray | Array,
        params: AttrDict,
        beta: np.ndarray | None = None,
        s: float = 1.0,
    ) -> Array:
        """Evaluate f_alg(z, nu, params) from physical variables (t, x, u)."""
        if beta is None:
            beta = np.zeros(len(self.indices.z.ctcs))

        z = self.pack_z(x, t, beta)
        nu = self.pack_nu(u, s)
        return f_alg(z, nu, params)

############################################################
# BUILD INDICES OBJECTS FROM CONFIGS
############################################################

    def _infer_constraint_dimensions(self) -> None:
        """Infer and inject 'dimension' into each constraint config from its fields.

        Handles all constraint types:
        - initial_state / final_state: count non-None entries in value, or len(idx)
        - state_limits / control_limits: count non-None in lower + upper
        - control_rate_limit: len(idx)
        - convex_inequality / nonconvex_inequality: resolve the function, call it
          with dummy inputs, and get output size. Doubles if both bounds present.
        - dynamics: set to 0 here (will be set to nz after ctcs count is known)
        """
        nx = self.n.state
        nu = self.n.control

        constraint_name_list = list(self.problem_config.constraints.keys())
        constraint_configs   = self.problem_config.constraints

        # resolve the raw fcns config so we can probe function-based constraints
        fcns_config = self.problem_config.get("fcns", {})
        resolved_fcns = AttrDict()
        for fname, fpath in fcns_config.items():
            if isinstance(fpath, str):
                resolved_fcns[fname] = resolve_function_from_string(fpath)
            else:
                resolved_fcns[fname] = fpath

        params = self.problem_config.get("params", None)

        for name in constraint_name_list:
            cfg   = constraint_configs[name]
            ctype = cfg.type

            if "dimension" in cfg:
                base_dim = int(cfg["dimension"])
            elif ctype in ("initial_state", "final_state"):
                base_dim = len(cfg["idx"]) if "idx" in cfg else sum(1 for v in cfg["value"] if v is not None)
            elif ctype in ("state_limits", "control_limits"):
                raw_lower = cfg.get("lower", [])
                raw_upper = cfg.get("upper", [])
                base_dim = (sum(1 for v in raw_lower if v is not None)
                          + sum(1 for v in raw_upper if v is not None))
            elif ctype == "control_rate_limit":
                base_dim = len(cfg["idx"])
            elif ctype == "convex_inequality":
                fcn = resolve_function_from_string(cfg["fcn"], resolved_fcns)
                _out = fcn(np.ones((1, nx)), np.ones((1, nu)), params)
                try:
                    base_dim = 1 if _out.ndim == 1 else _out.shape[1]
                except (TypeError, AttributeError):
                    base_dim = 1
            elif ctype == "nonconvex_inequality":
                fcn = resolve_function_from_string(cfg["fcn"], resolved_fcns)
                _out = fcn(0, np.ones(nx), np.ones(nu), params)
                try:
                    base_dim = jnp.atleast_1d(_out).shape[0]
                except TypeError:
                    base_dim = int(np.prod(_out.shape)) if _out.shape else 1
            elif ctype == "dynamics":
                cfg["dimension"] = 0
                continue
            elif ctype == "final_time":
                cfg["dimension"] = 1
                continue

            cfg["dimension"] = base_dim

    def update_indices(self) -> None:
        """Build and store all index arrays and dimension counts from the problem config."""
        nx      = self.n.state
        n_u     = self.n.control

        constraint_name_list = list(self.problem_config.constraints.keys())
        constraint_configs   = self.problem_config.constraints

        # infer dimensions for all constraints from config fields
        self._infer_constraint_dimensions()

        # if the method-level ctcs flag is set, enable ct for all nonconvex inequality constraints
        if self.method_config.get("flags", {}).get("ctcs", 0):
            for name in constraint_name_list:
                if constraint_configs[name].get("type") == "nonconvex_inequality":
                    constraint_configs[name]["ct"] = 1

        # count ctcs dimensions (constraints with ct=1)
        n_ctcs = 0
        for name in constraint_name_list:
            cfg = constraint_configs[name]
            if cfg.get("ct", 0) == 1:
                n_ctcs += cfg.dimension

        # now set dynamics dimension to nz
        n_t  = 1
        nz   = nx + n_t + n_ctcs
        for name in constraint_name_list:
            if constraint_configs[name].type == "dynamics":
                constraint_configs[name]["dimension"] = nz

        # accumulate total dimension per constraint type
        n_by_type = {}
        for name in constraint_name_list:
            cfg = constraint_configs[name]
            dim = cfg.dimension
            if cfg.type in ("convex_inequality", "nonconvex_inequality"):
                if cfg.get("upper") is not None and cfg.get("lower") is not None:
                    dim = 2 * dim
            n_by_type[cfg.type] = n_by_type.get(cfg.type, 0) + dim

        n_nu        = n_u + 1

        z_idx_all       = np.arange(0, nz)
        z_idx_state     = np.arange(0, nx)
        z_idx_time      = np.arange(nx, nx + n_t)
        z_idx_real      = np.concatenate((z_idx_state, z_idx_time))
        z_idx_ctcs      = np.arange(nx + n_t, nx + n_t + n_ctcs) if n_ctcs > 0 else np.array([], dtype=int)
        nu_idx_all      = np.arange(0, n_nu)
        u_idx_ctrl      = np.arange(0, n_u)
        u_idx_s         = np.arange(n_u, n_u + 1)

        Ak_ind          = np.arange(0, nz * nz)
        Bk_ind          = np.arange(Ak_ind[-1] + 1, Ak_ind[-1] + 1 + nz * n_nu)
        Bkp_ind         = np.arange(Bk_ind[-1] + 1, Bk_ind[-1] + 1 + nz * n_nu)

        time_grid       = self.N.time_grid

        n_real = nx + n_t

        # phase_idx_all = np.arange(
        #                         self.mission_config.phases.entry.start,
        #                         self.mission_config.phases.descent.end+1
        #                             )

        # entry_idx_all = np.arange(
        #                     self.mission_config.phases.entry.start,
        #                     self.mission_config.phases.entry.end+1
        #                             )
        # descent_idx_all = np.arange(
        #                     self.mission_config.phases.descent.start,
        #                     self.mission_config.phases.descent.end+1
        #                             )
        # phase_indices = AttrDict({
        #     "all":          phase_idx_all,
        #     "entry":        entry_idx_all,
        #     "descent":      descent_idx_all,
        # })

        z_indices = AttrDict({
            "all":          z_idx_all,
            "state":        z_idx_state,
            "time":         z_idx_time,
            "real":         z_idx_real,
            "ctcs":         z_idx_ctcs,
        })

        nu_indices = AttrDict({
            "all":              nu_idx_all,
            "control":          u_idx_ctrl,
            "dilation_factor":  u_idx_s,
        })

        # terminal = AttrDict({
        #     "eq":           term_eq_idx,
        #     "ineq":         term_ineq_idx,
        #     "ctcs":         term_ctcs_idx,
        #     "all":          np.arange(0, n_term_eq + n_term_ineq + n_term_ctcs),
        # })

        dynamics_indices = AttrDict({
            "Ak":           Ak_ind,
            "Bk":           Bk_ind,
            "Bkp":          Bkp_ind,
        })

        ctcs_indices = AttrDict({
            "state":        z_idx_ctcs,
            # "term":         term_ctcs_idx,
            # "ineq":         np.array([], dtype=int),
        })

        constraint_indices = AttrDict({"dynamics": dynamics_indices})
        for t, n in n_by_type.items():
            if t != "dynamics":
                constraint_indices[t] = AttrDict({"all": np.arange(0, n)})

        self.indices = AttrDict({
            "z":            z_indices,
            "nu":           nu_indices,
            "constraints":  constraint_indices,
            "ctcs":         ctcs_indices,
            # "phases":   phase_indices
        })

        self.n = AttrDict({
            "state":                int(nx),
            "time":                 int(n_t),
            "real":                 int(n_real),
            "control":              int(n_u),
            "nu":                   int(n_nu),
            "z":                    int(nz),
            "ctcs":                 int(n_ctcs),
            **{t: int(n) for t, n in n_by_type.items()},
            "dynamics":             int(nz),
        })

        # phase_dim = AttrDict({
        #     "all":          len(phase_idx_all),
        #     "entry":        len(entry_idx_all),
        #     "descent":      len(descent_idx_all),
        # })

        self.N = AttrDict({
            "nonconvex_inequality": time_grid,
            "dynamics":             time_grid - 1,
            "initial_state":        1,
            "final_state":          1,
            "box":                  time_grid,
            "control_rate_limit":   time_grid - 1,
            "axis_angle_cone":      time_grid,
            "max_norm_cone":        time_grid,
            "quaternion_cone":      time_grid,
            "AFFINE":               time_grid,
            "POLYTOPE":             time_grid,
            "SOC":                  time_grid,
            "final_time":           1,
            "equality_bc":          1,
            "inequality_bc":        1,
            "time_grid":            time_grid,
            "real":                 time_grid,
        })

    # TODO(Skye): Update this summary
    def summary(self) -> None:
        """Print a human-readable summary of all index arrays and dimension counts."""
        print("==== Problem Indices Summary ====")
        print("\nN (TIME-RELATED INDICES):")
        for k, v in self.N.items():
            print(f"  {k:25s}: {v}")

        print("\nINDEX ARRAYS:")
        for obj_name in ["z", "nu", "ctcs"]:
            obj = self.indices[obj_name]
            print(f"  {obj_name.upper()}:")
            for k, v in obj.items():
                print(f"    {k:25s}: {v.shape if hasattr(v,'shape') else len(v)}")

        print("\nCONSTRAINTS:")
        for constraint_type in ["nonlinear_inequality", "final_state", "dynamics"]:
            constraint_obj = getattr(self.indices.constraints, constraint_type)
            print(f"    {constraint_type.upper()}:")
            for k, v in constraint_obj.items():
                print(f"      {k:25s}: {v.shape if hasattr(v,'shape') else len(v)}")

        print("\nSCALAR COUNTS (self.n):")
        for k, v in self.n.items():
            print(f"  {k:25s}: {v}")
