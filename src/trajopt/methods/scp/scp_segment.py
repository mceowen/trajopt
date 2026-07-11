import copy
import time

import numpy as np
import cvxpy as cp
import jax.numpy as jnp

from trajopt.segment import Segment
from trajopt.methods.common import initial_guess
from trajopt.methods.common import convergence
import trajopt.methods.scp.scp_constraints.scp_constraint_types as scp_constraint_type_module
import trajopt.methods.scp.scp_costs.scp_cost_types as scp_cost_type_module
from trajopt.utils.tools import AttrDict, recursive_attrdict

class SCPSegment():

    def __init__(self, segment: Segment, method_config: AttrDict) -> None:
        self.name          = segment.name
        self.segment       = segment
        self.method_config = method_config

        self.index_map      = segment.index_map
        self.nondim         = segment.nondim
        self.params         = segment.params
        self.flags          = method_config.flags
        self.penalty_config = method_config.penalty

        # dictionary of scp-constraints for this scp-segment
        self.constraints = AttrDict()
        for cnstr_name, constraint in segment.constraints.items():
            scp_class_name = f"scp_{constraint.type}"
            constraintClass = getattr(scp_constraint_type_module, scp_class_name)
            self.constraints[cnstr_name] = constraintClass(constraint, self)

        # dictionary of scp costs types for this scp-segment
        self.costs = AttrDict()
        for cost_name, cost in segment.costs.items():
            scp_class_name = f"scp_{cost.type}"
            costClass = getattr(scp_cost_type_module, scp_class_name)
            self.costs[cost_name] = costClass(cost, self)

        self.initialize()

        self.cp_params            = AttrDict()
        self.cp_vars              = AttrDict()
        self.cp_constraints       = []
        self.cp_cost              = 0
        self.cp_subproblem_status = None
        self._active_set_layout_cache = None
        self._active_set_box_layout_cache = None

        self.create_cvxpy_parameters()
        self.create_cvxpy_variables()
        self.create_cvxpy_constraints()
        self.create_cvxpy_cost()

    def initialize(self) -> None:
        segment = self.segment

        self.initial_guess = AttrDict()

        cfg_guess             = segment.guess
        t_start               = getattr(cfg_guess, 't_start', 0.0)
        t_stop                = cfg_guess.t_stop
        t_start_nd            = t_start / self.nondim.time_scale
        t_stop_nd             = t_stop / self.nondim.time_scale
        self.Ts_init          = t_stop_nd - t_start_nd
        t_init                = np.linspace(t_start_nd, t_stop_nd, self.index_map.N.all)
        dt_init               = np.diff(t_init)
        self.initial_guess.t  = t_init
        self.initial_guess.dt = dt_init

        for constraint in self.constraints.values():
            constraint.compile(self)
            constraint.init_penalty(self)

        dyn = next((c for c in self.constraints.values() if c.type == "dynamics"), None)
        self.eps_dyn   = dyn.eps.copy() if dyn is not None else np.full(self.index_map.n.z, 1e-4)
        self.eps_dyn[self.index_map.indices.z.running_cost] = np.inf
        self.eps_state = self.eps_dyn[self.index_map.indices.z.state]
        self.eps_cost  = np.atleast_1d(1e-4)

        initial_guess.set_initial_guess(segment, self)

        self.iter_data_list = []
        self.lm_mu = 1e-8
        self._auto_kappa_state = 0.0

        self.current_iter_data = recursive_attrdict({
            "iter_num": 0,
            "z_opt": self.initial_guess.z,
            "nu_opt": self.initial_guess.nu,
            "cost": 0.0,
            "penalty_cost": 0.0,
            "vb":     AttrDict({c.name: c.vb     for c in self.constraints.values() if c.shape is not None}),
            "W":      AttrDict({c.name: c.W      for c in self.constraints.values() if c.shape is not None}),
            "dual":   AttrDict({c.name: c.dual   for c in self.constraints.values() if c.shape is not None}),
            "W_p":    AttrDict({c.name: c.W_p    for c in self.constraints.values() if c.vb_type == "split"}),
            "W_m":    AttrDict({c.name: c.W_m    for c in self.constraints.values() if c.vb_type == "split"}),
            "dual_p": AttrDict({c.name: c.dual_p for c in self.constraints.values() if c.vb_type == "split"}),
            "dual_m": AttrDict({c.name: c.dual_m for c in self.constraints.values() if c.vb_type == "split"}),
        })

        self.iter_data_list.append(copy.deepcopy(self.current_iter_data))

        self.compile_merit()

    def create_cvxpy_parameters(self) -> None:
        N = self.index_map.N.all
        n_z = self.index_map.n.z
        n_nu = self.index_map.n.nu

        self.cp_params.z_ref  = cp.Parameter((N, n_z),  name="z_ref")
        self.cp_params.nu_ref = cp.Parameter((N, n_nu), name="nu_ref")

        self.x_ref, self.t_ref, self.beta_ref, self.u_ref, self.s_ref = self.index_map.unpack_znu(self.cp_params.z_ref, self.cp_params.nu_ref)

        self.cp_params.tr_z   = cp.Parameter(nonneg=True, name="tr_z")
        self.cp_params.tr_nu  = cp.Parameter(nonneg=True, name="tr_u")

        self.cp_params.dcostdx = cp.Parameter((N, n_z),  name="dcostdx")
        self.cp_params.dcostdu = cp.Parameter((N, n_nu), name="dcostdu")
        self.cp_params.cost0   = cp.Parameter((N,),      name="cost0")

        if self.flags.discretize == "ps":
            self.cp_params.tau = cp.Parameter((N,), name="tau")
            self.cp_params.tau.value = self.ps_tau_norm
            self.cp_params.ps_t_offset = cp.Parameter((N,), name="ps_t_offset", value=np.zeros(N))

        for constraint in self.constraints.values():
            constraint.create_penalty_parameters(self)
            constraint.create_cvxpy_parameters(self)

    def create_cvxpy_variables(self) -> None:
        N, n_x, n_t, n_u = self.index_map.N.all, self.index_map.n.state, self.index_map.n.time, self.index_map.n.control
        n_ctcs = self.index_map.n.ctcs
        n_rc   = self.index_map.n.running_cost

        self.cp_vars.dx     = cp.Variable((N, n_x),    name="dx")
        self.cp_vars.dbeta  = cp.Variable((N, n_ctcs), name="dbeta")  if n_ctcs > 0 else None
        self.cp_vars.dgamma = cp.Variable((N, n_rc),   name="dgamma") if n_rc   > 0 else None
        self.cp_vars.du     = cp.Variable((N, n_u),    name="du")

        if bool(self.flags.free_final_time):
            self.cp_vars.dt = cp.Variable((N, n_t), name="dt")
            self.cp_vars.ds = cp.Variable((N, 1),   name="ds")
        else:
            self.cp_vars.dt = cp.Constant(np.zeros((N, n_t)))
            self.cp_vars.ds = cp.Constant(np.zeros((N, 1)))

        dz_components = [self.cp_vars.dx, self.cp_vars.dt]

        if n_ctcs > 0:
            dz_components.append(self.cp_vars.dbeta)
        if n_rc > 0:
            dz_components.append(self.cp_vars.dgamma)

        self.dz  = cp.hstack(dz_components)
        self.dnu = cp.hstack([self.cp_vars.du, self.cp_vars.ds])
        self.dt  = self.cp_vars.dt
        self.ds  = self.cp_vars.ds

        for constraint in self.constraints.values():
            constraint.create_penalty_variables(self)
            constraint.create_cvxpy_variables(self)

    def create_cvxpy_constraints(self) -> None:
        for constraint in self.constraints.values():
            constraint.create_cvxpy_constraints(self)

        if bool(self.flags.free_final_time):
            self.create_free_final_time_constraints()

    def create_cvxpy_cost(self) -> None:
        for cost in self.costs.values():
            cost.create_cvxpy_cost(self)

        self.create_cost_trust_region()

        for constraint in self.constraints.values():
            constraint.add_penalty_cost(self)

    def create_free_final_time_constraints(self) -> None:
        N = self.index_map.N.all

        self.cp_constraints.append(self.dt[0, 0] == 0)

        if self.flags.discretize == "ps":
            tau = self.cp_params.tau
            for k in range(1, N - 1):
                self.cp_constraints.append(self.dt[k, 0] == self.cp_params.ps_t_offset[k] + tau[k] * self.dt[N - 1, 0])

            for k in range(N - 1):
                self.cp_constraints.append(0.0 <= self.s_ref[k, 0] + self.ds[k, 0])
                s_k  = self.s_ref[k, 0] + self.ds[k, 0]
                s_kp = self.s_ref[k + 1, 0] + self.ds[k + 1, 0]
                self.cp_constraints.append(s_k == s_kp)

            self.cp_constraints.append(self.t_ref[N - 1, 0] + self.dt[N - 1, 0] >= 0.0)
            self.cp_constraints.append(0.0 <= self.s_ref[N - 1, 0] + self.ds[N - 1, 0])
            return

        for k in range(N - 1):
            t_0 = self.t_ref[0, 0] + self.dt[0, 0]
            t_1 = self.t_ref[1, 0] + self.dt[1, 0]

            t_k = self.t_ref[k, 0] + self.dt[k, 0]
            t_kp = self.t_ref[k+1, 0] + self.dt[k+1, 0]
            
            s_k = self.s_ref[k, 0] + self.ds[k, 0]
            s_kp = self.s_ref[k+1, 0] + self.ds[k+1, 0]
            
            self.cp_constraints.append(t_k >= 0)
            # self.cp_constraints.append(0.1 <= s_k)
            # self.cp_constraints.append(cp.abs(s_kp - s_k) <= 0.5)

            if hasattr(self.flags, "equal_dt") and bool(self.flags.equal_dt):
                interval_k = t_kp - t_k
                interval_0 = t_1 - t_0
                self.cp_constraints.append(interval_k == interval_0)

            if hasattr(self.flags, "zoh_dilation") and bool(self.flags.zoh_dilation):
                self.cp_constraints.append(s_k == s_kp)

    def create_cost_trust_region(self) -> None:
        if self.flags.discretize not in ("ms", "ps"):
            return
        if getattr(self.flags, 'second_order', True):
            for k in range(self.index_map.N.all):
                w_k = cp.hstack([self.dz[k], self.dnu[k]])
                self.cp_cost += 0.5 * cp.sum_squares(self.cp_params.L[k] @ w_k)
        else:
            for k in range(self.index_map.N.all):
                self.cp_cost += 0.5 * self.cp_params.tr_z * cp.sum_squares(self.dz[k])
                self.cp_cost += 0.5 * self.cp_params.tr_nu * cp.sum_squares(self.dnu[k])

    def compile_merit(self) -> None:
        for cost in self.costs.values():
            cost.compile_merit_cost(self)
        for constraint in self.constraints.values():
            constraint.compile_merit_penalty(self)

    def evaluate_merit_at_alpha(self, alpha):
        z  = jnp.asarray(self.current_iter_data.z_opt)  + alpha * jnp.asarray(self._dz_new)
        nu = jnp.asarray(self.current_iter_data.nu_opt) + alpha * jnp.asarray(self._dnu_new)
        phi = sum(c.evaluate_merit_cost(z, nu, self.params) for c in self.costs.values())
        for c in self.constraints.values():
            phi += c.evaluate_merit(z, nu, self.params)
        return phi

    def merit_grad_at_zero(self):
        z_ref  = jnp.asarray(self.current_iter_data.z_opt)
        nu_ref = jnp.asarray(self.current_iter_data.nu_opt)
        dz     = jnp.asarray(self._dz_new)
        dnu    = jnp.asarray(self._dnu_new)
        phi, dphi = 0.0, 0.0
        for c in self.costs.values():
            v, g = c.merit_cost_value_and_grad_alpha(0.0, z_ref, dz, nu_ref, dnu, self.params)
            phi += v
            dphi += g
        for c in self.constraints.values():
            v, g = c.merit_value_and_grad_alpha(0.0, z_ref, dz, nu_ref, dnu, self.params)
            phi += v
            dphi += g
        return phi, dphi

    def update_cvxpy_parameters(self) -> None:
        z_opt  = self.current_iter_data.z_opt
        nu_opt = self.current_iter_data.nu_opt

        self.cp_params.z_ref.value  = z_opt
        self.cp_params.nu_ref.value = nu_opt

        self.x_ref, self.t_ref, self.beta_ref, self.u_ref, self.s_ref = self.index_map.unpack_znu(z_opt, nu_opt)

        if self.flags.discretize == "ps":
            t_ref_vals = z_opt[:, self.index_map.indices.z.time].flatten()
            t0, tf = t_ref_vals[0], t_ref_vals[-1]
            self.cp_params.ps_t_offset.value = t0 + self.ps_tau_norm * (tf - t0) - t_ref_vals

        disc_start_time = time.perf_counter()

        for constraint in self.constraints.values():
            constraint.update_cvxpy_parameters(self)

        for cost in self.costs.values():
            cost.update_cvxpy_parameters(self)

        self.update_lagrangian_hessian()
        disc_end_time = time.perf_counter()

        self.current_iter_data.discretization_time = (disc_end_time - disc_start_time) * 1000

        self.cp_params.tr_z.value  = (1 / self.method_config.weights.alpha_z)
        self.cp_params.tr_nu.value = (1 / self.method_config.weights.alpha_nu)

        for constraint in self.constraints.values():
            constraint.update_penalty_parameters(self)

    def update_lagrangian_hessian(self) -> None:
        if self.flags.discretize not in ("ms", "ps"):
            return
        if not getattr(self.flags, 'second_order', True):
            return

        N    = self.index_map.N.all
        n_z  = self.index_map.n.z
        n_nu = self.index_map.n.nu

        iteration = len(self.iter_data_list)

        if iteration == 0:
            H_base = np.tile(np.eye(n_z + n_nu), (N, 1, 1))
        else:
            H_base = np.zeros((N, n_z + n_nu, n_z + n_nu))
            self._adapt_levenberg(iteration)
            H_base += self.lm_mu * np.eye(n_z + n_nu)[np.newaxis, :, :]

        H = H_base.copy()

        for constraint in self.constraints.values():
            constraint.accumulate_hessian(self, H)

        for cost in self.costs.values():
            cost.accumulate_hessian(self, H)

        if getattr(self.flags, 'active_set_hessian', False):
            H = self._project_active_set_hessian(H, H_base)

        self.cp_params.L.value = _psd_sqrt(H)

    def _active_set_layout(self):
        """(constraint, is_eq, dim) for every nonconvex ineq/eq constraint
        eligible for the active-set projection. Cached: fixed for the life
        of a solve.
        """
        if self._active_set_layout_cache is None:
            layout = []
            for constraint in self.constraints.values():
                is_ineq = isinstance(constraint, scp_constraint_type_module.scp_nonconvex_inequality)
                is_eq   = isinstance(constraint, scp_constraint_type_module.scp_nonconvex_equality)
                if not (is_ineq or is_eq):
                    continue
                if getattr(constraint, 'g0_param', None) is None:
                    continue
                layout.append((constraint, is_eq, int(constraint.eps.shape[0])))
            self._active_set_layout_cache = layout
        return self._active_set_layout_cache

    def _active_set_box_layout(self):
        """(nodes, col, side, bound_value) for every state/control box-limit
        row eligible for the active-set projection. These are hard QP
        bounds, not linearized rows, so the Jacobian row is just a unit
        vector. Opt-in via `flags.active_set_box_tol` (0 = off). Cached.
        """
        if self._active_set_box_layout_cache is None:
            idx_state = self.index_map.indices.z.state
            idx_ctrl  = self.index_map.indices.nu.control
            n_z = self.index_map.n.z
            N = self.index_map.N.all
            # `col` below is already in the combined (n_z + n_nu) column
            # space that H/J use, i.e. control columns are offset by n_z.
            node_sets = [
                (scp_constraint_type_module.scp_state_limits,           np.arange(N),      idx_state,       0),
                (scp_constraint_type_module.scp_initial_state_limits,   np.array([0]),     idx_state,       0),
                (scp_constraint_type_module.scp_final_state_limits,     np.array([N - 1]), idx_state,       0),
                (scp_constraint_type_module.scp_control_limits,         np.arange(N),      idx_ctrl,       n_z),
                (scp_constraint_type_module.scp_initial_control_limits, np.array([0]),     idx_ctrl,       n_z),
                (scp_constraint_type_module.scp_final_control_limits,   np.array([N - 1]), idx_ctrl,       n_z),
            ]
            layout = []
            for constraint in self.constraints.values():
                for cls, nodes, idx_map, col_offset in node_sets:
                    if not isinstance(constraint, cls):
                        continue
                    c = constraint.constraint
                    for j, local_i in enumerate(c.lower_idx):
                        col = col_offset + int(idx_map[local_i])
                        layout.append((nodes, col, 'lower', float(np.atleast_1d(c.lower_value)[j])))
                    for j, local_i in enumerate(c.upper_idx):
                        col = col_offset + int(idx_map[local_i])
                        layout.append((nodes, col, 'upper', float(np.atleast_1d(c.upper_value)[j])))
                    break
            self._active_set_box_layout_cache = layout
        return self._active_set_box_layout_cache

    def _project_active_set_hessian(self, H: np.ndarray, H_base: np.ndarray) -> np.ndarray:
        """Project each node's H_k onto the null space of currently-active
        constraint rows (SQP-style reduced Hessian), batched over all N
        nodes:

            P_k = I - J_k^T (J_k J_k^T)^+ J_k
            H_floor_k  = (1 - kappa) * P_k H_base_k P_k^T + kappa * H_base_k
            H_proj_k   = H_floor_k + P_k (H_k - H_base_k) P_k^T

        Rows come from `_active_set_layout` (nonconvex ineq/eq) and
        `_active_set_box_layout` (box limits, opt-in). Row activeness is
        a continuous `[0, 1]` weight (`flags.active_set_tol_factor`
        hard/smoothed threshold, or `active_set_dual_weighted`), scaled
        into `J` before the Gram matrix is formed. `kappa` is
        `flags.active_set_lm_floor_fraction` or `_auto_kappa()`.
        """
        N = self.index_map.N.all
        d = H.shape[-1]
        tol_factor = float(getattr(self.flags, 'active_set_tol_factor', 1.0))
        smooth_width_factor = float(getattr(self.flags, 'active_set_smooth_width_factor', 0.0))
        smooth_lambda = float(getattr(self.flags, 'active_set_smooth_lambda', 0.0))

        box_tol     = float(getattr(self.flags, 'active_set_box_tol', 0.0))
        box_layout  = self._active_set_box_layout() if box_tol > 0 else []

        layout        = self._active_set_layout()
        dim_nonconvex = sum(dim for _, _, dim in layout)
        dim_box       = len(box_layout)
        dim_total     = dim_nonconvex + dim_box
        if dim_total == 0:
            return H

        dual_weighted = bool(getattr(self.flags, 'active_set_dual_weighted', False))
        dual_eps      = float(getattr(self.flags, 'active_set_dual_eps', 1e-6))

        J    = np.zeros((N, dim_total, d))
        row_weight = np.zeros((N, dim_total))  # continuous row weight in [0, 1]; scales J below
        lam        = np.zeros((N, dim_total))

        row = 0
        for constraint, is_eq, dim in layout:
            if constraint.g0_param.value is None:
                row += dim
                continue

            g0    = constraint.g0_param.value    # (nn, dim)
            dgdz  = constraint.dgdz_param.value  # (nn, dim, n_z)
            dgdnu = constraint.dgdnu_param.value # (nn, dim, n_nu)
            tol   = tol_factor * constraint.eps  # (dim,)
            nodes = constraint.nodes

            if is_eq:
                weight = np.ones_like(g0)
                lam_g  = np.zeros_like(g0)
            elif dual_weighted:
                # weight from the row's own dual, normalized by W (dual scales with W's range)
                dual = np.abs(np.asarray(constraint.lagrangian_dual))
                W    = np.asarray(constraint.W)
                if W.shape == dual.shape:
                    dual = dual / np.maximum(W, 1e-12)  # guard only; W is already >= 1e-5
                weight = dual / (dual + dual_eps)
                lam_g  = np.zeros_like(g0)
            elif smooth_width_factor > 0:
                width   = smooth_width_factor * constraint.eps  # (dim,)
                d0      = g0 - (-tol)                            # >= 0 means hard-active
                weight  = (d0 >= -width).astype(float)
                lam_g   = smooth_lambda * np.clip(-d0 / width, 0.0, 1.0)
            else:
                weight = (g0 >= -tol).astype(float)
                lam_g  = np.zeros_like(g0)

            J[nodes, row:row + dim, :]      = np.concatenate([dgdz, dgdnu], axis=-1)
            row_weight[nodes, row:row + dim] = weight
            lam[nodes, row:row + dim]        = lam_g
            row += dim

        assert row == dim_nonconvex

        if dim_box > 0:
            z_ref  = self.cp_params.z_ref.value   # (N, n_z)  or None on the very first call
            nu_ref = self.cp_params.nu_ref.value  # (N, n_nu)
            if z_ref is not None and nu_ref is not None:
                ref_full = np.concatenate([z_ref, nu_ref], axis=-1)  # (N, d)
                for nodes, col, side, bound in box_layout:
                    J[nodes, row, col] = 1.0
                    val    = ref_full[nodes, col]
                    active = (val <= bound + box_tol) if side == 'lower' else (val >= bound - box_tol)
                    row_weight[nodes, row] = active.astype(float)
                    row += 1
            else:
                row += dim_box

        J *= np.sqrt(row_weight)[:, :, np.newaxis]

        idx = np.arange(dim_total)
        Lambda = np.zeros((N, dim_total, dim_total))
        Lambda[:, idx, idx] = lam

        Jt   = np.swapaxes(J, -1, -2)                 # (N, d, dim_total)
        gram = J @ Jt + Lambda                        # (N, dim_total, dim_total)
        P    = np.eye(d)[np.newaxis, :, :] - Jt @ np.linalg.pinv(gram) @ J

        if bool(getattr(self.flags, 'active_set_auto_kappa', False)):
            kappa = self._auto_kappa()
        else:
            kappa_default = 1.0 if bool(getattr(self.flags, 'active_set_preserve_lm', False)) else 0.0
            kappa = float(getattr(self.flags, 'active_set_lm_floor_fraction', kappa_default))
        kappa = min(max(kappa, 0.0), 1.0)

        Pt = np.swapaxes(P, -1, -2)
        H_extra = H - H_base
        if kappa == 0.0:
            return P @ H @ Pt
        H_floor = (1.0 - kappa) * (P @ H_base @ Pt) + kappa * H_base
        return H_floor + P @ H_extra @ Pt

    def _adapt_levenberg(self, iteration: int) -> None:
        if iteration < 2:
            self._last_merit_ratio = 1.0
            return
        prev = self.iter_data_list[-1]
        prev_prev = self.iter_data_list[-2]
        merit_curr = prev.cost + prev.penalty_cost
        merit_prev = prev_prev.cost + prev_prev.penalty_cost
        ratio = merit_curr / merit_prev if merit_prev != 0 else 1.0
        self._last_merit_ratio = ratio
        if ratio > 1.0:
            self.lm_mu = min(self.lm_mu * 3.0, 1e4)
        elif ratio > 0.99:
            self.lm_mu = min(self.lm_mu * 1.5, 1e4)
        else:
            self.lm_mu = max(self.lm_mu * 0.7, 1e-6)

    def _auto_kappa(self) -> float:
        """Auto-tuned `kappa`: persists across iterations, nudged toward 1
        on a bad/flat merit ratio and toward 0 on a good one.
        """
        kappa = getattr(self, '_auto_kappa_state', 0.0)
        ratio = getattr(self, '_last_merit_ratio', 1.0)
        if ratio > 1.0:
            kappa = min(kappa + 0.25, 1.0)
        elif ratio > 0.99:
            kappa = min(kappa + 0.05, 1.0)
        else:
            kappa = max(kappa - 0.1, 0.0)
        self._auto_kappa_state = kappa
        return kappa

    def read_solution(self) -> None:
        self._dz_new  = self.dz.value
        self._dnu_new = self.dnu.value

    def apply_step(self, alpha: float) -> None:
        dz_new  = self._dz_new
        dnu_new = self._dnu_new

        self.current_iter_data.dz  = alpha * dz_new
        self.current_iter_data.dnu = alpha * dnu_new
        self.current_iter_data.alpha = alpha

        z_new  = self.current_iter_data.z_opt  + alpha * dz_new
        nu_new = self.current_iter_data.nu_opt + alpha * dnu_new

        self.current_iter_data.z_opt  = z_new
        self.current_iter_data.nu_opt = nu_new

        x_opt_new, t_opt_new, beta_opt_new, u_opt_new, s_opt_new = self.index_map.unpack_znu(z_new, nu_new)

        T_opt_new  = float(np.asarray(t_opt_new[-1]).ravel()[0])

        self.current_iter_data.x_opt    = x_opt_new
        self.current_iter_data.t_opt    = t_opt_new
        self.current_iter_data.T_opt    = T_opt_new * self.nondim.time_scale
        self.current_iter_data.beta_opt = beta_opt_new
        self.current_iter_data.u_opt    = u_opt_new
        self.current_iter_data.s_opt    = s_opt_new
        z_jnp  = jnp.asarray(z_new)
        nu_jnp = jnp.asarray(nu_new)
        self.current_iter_data.cost         = sum(c.evaluate_merit_cost(z_jnp, nu_jnp, self.params) for c in self.costs.values())
        self.current_iter_data.penalty_cost = sum(c.evaluate_merit(z_jnp, nu_jnp, self.params) for c in self.constraints.values())

        for constraint in self.constraints.values():
            constraint.read_vb(self)

        for constraint in self.constraints.values():
            constraint.update_current_iter_data(self)

        self.current_iter_data.iter_num += 1

        self.current_iter_data.vb     = AttrDict({c.name: c.vb     for c in self.constraints.values() if c.shape is not None})
        self.current_iter_data.W      = AttrDict({c.name: c.W      for c in self.constraints.values() if c.shape is not None})
        self.current_iter_data.dual   = AttrDict({c.name: c.dual   for c in self.constraints.values() if c.shape is not None})
        self.current_iter_data.W_p    = AttrDict({c.name: c.W_p    for c in self.constraints.values() if c.vb_type == "split"})
        self.current_iter_data.W_m    = AttrDict({c.name: c.W_m    for c in self.constraints.values() if c.vb_type == "split"})
        self.current_iter_data.dual_p = AttrDict({c.name: c.dual_p for c in self.constraints.values() if c.vb_type == "split"})
        self.current_iter_data.dual_m = AttrDict({c.name: c.dual_m for c in self.constraints.values() if c.vb_type == "split"})

        convergence.check_convergence_tolerance(self)

    def record_iter_data(self) -> None:
        self.iter_data_list.append(copy.deepcopy(self.current_iter_data))

    def update_W_dual(self, alpha: float = 1.0) -> None:
        for constraint in self.constraints.values():
            constraint.update_W_dual(self, alpha)

def _psd_sqrt(H_batch):
    eigvals, eigvecs = np.linalg.eigh(H_batch)
    eigvals_reg = np.maximum(eigvals, 0.0)
    sqrt_eigvals = np.sqrt(eigvals_reg)
    return sqrt_eigvals[..., :, np.newaxis] * np.transpose(eigvecs, (0, 2, 1))
