import copy
import time

import numpy as np
import cvxpy as cp

from trajopt.segment import Segment
from trajopt.methods.common import initial_guess
from trajopt.methods.common import convergence
from trajopt.methods.common import penalties
import trajopt.methods.dev.scvx.scp_constraints.scp_constraint_types as scp_constraint_type_module
import trajopt.methods.dev.scvx.scp_costs.scp_cost_types as scp_cost_type_module
from trajopt.utils.tools import AttrDict, recursive_attrdict, expand_to_array_if_scalar

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

        self.free_final_time = self.derive_free_final_time()

        self.initialize()

        self.cp_params            = AttrDict()
        self.cp_vars              = AttrDict()
        self.cp_constraints       = []
        self.cp_cost              = 0
        self.cp_subproblem_status = None

        self.create_cvxpy_parameters()
        self.create_cvxpy_variables()
        self.create_cvxpy_constraints()
        self.create_cvxpy_cost()

    def find_constraint(self, cnstr_type):
        """The segment's constraint of the given type, or None."""
        return next((c.constraint for c in self.constraints.values() if c.type == cnstr_type), None)

    def derive_free_final_time(self) -> bool:
        """False only when initial_time and a fixed final_time pin both ends."""
        initial = self.find_constraint("initial_time")
        final   = self.find_constraint("final_time")

        if initial is not None and self.find_constraint("time_continuity") is not None:
            raise ValueError(
                f"segment '{self.name}' declares both initial_time and time_continuity; "
                "its start epoch comes from the preceding segment, so drop the initial_time"
            )

        return not (initial is not None and final is not None and final.is_fixed)

    def initialize(self) -> None:
        segment = self.segment

        self.initial_guess = AttrDict()

        # the guess supplies whichever end the constraints do not give
        cfg_guess             = segment.guess
        initial_time_cnstr    = self.find_constraint("initial_time")
        final_time_cnstr      = self.find_constraint("final_time")

        t_start_nd            = (initial_time_cnstr.value if initial_time_cnstr is not None
                                 else getattr(cfg_guess, 't_start', 0.0) / self.nondim.time_scale)

        if not self.free_final_time:
            t_stop_nd = final_time_cnstr.fixed_value
        else:
            t_stop_nd = cfg_guess.t_stop / self.nondim.time_scale

        self.Ts_init          = t_stop_nd - t_start_nd
        t_init                = np.linspace(t_start_nd, t_stop_nd, self.index_map.N.all)
        dt_init               = np.diff(t_init)
        self.initial_guess.t  = t_init
        self.initial_guess.dt = dt_init

        for constraint in self.constraints.values():
            constraint.compile(self)
            constraint.init_penalty(self)

        dyn = next((c for c in self.constraints.values() if c.type == "dynamics"), None)
        self.eps_dyn   = dyn.penalty_state.eps.copy() if dyn is not None else np.full(self.index_map.n.z, 1e-4)
        self.eps_dyn[self.index_map.indices.z.running_cost] = np.inf

        self.eps_state = expand_to_array_if_scalar(getattr(self.flags, 'eps_state', 1e-4), self.index_map.n.state)
        self.eps_cost  = np.atleast_1d(float(getattr(self.penalty_config, 'eps_cost', 1e-4)))
        self.w_cost    = float(getattr(self.penalty_config, 'w_cost', 1.0))

        initial_guess.set_initial_guess(segment, self)

        self.iter_data_list = []

        self.current_iter_data = recursive_attrdict({
            "iter_num": 0,
            "z_opt": self.initial_guess.z,
            "nu_opt": self.initial_guess.nu,
            "t_start": float(self.initial_guess.t[0]) * self.nondim.time_scale,
            "t_final": float(self.initial_guess.t[-1]) * self.nondim.time_scale,
            "cost": 0.0,
            "vb":     AttrDict({c.name: c.penalty_state.vb     for c in self.constraints.values() if c.shape is not None}),
            "W":      AttrDict({c.name: c.penalty_state.W      for c in self.constraints.values() if c.shape is not None}),
            "dual":   AttrDict({c.name: c.penalty_state.dual   for c in self.constraints.values() if c.shape is not None}),
            "W_p":    AttrDict({c.name: c.penalty_state.W_p    for c in self.constraints.values() if c.penalty_state.vb_type == "split"}),
            "W_m":    AttrDict({c.name: c.penalty_state.W_m    for c in self.constraints.values() if c.penalty_state.vb_type == "split"}),
            "dual_p": AttrDict({c.name: c.penalty_state.dual_p for c in self.constraints.values() if c.penalty_state.vb_type == "split"}),
            "dual_m": AttrDict({c.name: c.penalty_state.dual_m for c in self.constraints.values() if c.penalty_state.vb_type == "split"}),
        })

        self.iter_data_list.append(copy.deepcopy(self.current_iter_data))

    def create_cvxpy_parameters(self) -> None:
        N       = self.index_map.N.all
        n_z     = self.index_map.n.z
        n_nu    = self.index_map.n.nu

        self.cp_params.z_ref  = cp.Parameter((N, n_z),  name="z_ref")
        self.cp_params.nu_ref = cp.Parameter((N, n_nu), name="nu_ref")

        self.x_ref, self.t_ref, self.beta_ref, self.u_ref, self.s_ref = self.index_map.unpack_znu(self.cp_params.z_ref, self.cp_params.nu_ref)

        self.cp_params.tr_x = cp.Parameter(nonneg=True, name="tr_x")
        self.cp_params.tr_t = cp.Parameter(nonneg=True, name="tr_t")
        if self.index_map.n.ctcs > 0:
            self.cp_params.tr_ctcs = cp.Parameter(nonneg=True, name="tr_ctcs")
        if self.index_map.n.running_cost > 0:
            self.cp_params.tr_gamma = cp.Parameter(nonneg=True, name="tr_gamma")
        self.cp_params.tr_u = cp.Parameter(nonneg=True, name="tr_u")
        self.cp_params.tr_s = cp.Parameter(nonneg=True, name="tr_s")

        self.cp_params.dcostdx = cp.Parameter((N, n_z),  name="dcostdx")
        self.cp_params.dcostdu = cp.Parameter((N, n_nu), name="dcostdu")
        self.cp_params.cost0   = cp.Parameter((N,),      name="cost0")

        if self.flags.discretize == "ps":
            self.cp_params.tau          = cp.Parameter((N,), name="tau")
            self.cp_params.tau.value    = self.ps_tau_norm
            self.cp_params.ps_t_offset  = cp.Parameter((N,), name="ps_t_offset", value=np.zeros(N))

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

        if self.free_final_time:
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

        if self.free_final_time:
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
        idx_z  = self.index_map.indices.z
        idx_nu = self.index_map.indices.nu

        self.cp_cost += self.cp_params.tr_x * cp.sum_squares(self.dz[:, idx_z.state])
        self.cp_cost += self.cp_params.tr_t * cp.sum_squares(self.dz[:, idx_z.time])
        if self.index_map.n.ctcs > 0:
            self.cp_cost += self.cp_params.tr_ctcs * cp.sum_squares(self.dz[:, idx_z.ctcs])
        if self.index_map.n.running_cost > 0:
            self.cp_cost += self.cp_params.tr_gamma * cp.sum_squares(self.dz[:, idx_z.running_cost])

        self.cp_cost += self.cp_params.tr_u * cp.sum_squares(self.dnu[:, idx_nu.control])
        self.cp_cost += self.cp_params.tr_s * cp.sum_squares(self.dnu[:, idx_nu.dilation_factor])

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

        disc_end_time = time.perf_counter()

        self.current_iter_data.discretization_time = (disc_end_time - disc_start_time) * 1000

        self.cp_params.tr_x.value = 1 / self._resolve_tr_step('z', 'x')
        self.cp_params.tr_t.value = 1 / self._resolve_tr_step('z', 't')
        if self.index_map.n.ctcs > 0:
            self.cp_params.tr_ctcs.value = 1 / self._resolve_tr_step('z', 'ctcs')
        if self.index_map.n.running_cost > 0:
            self.cp_params.tr_gamma.value = 1 / self._resolve_tr_step('z', 'gamma')
        self.cp_params.tr_u.value = 1 / self._resolve_tr_step('nu', 'u')
        self.cp_params.tr_s.value = 1 / self._resolve_tr_step('nu', 's')

        for constraint in self.constraints.values():
            constraint.update_penalty_parameters(self)

    def _resolve_tr_step(self, group: str, key: str) -> float:
        """Trust-region step size for one z/nu component.

        penalty.tr_step.<group> is a scalar (uniform across the group) or a
        mapping with a 'default' key plus named component overrides --
        same convention as a constraint's eps.
        """
        tr_step_cfg = getattr(self.penalty_config, 'tr_step', None)
        raw = getattr(tr_step_cfg, group, None) if tr_step_cfg is not None else None
        if raw is None:
            return 1.0
        if not hasattr(raw, 'items'):
            return float(raw)
        entries = dict(raw)
        default = entries.pop('default', 1.0)
        return float(entries.get(key, default))

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

        # times are measured from the start of the trajectory, not the segment
        t_start_new = float(np.asarray(t_opt_new[0]).ravel()[0])
        t_final_new = float(np.asarray(t_opt_new[-1]).ravel()[0])

        self.current_iter_data.x_opt    = x_opt_new
        self.current_iter_data.t_opt    = t_opt_new
        self.current_iter_data.t_start  = t_start_new * self.nondim.time_scale
        self.current_iter_data.t_final  = t_final_new * self.nondim.time_scale
        self.current_iter_data.beta_opt = beta_opt_new
        self.current_iter_data.u_opt    = u_opt_new
        self.current_iter_data.s_opt    = s_opt_new
        self.current_iter_data.cost     = self.cp_cost.value / self.w_cost

        for constraint in self.constraints.values():
            constraint.read_vb(self)

        self.current_iter_data.penalty_cost = sum(
            penalties.penalty_cost_value(c.penalty_state) for c in self.constraints.values()
        )

        for constraint in self.constraints.values():
            constraint.update_current_iter_data(self)

        self.current_iter_data.iter_num += 1

        self.current_iter_data.vb     = AttrDict({c.name: c.penalty_state.vb     for c in self.constraints.values() if c.shape is not None})
        self.current_iter_data.W      = AttrDict({c.name: c.penalty_state.W      for c in self.constraints.values() if c.shape is not None})
        self.current_iter_data.dual   = AttrDict({c.name: c.penalty_state.dual   for c in self.constraints.values() if c.shape is not None})
        self.current_iter_data.W_p    = AttrDict({c.name: c.penalty_state.W_p    for c in self.constraints.values() if c.penalty_state.vb_type == "split"})
        self.current_iter_data.W_m    = AttrDict({c.name: c.penalty_state.W_m    for c in self.constraints.values() if c.penalty_state.vb_type == "split"})
        self.current_iter_data.dual_p = AttrDict({c.name: c.penalty_state.dual_p for c in self.constraints.values() if c.penalty_state.vb_type == "split"})
        self.current_iter_data.dual_m = AttrDict({c.name: c.penalty_state.dual_m for c in self.constraints.values() if c.penalty_state.vb_type == "split"})

        convergence.check_convergence_tolerance(self)

    def record_iter_data(self) -> None:
        self.iter_data_list.append(copy.deepcopy(self.current_iter_data))

    def update_W_dual(self, alpha: float = 1.0) -> None:
        for constraint in self.constraints.values():
            constraint.update_W_dual(self, alpha)
