import copy
import time

import numpy as np
import cvxpy as cp
import jax.numpy as jnp

from trajopt.segment import Segment
from trajopt.methods.scp import initial_guess
from trajopt.methods.scp import convergence
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
        self.eps_dyn   = dyn.eps if dyn is not None else np.full(self.index_map.n.z, 1e-4)
        self.eps_state = self.eps_dyn[self.index_map.indices.z.state]
        self.eps_cost  = np.atleast_1d(1e-4)

        initial_guess.set_initial_guess(segment, self)

        self.lagrangian_duals = AttrDict()
        self.lagrangian_duals.dynamics = np.zeros((segment.index_map.N.all - 1, segment.index_map.n.z))

        self.iter_data_list = []

        self.current_iter_data = recursive_attrdict({
            "iter_num": 0,
            "z_opt": self.initial_guess.z,
            "nu_opt": self.initial_guess.nu,
            "cost": 0.0,
            "vb":   AttrDict({c.name: c.vb   for c in self.constraints.values() if c.shape is not None}),
            "W":    AttrDict({c.name: c.W    for c in self.constraints.values() if c.shape is not None}),
            "dual": AttrDict({c.name: c.dual for c in self.constraints.values() if c.shape is not None}),
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

        self.cp_vars.dx    = cp.Variable((N, n_x),    name="dx")
        self.cp_vars.dbeta = cp.Variable((N, n_ctcs), name="dbeta") if n_ctcs > 0 else None
        self.cp_vars.du    = cp.Variable((N, n_u),    name="du")

        if bool(self.flags.free_final_time):
            self.cp_vars.dt = cp.Variable((N, n_t), name="dt")
            self.cp_vars.ds = cp.Variable((N, 1),   name="ds")
        else:
            self.cp_vars.dt = cp.Constant(np.zeros((N, n_t)))
            self.cp_vars.ds = cp.Constant(np.zeros((N, 1)))

        dz_components = [self.cp_vars.dx, self.cp_vars.dt]

        if n_ctcs > 0:
            dz_components.append(self.cp_vars.dbeta)

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

        H = np.zeros((N, n_z + n_nu, n_z + n_nu))

        for constraint in self.constraints.values():
            constraint.accumulate_hessian(self, H)

        for cost in self.costs.values():
            cost.accumulate_hessian(self, H)

        self.cp_params.L.value = _psd_sqrt(H, delta=1e-6)

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
        self.current_iter_data.cost     = float(np.sum(self.cp_cost.value))

        for constraint in self.constraints.values():
            constraint.read_vb(self)

        lam_sp_dyn = np.array([c.dual_value for c in self.cp_dyn_constraints])
        self.lagrangian_duals.dynamics = (1.0 - alpha) * self.lagrangian_duals.dynamics + alpha * lam_sp_dyn

        for constraint in self.constraints.values():
            constraint.update_current_iter_data(self)

        self.current_iter_data.iter_num += 1

        self.current_iter_data.vb   = AttrDict({c.name: c.vb   for c in self.constraints.values() if c.shape is not None})
        self.current_iter_data.W    = AttrDict({c.name: c.W    for c in self.constraints.values() if c.shape is not None})
        self.current_iter_data.dual = AttrDict({c.name: c.dual for c in self.constraints.values() if c.shape is not None})

        convergence.check_convergence_tolerance(self)

    def record_iter_data(self) -> None:
        self.iter_data_list.append(copy.deepcopy(self.current_iter_data))

    def update_W_dual(self, alpha: float = 1.0) -> None:
        for constraint in self.constraints.values():
            constraint.update_W_dual(self, alpha)

def _psd_sqrt(H_batch, delta=1e-6):
    eigvals, eigvecs = np.linalg.eigh(H_batch)
    eigvals_reg = np.maximum(eigvals, delta)
    sqrt_eigvals = np.sqrt(eigvals_reg)
    return sqrt_eigvals[..., :, np.newaxis] * np.transpose(eigvecs, (0, 2, 1))
