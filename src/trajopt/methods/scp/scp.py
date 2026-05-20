import numpy as np
import cvxpy as cp

import copy
import time

from trajopt.methods.scp import convergence, initial_guess, discretize
from trajopt.utils import tools
from trajopt.utils.tools import AttrDict, recursive_attrdict

from trajopt.methods.scp.subproblem_constraint_types import (
    CREATE_PARAMETER_REGISTRY,
    CREATE_VARIABLE_REGISTRY,
    CREATE_CONSTRAINT_REGISTRY,
    CREATE_COST_REGISTRY,
    UPDATE_PARAMETER_REGISTRY,
    UPDATE_CURRENT_ITER_DATA_REGISTRY,
    INITIALIZE_METHOD_REGISTRY,
    armijo_line_search,
)

# =========================
# Subproblem (build-once)
# =========================

class SCP:
    """Reusable convex SCP with full baseline functionality & DPP updates."""

    def __init__(self, config, index_map, problem) -> None:
        self.problem   = problem
        self.index_map = self.problem.index_map
        self.indices   = self.index_map.indices

        # configs
        self.config         = config 
        self.flags          = config.method.flags
        self.conv_config    = config.method.conv
        self.penalty_config = config.method.penalty
        self.weights_config = config.method.weights

        self.n = self.index_map.n
        self.N = self.index_map.N

        if bool(self.flags.free_final_time):
            if "free_final_time" not in self.problem.constraints.constraint_type_list:
                self.problem.constraints.constraint_type_list.append("free_final_time")
                
        self.W_stack, self.dual_stack, self.vb_stack = self.create_W_dual_vb_stack(self.penalty_config)
        self.W_stack, self.dual_stack                = self.initialize_W_dual(self.penalty_config, self.W_stack, self.dual_stack)
        self.eps_stack                               = convergence.create_eps_stack(self)
        
        self.initialize()

        # cvxpy problem definition
        self.cp_params      = AttrDict()
        self.cp_vars        = AttrDict()
        self.cp_constraints = []
        self.cp_cost        = 0

        self.create_cvxpy_parameters()
        self.create_cvxpy_variables()
        self.create_cvxpy_constraints()
        self.create_cvxpy_cost()

        self.cp_subproblem  = cp.Problem(cp.Minimize(self.cp_cost), self.cp_constraints)
        total_param_scalars = sum(p.size for p in self.cp_subproblem.parameters())
        
        print("subproblem stats:")
        print("------------------------------------------------------------")
        print(f"total number of cvxpy parameters: {total_param_scalars}")
        print(f"total number of cvxpy constraints: {len(self.cp_constraints)}")
        print(f"is DPP: {self.cp_subproblem.is_dcp(dpp=True)}")

    def initialize(self):
        problem = self.problem

        self.initial_guess = AttrDict()

        # ---- Time grid initialization ----
        self.Ts_init          = self.config.method.guess.T_init / self.problem.nondim.time_scale
        t_init                = np.linspace(0.0, self.Ts_init, self.index_map.N.time_grid).reshape(-1, 1)
        dt_init               = np.diff(t_init, axis=0)
        self.initial_guess.t  = t_init.reshape(-1)
        self.initial_guess.dt = dt_init
        self.dt_init_min      = float(np.min(dt_init))

        discretize.compile_rk4_discretization(problem, self)

        dT_max = getattr(self.config.method.guess, "dT_max", None)
        if dT_max is not None:
            self.ddt_max = dT_max / ((self.index_map.N.time_grid - 1) * self.problem.nondim.time_scale)
        else:
            self.ddt_max = np.inf

        initial_guess.set_initial_guess(problem, self)

        self.lagrangian_duals = AttrDict()
        self.lagrangian_duals.dynamics = np.zeros((self.N.time_grid - 1, self.n.z))
        if hasattr(self.n, 'nonconvex_inequality'):
            self.lagrangian_duals.nonconvex_inequality = np.zeros((self.N.time_grid, self.n.nonconvex_inequality))

        # --------------------------
        # Initialize unified history
        # --------------------------
        self.iter_data_list = []

        self.current_iter_data = recursive_attrdict(
                {
                    "iter_num": 0,
                    "z_opt": self.initial_guess.z,
                    "nu_opt": self.initial_guess.nu,
                    "cost": 0.0,
                    "vb": copy.deepcopy(self.vb_stack),
                    "W": copy.deepcopy(self.W_stack),
                    "dual": copy.deepcopy(self.dual_stack),
                }
            )
        
        self.iter_data_list.append(copy.deepcopy(self.current_iter_data))

        for key, fn in INITIALIZE_METHOD_REGISTRY.items():
            fn(self)

    def create_W_dual_vb_stack(self, penalty_config):
        
        W_stack    = tools.AttrDict()
        dual_stack = tools.AttrDict()
        vb_stack   = tools.AttrDict()

        type_names = self.problem.constraints.constraint_type_list

        for cnstr_type in type_names:
            if cnstr_type not in penalty_config:
                continue
            if penalty_config[cnstr_type].W.penalty or penalty_config[cnstr_type].dual.penalty:
                N_type = self.index_map.N[cnstr_type]
                n_type = self.index_map.n[cnstr_type]
                shape  = (N_type, n_type)

                vb_stack[cnstr_type] = np.zeros(shape)

            if penalty_config[cnstr_type].W.penalty:
                W_stack[cnstr_type] = np.zeros(shape)
            
            if penalty_config[cnstr_type].dual.penalty:
                dual_stack[cnstr_type] = np.zeros(shape)

        return W_stack, dual_stack, vb_stack

    def initialize_W_dual(self, penalty_config, W_stack, dual_stack):
        
        for cnstr_type in W_stack.keys():
                W_stack[cnstr_type] += penalty_config[cnstr_type].W.init

        for cnstr_type in dual_stack.keys():
                W_stack[cnstr_type] += penalty_config[cnstr_type].dual.init

        return  W_stack, dual_stack
    
    def create_W_dual_parameters(self):
        self.cp_params.W_sqrt = tools.AttrDict()
        self.cp_params.dual   = tools.AttrDict()

        for cnstr_type in self.W_stack.keys():
            N_type = self.index_map.N[cnstr_type]
            n_type = self.index_map.n[cnstr_type]
            shape  = (N_type, n_type)
            self.cp_params.W_sqrt[cnstr_type] = cp.Parameter(shape, nonneg=True, name=f"W_{cnstr_type}_sqrt", value=np.zeros(shape))
        
        for cnstr_type in self.dual_stack.keys():
            N_type = self.index_map.N[cnstr_type]
            n_type = self.index_map.n[cnstr_type]
            shape  = (N_type, n_type)
            self.cp_params.dual[cnstr_type] = cp.Parameter(shape, name=f"dual_{cnstr_type}", value=np.zeros(shape))

    def create_vb_cvxpy_variables(self):
        self.cp_vars.vb_stack = tools.AttrDict()

        for cnstr_type in self.W_stack.keys() | self.dual_stack.keys():

            vb_type = self.penalty_config[cnstr_type].vb

            N_type = self.index_map.N[cnstr_type]
            n_type = self.index_map.n[cnstr_type]
            shape = (N_type, n_type)
            
            if vb_type == "standard":
                self.cp_vars.vb_stack[cnstr_type] = cp.Variable(shape, name=f"vb_{cnstr_type}")

    def create_cvxpy_parameters(self) -> None:

        N, n_z, n_nu = self.N.time_grid, self.n.z, self.n.nu

        self.cp_params.z_ref  = cp.Parameter((N, n_z),  name="z_ref")
        self.cp_params.nu_ref = cp.Parameter((N, n_nu), name="nu_ref")

        self.x_ref, self.t_ref, self.beta_ref, self.u_ref, self.s_ref = self.index_map.unpack_znu(self.cp_params.z_ref, self.cp_params.nu_ref)

        self.cp_params.w_cost      = cp.Constant(value=self.config.method.weights.w_cost, name="w_cost")
        self.cp_params.tr_z        = cp.Parameter(nonneg=True, name="tr_z")
        self.cp_params.tr_nu       = cp.Parameter(nonneg=True, name="tr_u")

        self.cp_params.w_cost_times_dcostdx = cp.Parameter((N, n_z),  name="w_cost_times_dcostdx")
        self.cp_params.w_cost_times_dcostdu = cp.Parameter((N, n_nu), name="w_cost_times_dcostdu")
        self.cp_params.w_cost_times_cost0   = cp.Parameter((N,),      name="w_cost_times_cost0")

        self.cp_params.eps_ctcs = cp.Parameter(nonneg=True, name="eps_ctcs")

        self.create_W_dual_parameters()

        print("  [parameters]")
        for constraint_type in self.problem.constraints.constraint_type_list:
            if constraint_type in CREATE_PARAMETER_REGISTRY.keys():
                print(f"    + {constraint_type}")
                CREATE_PARAMETER_REGISTRY[constraint_type](self)

    def create_cvxpy_variables(self):
        N, n_x, n_t, n_u, n_ctcs = (self.N.time_grid, self.n.state, self.n.time, self.n.control, self.n.ctcs)

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

        print("  [variables]")
        print(f"    dx: {self.cp_vars.dx.shape}, du: {self.cp_vars.du.shape}")
        print(f"    dz: {self.dz.shape}, dnu: {self.dnu.shape}")
        print(f"    free_final_time: {bool(self.flags.free_final_time)}")
        print(f"    W_stack keys: {list(self.W_stack.keys())}")
        print(f"    dual_stack keys: {list(self.dual_stack.keys())}")

        self.create_vb_cvxpy_variables()
        print(f"    vb_stack keys: {list(self.cp_vars.vb_stack.keys())}")

        for constraint_type in self.problem.constraints.constraint_type_list:
            if constraint_type in CREATE_VARIABLE_REGISTRY:
                print(f"    + variable: {constraint_type}")
                CREATE_VARIABLE_REGISTRY[constraint_type](self)

    def create_cvxpy_constraints(self):
        print("  [constraints]")
        for constraint_type in self.problem.constraints.constraint_type_list:
            if constraint_type in CREATE_CONSTRAINT_REGISTRY:
                print(f"    + {constraint_type}")
                CREATE_CONSTRAINT_REGISTRY[constraint_type](self)

    def create_cvxpy_cost(self):
        print("  [costs]")
        for cost_type in self.problem.costs.cost_type_list:
            if cost_type in CREATE_COST_REGISTRY:
                print(f"    + {cost_type}")
                CREATE_COST_REGISTRY[cost_type](self)

        METHOD_COSTS = ["trust_region", "quadratic_penalty", "dual"]
        print("  [method costs]")
        for cost_type in METHOD_COSTS:
            print(f"    + {cost_type}")
            CREATE_COST_REGISTRY[cost_type](self)
        
    def update_cvxpy_parameters(self):

        z_opt  = self.current_iter_data.z_opt
        nu_opt = self.current_iter_data.nu_opt

        self.cp_params.z_ref.value  = z_opt
        self.cp_params.nu_ref.value = nu_opt

        self.x_ref, self.t_ref, self.beta_ref, self.u_ref, self.s_ref = self.index_map.unpack_znu(z_opt, nu_opt)

        disc_start_time = time.perf_counter()
        for fcn_type, fcn in UPDATE_PARAMETER_REGISTRY.items():
            fcn(self)
        disc_end_time = time.perf_counter()

        self.current_iter_data.discretization_time = (disc_end_time - disc_start_time) * 1000

        self.cp_params.tr_z.value  = (1 / self.config.method.weights.alpha_z)
        self.cp_params.tr_nu.value = (1 / self.config.method.weights.alpha_nu)

        self.cp_params.eps_ctcs.value = float(self.config.method.conv.eps_ctcs)

        for cnstr_type, W_val in self.W_stack.items():
            W_param = self.cp_params.W_sqrt.get(cnstr_type)
            if W_param is not None and hasattr(W_param, "value"):
                W_param.value = np.sqrt(W_val)
        
        for cnstr_type, dual_val in self.dual_stack.items():
            dual_param = self.cp_params.dual.get(cnstr_type)
            if dual_param is not None and hasattr(dual_param, "value"):
                dual_param.value = dual_val

    def update_current_iter_data(self):

        dz_new  = self.dz.value
        dnu_new = self.dnu.value

        self.current_iter_data.parse_time = self.cp_subproblem.compilation_time * 1000.0
        self.current_iter_data.solve_time = self.cp_subproblem.solver_stats.solve_time * 1000.0

        self.current_iter_data.dz  = dz_new
        self.current_iter_data.dnu = dnu_new

        alpha = armijo_line_search(self, dz_new, dnu_new)
        # alpha = 1.0
        self.current_iter_data.alpha = alpha

        z_new  = self.current_iter_data.z_opt  + alpha * dz_new
        nu_new = self.current_iter_data.nu_opt + alpha * dnu_new

        self.current_iter_data.z_opt  = z_new
        self.current_iter_data.nu_opt = nu_new

        x_opt_new, t_opt_new, beta_opt_new, u_opt_new, s_opt_new = self.index_map.unpack_znu(z_new, nu_new)

        T_opt_new  = float(np.asarray(t_opt_new[-1]).ravel()[0])

        self.current_iter_data.x_opt    = x_opt_new
        self.current_iter_data.t_opt    = t_opt_new
        self.current_iter_data.T_opt    = T_opt_new * self.problem.nondim.time_scale
        self.current_iter_data.beta_opt = beta_opt_new
        self.current_iter_data.u_opt    = u_opt_new
        self.current_iter_data.s_opt    = s_opt_new
        self.current_iter_data.cost     = float(np.sum(self.cp_cost.value) / self.cp_params.w_cost.value)

        for cnstr_type in self.vb_stack:
            vb_var = self.cp_vars.vb_stack.get(cnstr_type)
            if vb_var is None:
                continue
            self.vb_stack[cnstr_type] = np.array(vb_var.value)

        # update Lagrangian multiplier estimates (for regularized Hessian)
        lam_sp_dyn = np.array([c.dual_value for c in self.cp_dyn_constraints])
        self.lagrangian_duals.dynamics = (1.0 - alpha) * self.lagrangian_duals.dynamics + alpha * lam_sp_dyn

        if hasattr(self, 'cp_ineq_constraints') and self.cp_ineq_constraints:
            lam_sp_ineq = np.array([c.dual_value for c in self.cp_ineq_constraints])
            self.lagrangian_duals.nonconvex_inequality = ((1.0 - alpha) * self.lagrangian_duals.nonconvex_inequality + alpha * lam_sp_ineq)

        for constraint_type in self.problem.constraints.constraint_type_list:
            if constraint_type in UPDATE_CURRENT_ITER_DATA_REGISTRY:
                UPDATE_CURRENT_ITER_DATA_REGISTRY[constraint_type](self)

        self.current_iter_data.iter_num  = len(self.iter_data_list)

        self.current_iter_data.vb   = copy.deepcopy(self.vb_stack)
        self.current_iter_data.W    = copy.deepcopy(self.W_stack)
        self.current_iter_data.dual = copy.deepcopy(self.dual_stack)

        convergence.check_convergence_tolerance(self)

        self.update_W_dual()

        self.iter_data_list.append(copy.deepcopy(self.current_iter_data))

    def update_W_dual(self):

        alpha = self.current_iter_data.alpha

        for cnstr_type, cnstr_penalty_cfg in self.penalty_config.items():
            if cnstr_type not in self.W_stack:
                continue

            vb   = self.vb_stack[cnstr_type]
            W    = self.W_stack[cnstr_type]
            dual = self.dual_stack[cnstr_type]

            eps = np.atleast_1d(self.eps_stack[cnstr_type])

            W_cfg = cnstr_penalty_cfg["W"]
            if W_cfg["autotune"]:
                eps_floor  = W_cfg["eps_floor"]
                fac_eps    = W_cfg["fac_eps"]
                fac_target = W_cfg["fac_target"]
                eps_target = fac_target * eps
                Wh = np.where(eps_target > 0, np.abs(W * vb / eps_target), 0.0)

                if np.sum(W) > 0:
                    Wh = np.maximum(Wh, eps_floor)

                W_max = 1e3 if self.current_iter_data.iter_num < 5 else 1e8
                Wh = np.minimum(Wh, W_max)
                Wh = np.maximum(Wh, W)
                self.W_stack[cnstr_type] = Wh

            dual_cfg = cnstr_penalty_cfg["dual"]
            if dual_cfg["autotune"]:

                dual_new = dual + W * vb

                if cnstr_type in {"nonconvex_inequality", "convex_inequality"}:
                    dual_new = np.maximum(0, dual_new)

                self.dual_stack[cnstr_type] = dual_new
                # satisfied = np.abs(vb) <= eps
                # self.dual_stack[cnstr_type] = np.where(satisfied, dual, dual_new)

    def solve(self):

        print("-" * 172)
        print("  Iteration |  Discretization |   Solve   |    Parse   |  log(dx/eps) | log(vb_ineq/eps) | log(vb_term/eps) | log(vb_dyn/eps) | Solve status | alpha |  Time of    |   Cost    ")
        print("            |    time [ms]    | time [ms] |  time [ms] |     (state)  |    (ncvx_ineq)   |      (terminal)  |    (dynamics)   |              |       |  Flight [s] |           ")
        print("-" * 172)

        max_iter = int(self.config.method.conv.iter_max)

        for _ in range(max_iter + 1):
            self.update_cvxpy_parameters()
            self.cp_subproblem.solve(warm_start=False, **self.config.method.solver_opts)
            
            if self.cp_subproblem.status not in {"optimal", "optimal_inaccurate"}:
                print(f"Terminated from non-optimal convex subproblem! Status: {self.cp_subproblem.status}")
                break

            self.update_current_iter_data()
            self.display_subprob_status()

            if self.iter_data_list[-1].converged:
                print("Terminated from convergence criteria!")
                break

        if self.iter_data_list[-1].iter_num > 0 and not self.iter_data_list[-1].converged:
            print("Terminated from hitting maximum iterations!")
    
    def display_subprob_status(self):
        current_iter_data = self.current_iter_data

        with np.errstate(divide='ignore'):
            log_dz_ratio      = float(np.log10(current_iter_data.chk.dz))
            log_vb_ineq_ratio = float(np.log10(current_iter_data.chk.nonconvex_inequality))
            log_vb_term_ratio = float(np.log10(current_iter_data.chk.final_state))
            log_vb_dyn_ratio  = float(np.log10(current_iter_data.chk.dynamics))

        solve_stat  = current_iter_data.status
        iter_num    = int(current_iter_data.iter_num)
        alpha       = float(current_iter_data.get("alpha", 1.0))

        T    = float(current_iter_data.T_opt)
        cost = float(current_iter_data.cost)

        discretization_ms  = float(current_iter_data.discretization_time)
        solve_ms = float(current_iter_data.solve_time)
        parse_ms = float(current_iter_data.parse_time)

        print(
            "{:^12d}|{:^17.1f}|{:^11.1f}|{:^12.1f}|{:^+14.1f}|{:^+18.1f}|{:^+18.1f}|{:^+17.1f}|{:^14s}|{:^7.3f}|{:^13.2f}|{:^11.1f}".format(
                iter_num,
                discretization_ms,
                solve_ms,
                parse_ms,
                log_dz_ratio,
                log_vb_ineq_ratio,
                log_vb_term_ratio,
                log_vb_dyn_ratio,
                str(solve_stat),
                alpha,
                T,
                cost
            )
        )
