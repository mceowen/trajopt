from typing import Dict, Any, Optional, List
import time
import numpy as np
import cvxpy as cp
import jax.numpy as jnp
import copy

from trajopt.library.methods.subproblem_constraints import SubproblemConstraints
from trajopt.library.methods import discretize
from trajopt.library.methods import convergence
from trajopt.library.methods import hyperparameters as hp
from trajopt.utils import tools
from trajopt.utils.tools import recursive_attrdict

from trajopt.library.methods.subproblem_constraints import configure_penalty_weights


# =========================
# Subproblem (build-once)
# =========================
class Subproblem:
    """Reusable convex SCP with full baseline functionality & DPP updates.
    """
    def __init__(self, problem, method) -> None:
        self.problem = problem
        self.method  = method

        # Reference to method data
        self.flags      = method.flags
        self.index_map  = method.index_map          # Extract index_map
        self.n          = method.index_map.n        # Use index_map.n for all dimension queries
        self.N          = method.index_map.N        # Use index_map.N for all time-related queries
        self.indices    = method.index_map.indices  # Use index_map.indices for index arrays

        # Reference constraint data
        self.constraints                = SubproblemConstraints(problem=problem, method=method)
        self.W_stack, self.dual_stack   = self.constraints.stack_W_and_dual(problem, method)
        self.w                          = tools.AttrDict()

        # Build the DPP graph once
        self._create_parameters()
        self._create_variables()
        self.cp_constraints: List[cp.constraints.constraint.Constraint] = []
        self._build_constraints_once()
        self.cp_cost = self._build_cost_once()

        # Compile CVXPY problem once
        self.cp_subproblem = cp.Problem(cp.Minimize(self.cp_cost), self.cp_constraints)

        total_param_scalars = sum(p.size for p in self.cp_subproblem.parameters())
        print("subproblem stats:")
        print("------------------------------------------------------------")
        print(f"total number of parameters: {total_param_scalars}")

        # --------------------------
        # Initialize unified history
        # --------------------------
        self.iter_data: List[Dict[str, Any]] = [recursive_attrdict({
            "iter_num": 0,  # init only (no outputs yet)
            "z_opt": method.initial_guess.z,
            "nu_opt": method.initial_guess.nu,
            "conv_data": {
                "vb_ineq": np.zeros((self.N.time_grid, self.n.nonconvex_inequality)),
                "vb_dyn":  np.zeros((self.N.time_grid - 1, self.n.dynamics)),
                "vb_terminal": np.zeros((self.n.term_total, 1)),
            },
            # W and dual are None initially - will be initialized by configure_penalty_weights on first iteration
            "W": None,
            "dual": None,
            "penalty": copy.deepcopy(method.penalty),
        })]

    def _create_parameters(self) -> None:
        problem, method                 = self.problem, self.method
        N, n_x, n_t, n_u, n_z, n_nu, n_ctcs   \
            = self.N.time_grid, self.n.state, self.n.time, self.n.control, self.n.z, self.n.nu, self.n.ctcs

        # Linearized dynamics & trajectory references
        self.Ak                    = cp.Parameter((N - 1, n_z, n_z),      name="Ak")
        self.Bk                    = cp.Parameter((N - 1, n_z, n_nu),    name="Bk")
        self.Bkp                   = cp.Parameter((N - 1, n_z, n_nu),    name="Bkp")
        self.z_m                   = cp.Parameter((N, n_z),              name="z_minus")

        #self.Ak_bwd                = cp.Parameter((N - 1, n_z, n_z),     name="Ak_bwd")
        #self.Bk_bwd                = cp.Parameter((N - 1, n_z, n_nu),   name="Bk_bwd")
        #self.Bkp_bwd               = cp.Parameter((N - 1, n_z, n_nu),   name="Bkp_bwd")
        #self.z_m_bwd               = cp.Parameter((N, n_z),             name="z_minus_bwd")

        self.z_ref                 = cp.Parameter((N, n_z),      name="z_ref")
        self.nu_ref                = cp.Parameter((N, n_nu),    name="nu_ref")

        # Sliced physical reference data
        self.x_ref, self.t_ref, self.beta_ref, self.u_ref, self.s_ref \
            = self.index_map.unpack_znu(self.z_ref, self.nu_ref)

        self.nu_ref_sq         = cp.Parameter((N,), name="nu_ref_sq")

        # Path/NFZ/AUX linearized constraints
        if problem.constraints.get(ct=0, type="nonconvex_inequality"):
            self.dgdx       = cp.Parameter((N, problem.index_map.n.nonconvex_inequality, n_x),  name="dgdx")
            self.dgdu       = cp.Parameter((N, problem.index_map.n.nonconvex_inequality, n_u),  name="dgdu")
            self.g0         = cp.Parameter((N, problem.index_map.n.nonconvex_inequality),       name="g0")
        else:
            self.dgdx = self.dgdu = self.g0 = None

        # time-step scalar bounds
        if bool(self.flags.free_final_time):
            self.dt_min     = cp.Parameter(nonneg=True, name="dt_min")
            self.dt_max     = cp.Parameter(nonneg=True, name="dt_max")
            self.ddt_max    = cp.Parameter(nonneg=True, name="ddt_max")
            self.T_min      = cp.Parameter(nonneg=True, name="T_min")
            self.T_max      = cp.Parameter(nonneg=True, name="T_max")
        else:
            self.dt_min = self.dt_max = self.ddt_max = None

        # Weights & Trust Region weights (stored as simple AttrDict values)
        self.w.cost                     = cp.Constant(value= method.penalty.w_cost, name="w_cost") 
        self.w.tr_z                     = cp.Parameter(nonneg=True, name="tr_z")
        self.w.tr_u                     = cp.Parameter(nonneg=True, name="tr_u")

        self.w_cost_times_dcostdx       = cp.Parameter((N, 1, n_x),  name="w_cost_times_dcostdx")
        self.w_cost_times_dcostdu       = cp.Parameter((N, 1, n_u),  name="w_cost_times_dcostdu")
        self.w_cost_times_cost0         = cp.Parameter((N, 1),       name="w_cost_times_cost0")

        # Build W_sqrt as attribute dictionary with CVXPY parameters, initialized to zeros
        self.W_sqrt = tools.AttrDict()
        self.W_sqrt.nonconvex_inequality= cp.Parameter((N, max(self.n.nonconvex_inequality, 1)),  nonneg=True, name="W_ineq_sqrt",        value=np.zeros((N, max(self.n.nonconvex_inequality, 1))))
        self.W_sqrt.final_state         = cp.Parameter((max(self.n.term_total, 1),),                         nonneg=True, name="W_term_sqrt",        value=np.zeros((max(self.n.term_total, 1),)))
        self.W_sqrt.dynamics            = cp.Parameter((N - 1, max(n_z, 1)),                                  nonneg=True, name="W_dyn_sqrt",         value=np.zeros((N - 1, max(n_z, 1))))
        self.W_sqrt.plus_real           = cp.Parameter((max(self.N.pm_real, 1), max(self.n.plus_real, 1)),   nonneg=True, name="W_plus_real_sqrt",   value=np.zeros((max(self.N.pm_real, 1), max(self.n.plus_real, 1))))
        self.W_sqrt.minus_real          = cp.Parameter((max(self.N.pm_real, 1), max(self.n.minus_real, 1)),  nonneg=True, name="W_minus_real_sqrt",  value=np.zeros((max(self.N.pm_real, 1), max(self.n.minus_real, 1))))
        self.W_sqrt.plus_ctcs           = cp.Parameter((max(self.N.pm_ctcs, 1), max(self.n.plus_ctcs, 1)),   nonneg=True, name="W_plus_ctcs_sqrt",   value=np.zeros((max(self.N.pm_ctcs, 1), max(self.n.plus_ctcs, 1))))
        self.W_sqrt.minus_ctcs          = cp.Parameter((max(self.N.pm_ctcs, 1), max(self.n.minus_ctcs, 1)),  nonneg=True, name="W_minus_ctcs_sqrt",  value=np.zeros((max(self.N.pm_ctcs, 1), max(self.n.minus_ctcs, 1))))

        # duals (same ≥1-column pattern, unified inequality structure)
        self.dual = tools.AttrDict()
        self.dual.nonconvex_inequality  = cp.Parameter((N,  max(self.n.nonconvex_inequality, 1)), name="dual_ineq")
        self.dual.dynamics              = cp.Parameter((N - 1, max(n_z, 1)),              name="dual_dyn")
        self.dual.plus_real             = cp.Parameter((max(self.N.pm_real, 1), max(self.n.plus_real, 1)),     name="dual_plus_real")
        self.dual.minus_real            = cp.Parameter((max(self.N.pm_real, 1), max(self.n.minus_real, 1)),    name="dual_minus_real")
        self.dual.plus_ctcs             = cp.Parameter((max(self.N.pm_ctcs, 1), max(self.n.plus_ctcs, 1)),     name="dual_plus_ctcs")
        self.dual.minus_ctcs            = cp.Parameter((max(self.N.pm_ctcs, 1), max(self.n.minus_ctcs, 1)),    name="dual_minus_ctcs")
        self.dual.final_state           = cp.Parameter((max(self.n.term_total, 1),),                           name="dual_term")

        # CTCS epsilon (scalar)
        self.eps_ctcs                   = cp.Parameter(nonneg=True, name="eps_ctcs")

        self.constraint_params = {}
        for constraint in problem.constraints.get(ct=0, type="equality_bc"):
            if constraint.name == "initial_state":
                self.constraint_params[constraint.name] = {
                    "value": cp.Parameter(len(constraint.idx), name=f"{constraint.name}_value")
                }
                self.constraint_params[constraint.name]["value"].value = constraint.value


    # ============================================================
    # VARIABLE & PARAMETER CREATION
    # ============================================================
    def _create_variables(self) -> None:
        problem, method                 = self.problem, self.method
        N, n_x, n_t, n_u, n_z, n_nu, n_ctcs   \
            = self.N.time_grid, self.n.state, self.n.time, self.n.control, self.n.z, self.n.nu, self.n.ctcs

        # Create physical component variables individually
        self.dx     = cp.Variable((N, n_x), name="dx")
        self.dbeta  = cp.Variable((N, n_ctcs), name="dbeta") if n_ctcs > 0 else None
        self.du     = cp.Variable((N, n_u), name="du")

        # Time and dilation perturbations: variable if free_final_time, else constant zeros
        if bool(self.flags.free_final_time):
            self.dt = cp.Variable((N, n_t), name="dt")
            self.ds = cp.Variable((N, 1), name="ds")
        else:
            self.dt = cp.Constant(np.zeros((N, n_t)))
            self.ds = cp.Constant(np.zeros((N, 1)))

        # Pack physical components back into augmented variables
        # dz = [dx, dt, dbeta]
        dz_components = [self.dx, self.dt]
        if n_ctcs > 0:
            dz_components.append(self.dbeta)
        self.dz = cp.hstack(dz_components)

        # dnu = [du, ds]
        self.dnu = cp.hstack([self.du, self.ds])

        # Virtual buffers (None if zero-sized)
        self.vb_ineq    = cp.Variable((N, self.n.nonconvex_inequality), name="vb_ineq")  \
              if self.n.nonconvex_inequality  > 0 else None 
        
        # ---------------------------------------------
        # TERMINAL CONDITION BUFFERS (REAL + CTCS)
        # --- ------------------------------------------
        n_term_real = self.n.final_state + self.n.term_ineq
        n_term_ctcs  = self.n.term_ctcs
        # ------------ State terminal constraints (size = n_term_state) ------------
        self.vb_term_state = (
            cp.Variable(n_term_real, name="vb_term_state")
            if (method.flags.buff_dyn == "term"
                and method.flags.dynamics_nonconvex != 0
                and n_term_real > 0)
            else (cp.Constant(np.zeros(n_term_real)) if n_term_real > 0 else None)
        )
        # ------------ CTCS terminal constraints (size = n_term_ctcs) ------------
        self.vb_term_ctcs = (
            cp.Variable(n_term_ctcs, name="vb_term_ctcs")
            if (n_term_ctcs > 0)  
            else None
        )
        # ------------ Unified stacked terminal buffer ------------
        self.vb_term = (
            cp.hstack([self.vb_term_state, self.vb_term_ctcs])
            if (self.vb_term_state is not None and self.vb_term_ctcs is not None)
            else self.vb_term_state
            if self.vb_term_ctcs is None
            else self.vb_term_ctcs
            if self.vb_term_state is None
            else None # when both are None
        )

        # ---------------------------------------------
        # DYNAMICS VIRTUAL BUFFERS (REAL + CTCS)
        # ---------------------------------------------
        # --- State dynamics ---
        self.vb_dyn_state_p = (
            cp.Variable((N - 1, self.n.state), name="vb_dyn_state_plus")
            if method.flags.buff_dyn != "term" and method.flags.dynamics_nonconvex != 0 and self.n.state > 0
            else cp.Constant(np.zeros((N - 1, self.n.state))) if self.n.state > 0
            else None
        )
        self.vb_dyn_state_m = (
            cp.Variable((N - 1, self.n.state), name="vb_dyn_state_minus")
            if method.flags.buff_dyn != "term" and method.flags.dynamics_nonconvex != 0 and self.n.state > 0
            else cp.Constant(np.zeros((N - 1, self.n.state))) if self.n.state > 0
            else None
        )
        # --- Time dynamics (always zero - no time buffering) ---
        self.vb_dyn_time_p = cp.Constant(np.zeros((N - 1, self.n.time)))
        self.vb_dyn_time_m = cp.Constant(np.zeros((N - 1, self.n.time)))
        
        # --- CTCS dynamics (self.indices.z.ctcs) ---
        self.vb_dyn_ctcs_p = (
            cp.Variable((N - 1, n_ctcs), name="vb_dyn_ctcs_plus")
            if method.flags.ctcs not in {"none", "term"} and n_ctcs > 0
            else cp.Constant(np.zeros((N - 1, n_ctcs))) if n_ctcs > 0
            else None
        )
        self.vb_dyn_ctcs_m = (
            cp.Variable((N - 1, n_ctcs), name="vb_dyn_ctcs_minus")
            if method.flags.ctcs not in {"none", "term"} and n_ctcs > 0
            else cp.Constant(np.zeros((N - 1, n_ctcs))) if n_ctcs > 0
            else None
        )
        # --- Unified composite buffers (always same shape for DPP) ---
        self.vb_dyn_p = (
            cp.hstack([self.vb_dyn_state_p, self.vb_dyn_time_p, self.vb_dyn_ctcs_p])
            if n_ctcs>0 else cp.hstack([self.vb_dyn_state_p, self.vb_dyn_time_p])
        )
        self.vb_dyn_m = (
            cp.hstack([self.vb_dyn_state_m, self.vb_dyn_time_m, self.vb_dyn_ctcs_m])
            if n_ctcs>0 else cp.hstack([self.vb_dyn_state_m, self.vb_dyn_time_m])
        )

        # Aggregate buffers (optional)
        self.vb_plus_real   = cp.Variable((self.N.pm_real, self.n.plus_real),  name="vb_plus_real")  if self.n.plus_real  > 0 else None
        self.vb_minus_real  = cp.Variable((self.N.pm_real, self.n.minus_real), name="vb_minus_real") if self.n.minus_real > 0 else None
        self.vb_plus_ctcs   = cp.Variable((self.N.pm_ctcs, self.n.plus_ctcs),  name="vb_plus_ctcs")  if self.n.plus_ctcs  > 0 else None
        self.vb_minus_ctcs  = cp.Variable((self.N.pm_ctcs, self.n.minus_ctcs), name="vb_minus_ctcs") if self.n.minus_ctcs > 0 else None

        if problem.costs.has(type="nonconvex", minimax=1):
            self.minimax_epigraph_upperbound = cp.Variable((1,), name="minimax_epigraph_upperbound")

    # ============================================================
    # CONSTRAINTS (build-once)
    # ============================================================
    def _build_constraints_once(self) -> None:
        problem     = self.problem
        method      = self.method

        N, n_x, n_u, n_z, n_nu, n_ctcs \
            = self.N.time_grid, self.n.state, self.n.control, self.n.z, self.n.nu, self.n.ctcs

        C: List[cp.Constraint] = []

        if bool(self.flags.free_final_time):
            # Initial time is always fixed: dt[0] = 0 (whether free_final_time is True or False)
            C.append(self.dt[0, 0] == 0)

        # Terminal equalities / inequalities
        term_idx  = self.indices.constraints.final_state  

        for constraint in problem.constraints.get(ct=0, type="equality_bc"):
            cnst_idx        = constraint.idx
            boundary_idx    = constraint.boundary_idx
            
            if constraint.name in self.constraint_params:
                value = self.constraint_params[constraint.name]["value"]
            else:
                value = constraint.value

            if constraint.set == "state":
                if constraint.boundary == "final":
                    vb = self.vb_term[term_idx.eq] if self.vb_term is not None else 0.0
                else:
                    vb = 0
                C.append(self.dz[boundary_idx,cnst_idx] + self.z_ref[boundary_idx, cnst_idx] - vb == value)
            elif constraint.set == "control":
                C.append(self.dnu[boundary_idx,cnst_idx] + self.nu_ref[boundary_idx, cnst_idx] == value)

        for constraint in problem.constraints.get(ct=0, type="inequality_bc"):

            min_value_idx   = constraint.min_value_idx
            max_value_idx   = constraint.max_value_idx
            min_value       = constraint.min_value
            max_value       = constraint.max_value
            cnst_idx        = constraint.idx
            boundary_idx    = constraint.boundary_idx
            M_select        = constraint.M_select

            if constraint.set == "state":
                if constraint.boundary == "final":
                    vb = self.vb_term[term_idx.ineq] if self.vb_term is not None else 0.0
                else:
                    vb = 0
                C.append(M_select @ (self.dz[boundary_idx, cnst_idx] + self.z_ref[boundary_idx, cnst_idx]) - vb <= cp.hstack([-min_value, max_value]))
            
            elif constraint.set == "control":
                C.append(M_select @ (self.dnu[boundary_idx, cnst_idx] + self.nu_ref[boundary_idx, cnst_idx]) <= cp.hstack([-min_value, max_value]))
        
        # CTCS terminal equalities
        if problem.index_map.n.term_ctcs>0:
            vbN_ctcs = self.vb_term[term_idx.ctcs] if self.vb_term is not None else 0.0
            C.append(
                self.dz[-1, self.indices.z.ctcs] + self.z_ref[-1, self.indices.z.ctcs] - vbN_ctcs == 0.0
            )
        
        if self.flags.buff_dyn == "quad-1":
            C.append(cp.sum(self.vb_dyn_p) == self.vb_plus_real)
            C.append(cp.sum(self.vb_dyn_m) == self.vb_minus_real)

        if self.flags.ctcs == "quad-1":
            C.append(cp.sum(self.vb_dyn_p) == self.vb_plus_ctcs)
            C.append(cp.sum(self.vb_dyn_m) == self.vb_minus_ctcs)

        if self.flags.buff_dyn == "quad-2":
            C.append(cp.sum(self.vb_dyn_state_p, axis=1) == self.vb_plus_real[:, 0])
            C.append(cp.sum(self.vb_dyn_state_m, axis=1) == self.vb_minus_real[:, 0])

        if self.flags.ctcs == "quad-2":
            C.append(cp.sum(self.vb_dyn_p[:, self.indices.z.ctcs], axis=1) == self.vb_plus_ctcs[:, 0])
            C.append(cp.sum(self.vb_dyn_m[:, self.indices.z.ctcs], axis=1) == self.vb_minus_ctcs[:, 0])

        if self.flags.buff_dyn == "quad-3":
            C.append(cp.sum(self.vb_dyn_p[:, self.indices.z.real], axis=0) == self.vb_plus_real[0, :])
            C.append(cp.sum(self.vb_dyn_m[:, self.indices.z.real], axis=0) == self.vb_minus_real[0, :])
        
        if self.flags.ctcs == "quad-3":
            C.append(cp.sum(self.vb_dyn_p[:, self.indices.z.ctcs], axis=0) == self.vb_plus_ctcs[0, :])
            C.append(cp.sum(self.vb_dyn_m[:, self.indices.z.ctcs], axis=0) == self.vb_minus_ctcs[0, :])


        # pseudospectral collocation
        if self.flags.discretize == "ps":
            C.append(discretize.build_ps_dyn_constraints(self))

        # Per-stage constraints
        for k in range(N):
            if k < N - 1:
                
                # multiple shooting discretization
                if self.flags.discretize == "ms":
                    C.append(discretize.build_ms_dyn_constraint(self, k))

                # #backwards shooting dynamics constraints:
                # rhs = (
                #     self.Ak_bwd[k] @ self.dz[k+1]
                #     + self.Bk_bwd[k] @ self.dnu[k]
                #     + self.Bkp_bwd[k] @ self.dnu[k + 1]
                #     # + (self.vb_dyn_p[k] - self.vb_dyn_m[k])
                # )
                # C.append(self.dz[k] + self.z_ref[k] - self.z_m_bwd[k] == rhs)

                if self.flags.buff_dyn != "term":
                    C.append(self.vb_dyn_p[k, self.indices.z.real] >= 0)
                    C.append(self.vb_dyn_m[k, self.indices.z.real] >= 0)

                if self.flags.ctcs != "term" and n_ctcs > 0:
                    C.append(self.vb_dyn_p[k, self.indices.z.ctcs] >= 0)
                    C.append(self.vb_dyn_m[k, self.indices.z.ctcs] >= 0)
                
                # CTCS coupling on extra components
                if method.flags.ctcs != "none" and n_ctcs>0:
                    C.append((self.z_ref[k + 1, self.indices.z.ctcs] + self.dz[k + 1, self.indices.z.ctcs]) - (self.z_ref[k, self.indices.z.ctcs] + self.dz[k, self.indices.z.ctcs]) <= self.eps_ctcs)

                # TODO(Skye): Verify below timestep constraints
                # Free-final-time bounds
                if bool(self.flags.free_final_time):
                    t_k         = self.t_ref[k, 0] + self.dt[k, 0]
                    t_kp        = self.t_ref[k + 1, 0] + self.dt[k + 1, 0]
                    t_interval_k= t_kp - t_k
                    C.append(t_interval_k <= self.dt_max)
                    C.append(t_interval_k >= self.dt_min)
                    C.append(cp.abs(self.dt[k, 0]) <= self.ddt_max)
                    C.append(t_k >= 0)
                    C.append(0.0 <= self.s_ref[k, 0] + self.ds[k, 0])

                # Control slew (udot)
                for constraint in problem.constraints.get(ct=0, type="control_rate_limit"):
                    value = constraint.value
                    M_sel = constraint.M_select
                    du_k = (self.nu_ref[k + 1, self.indices.nu.control] + self.dnu[k + 1, self.indices.nu.control] - (self.nu_ref[k, self.indices.nu.control] + self.dnu[k, self.indices.nu.control]))
                    dt_k = (self.t_ref[k+1, 0] + self.dt[k+1, 0]) - (self.t_ref[k, 0] + self.dt[k, 0])
                    C.append(
                        M_sel @ du_k
                        <= dt_k * np.concatenate([value, value])
                    )

            # Time dilation constraints
            if bool(self.flags.free_final_time):
                C.append(self.T_min <= self.t_ref[-1, 0] + self.dt[-1, 0])
                C.append(self.t_ref[-1, 0] + self.dt[-1, 0] <= self.T_max)
                # Equal time step perturbations: force all dt[k] = dt[1] for k >= 1
                # (dt[0] is always 0, so we want dt[1] = dt[2] = ... = dt[N-1])
                if hasattr(self.flags, 'equal_dt') and bool(self.flags.equal_dt) and k > 1:
                    C.append(self.dt[k, 0] == self.dt[1, 0])

            # State box constraints
            for constraint in problem.constraints.get(ct=0, type="box"):
                min_value = constraint.min_value
                max_value = constraint.max_value

                M_select = constraint.M_select

                if constraint.set == "state":

                    C.append(M_select @ (self.z_ref[k, self.indices.z.state] + self.dz[k, self.indices.z.state]) <= np.concatenate([-min_value, max_value]))
                elif constraint.set == "control":
                    C.append(M_select @ (self.nu_ref[k, self.indices.nu.control] + self.dnu[k, self.indices.nu.control]) <= np.concatenate([-min_value, max_value]))

            # Linearized inequality constraints (path + nfz + custom)
            # if problem.constraints.has("nodal", "nonconvex_inequality","POLYTOPE_OUT","SOC_OUT"):  ### DAN: UPDATE BELOW LINE TO
            if problem.constraints.has(ct=0, type="nonconvex_inequality"):
                C.append(self.dgdx[k] @ self.dz[k, self.indices.z.state] + self.dgdu[k] @ self.dnu[k, self.indices.nu.control] + self.g0[k] - self.vb_ineq[k] <= 0)
                if str(self.flags.flag_autotune) in {"1", "3", "al-scvx"} and self.vb_ineq[k]:
                    C.append(self.vb_ineq[k] >= 0)

                # TODO: TEMPORARY: force slack to zero for hard constraints
                idx = 0
                for c in problem.constraints.get(ct=0, type="nonconvex_inequality"):
                    if c.hard:
                        C.append(self.dgdx[k,idx:idx+c.dimension] @ self.dz[k, self.indices.z.state] + self.dgdu[k,idx:idx+c.dimension] @ self.dnu[k, self.indices.nu.control] + self.g0[k,idx:idx+c.dimension] <= 0)
                    idx += c.dimension

            for cost in problem.costs.get(type="nonconvex", minimax=1):
                C.append(self.w_cost_times_dcostdx[k] @ self.dz[k, self.indices.z.state] + self.w_cost_times_dcostdu[k] @ self.dnu[k, self.indices.nu.control] + self.w_cost_times_cost0[k] <= self.minimax_epigraph_upperbound)

            # convex constraints
            for constraint in problem.constraints.get(ct=0, type="axis_angle_cone"):

                cnst_idx = constraint.idx
                axis = constraint.axis

                if constraint.set == "state":
                    C.append(constraint.cos_theta_max * cp.norm(self.z_ref[k,cnst_idx] + self.dz[k,cnst_idx]) <= axis @ (self.z_ref[k,cnst_idx] + self.dz[k,cnst_idx]))
                elif constraint.set == "control":
                    C.append(constraint.cos_theta_max * cp.norm(self.nu_ref[k,cnst_idx] + self.dnu[k,cnst_idx]) <= axis @ (self.nu_ref[k,cnst_idx] + self.dnu[k,cnst_idx]))

            for constraint in problem.constraints.get(ct=0, type="max_norm_cone"):

                cnst_idx = constraint.idx
                max_value = constraint.max_value
                
                if constraint.set == "state":
                    C.append(cp.norm(self.z_ref[k,cnst_idx] + self.dz[k,cnst_idx]) <= max_value)
                elif constraint.set == "control":
                    C.append(cp.norm(self.nu_ref[k,cnst_idx] + self.dnu[k,cnst_idx]) <= max_value)

            #############################################################################################
            #### ----------------------------------------------------------------------------------- ####
            #### ----------------------------------- FROM DAN -------------------------------------- ####
            #### -------------------------- GENERALIZED CONSTRAINTS -------------------------------- ####


            #### ----------------- DECIDE TO USE A CONSTRAINT BASED ON TIME STEP ------------------- ####
            def use_constraint_at_time_query(time_steps):
                use_constraint = False
                if time_steps == 'all': use_constraint = True ## use if applied to all time steps
                elif k in time_steps: use_constraint = True ## use if positive index is in time_steps
                elif k-N in time_steps: use_constraint = True ## use if negative index is in time_steps
                return use_constraint

            ############# ----------------------- CONVEX CONSTRAINTS ------------------- ################
            # for constraint in problem.constraints.get('POLYTOPE_IN'):
            for constraint in problem.constraints.get(ct=0, type='AFFINE'):                
                if constraint.convex == True:
                    if use_constraint_at_time_query(constraint.time_steps):
                        cnst_idx = constraint.idx
                        AA = constraint.A; bb = constraint.b
                        if constraint.set == 'state': C.append(AA @ (self.z_ref[k, cnst_idx] + self.dz[k, cnst_idx]) == bb)
                        elif constraint.set == "control": C.append(AA @ (self.nu_ref[k, cnst_idx] + self.dnu[k, cnst_idx]) == bb)

            for constraint in problem.constraints.get(ct=0, type='POLYTOPE'):
                if constraint.convex == True:
                    if use_constraint_at_time_query(constraint.time_steps):
                        cnst_idx = constraint.idx
                        AA = constraint.A; bb = constraint.b
                        if constraint.set == 'state': C.append(AA @ (self.z_ref[k, cnst_idx] + self.dz[k, cnst_idx]) <= bb)
                        elif constraint.set == "control": C.append(AA @ (self.nu_ref[k, cnst_idx] + self.dnu[k, cnst_idx]) <= bb)
            for constraint in problem.constraints.get(ct=0, type='SOC'):
                if constraint.convex == True:
                    if use_constraint_at_time_query(constraint.time_steps):
                        cnst_idx = constraint.idx
                        AA = constraint.A; bb = constraint.b
                        CC = constraint.C; dd = constraint.d
                        if constraint.set == 'state': C.append(cp.norm(AA@(self.z_ref[k, cnst_idx] + self.dz[k, cnst_idx]) + bb) <= CC @ (self.z_ref[k, cnst_idx] + self.dz[k, cnst_idx]) + dd)
                        if constraint.set == 'control': C.append(cp.norm(AA@(self.nu_ref[k, cnst_idx] + self.dnu[k, cnst_idx]) + bb) <= CC @ (self.nu_ref[k, cnst_idx] + self.dnu[k, cnst_idx]) + dd)

            ############# -------------------- NONCONVEX CONSTRAINTS ------------------- ################
            # for constraint in problem.constraints.get('POLYTOPE_OUT','SOC_OUT'):
            #     if use_constraint_query(constraint.time_steps):
            #         if constraint.set == 'state': pass 
            #         if constraint.set == 'control': pass 
            
            #### ----------------------------------------------------------------------------------- ####
            #### ----------------------------------------------------------------------------------- ####
            ############################################################################################# 

            # TODO(carlos): think about where to put this, this is a special constraint
            # but general enough to apply to different 6-dof models using quaternions
            for constraint in problem.constraints.get(ct=0, type="quaternion_cone"):
                quat_start_idx = constraint.quat_start_idx
                C.append(cp.norm(self.z_ref[k, quat_start_idx + 2: quat_start_idx + 4] + self.dz[k, quat_start_idx + 2: quat_start_idx + 4]) <= constraint.rhs)
        
        self.cp_constraints += C

    # ============================================================
    # COST FUNCTION (DCP-safe)
    # ============================================================
    def _build_cost_once(self) -> cp.Expression:
        problem, method = self.problem, self.method
        """Full baseline cost: TRUE + TR + 0.5*VIRTUAL + DUAL; gated via flags & autotune."""

        self.TRUE = 0.0

        for cost in problem.costs.get(ct=0, type="nonconvex", minimax=0):
            self.TRUE = (cp.sum(self.w_cost_times_cost0)
                    + cp.sum(cp.multiply(self.w_cost_times_dcostdx, self.dz[:,self.indices.state])
                    + cp.sum(cp.multiply(self.w_cost_times_dcostdu, self.dnu)))
                    )
            
        for cost in problem.costs.get(type="nonconvex", minimax=1):
            self.TRUE += self.minimax_epigraph_upperbound

        if bool(self.flags.free_final_time):
            for cost in problem.costs.get(type="min_time"):
                time_cost = cp.sum(self.t_ref[-1] + self.dt[-1])
                self.TRUE += time_cost

        for cost in problem.costs.get(type="min_norm_terminal"):
            zf = self.z_ref[-1] + self.dz[-1]
            idx = cost.idx
            print(f"{zf.shape}, {idx}")
            term_cost = self.w.cost * cp.norm(zf[idx])
            self.TRUE += term_cost

        for cost in problem.costs.get(type="terminal_state"):
            zf = self.z_ref[-1] + self.dz[-1]

            idx = cost.idx
            self.TRUE += zf[idx]

        for cost in problem.costs.get(type="rate_regularization"):
            if cost.set == "control":
                nu = self.nu_ref + self.dnu
                nu_minus    = nu[:-1,  cost.idx]
                nu_plus     = nu[1:,   cost.idx]

                if cost.norm_type == "l2":
                    self.TRUE += cost.w * cp.sum_squares(nu_plus - nu_minus) * (1 / self.N.time_grid)

                if cost.norm_type == "l1":
                    self.TRUE += cost.w * cp.norm1(nu_plus - nu_minus) * (1 / self.N.time_grid)
            
            elif cost.set == "state":
                z = self.z_ref + self.dz
                z_minus = z[:-1, cost.idx]
                z_plus = z[1:, cost.idx]
                
                if cost.norm_type == "l2":
                    self.TRUE += cost.w * cp.sum_squares(z_plus - z_minus) * (1 / self.N.time_grid)

                if cost.norm_type == "l1":
                    self.TRUE += cost.w * cp.norm1(z_plus - z_minus) * (1 / self.N.time_grid)

        # === Trust-region penalties ===
        TR = self.flags.flag_tr * (self.w.tr_z * cp.sum_squares(self.dz) + self.w.tr_u * cp.sum_squares(self.dnu))
        # === Virtual buffer penalties ===
        VB = 0.0
        DUAL = 0.0

        # Quadratic penalties 
        if self.flags.flag_autotune in {"0", "2", "3", "al-scvx"}:
            VB = hp.build_virtual_buffer_cost(self)

        # Dual penalties
        if self.flags.flag_autotune in {"1", "3", "al-scvx"}:
            DUAL = hp.build_dual_buffer_cost(self)

        return self.TRUE + TR + VB + DUAL
    

    # ============================================================
    # SOLVE SCP ITERATION AND ASSEMBLE ITERATION RECORD (UNIFIED HISTORY)
    # ============================================================
    def solve_iteration(self) -> None:
        """
        Single SCP iteration (no return):
          - builds inputs for the iteration from self.iter_data[-1],
          - loads Parameters and solves,
          - assembles a unified record (inputs used + outputs),
          - appends it to self.iter_data.
        """
        # Build inputs for this iteration (refs/weights/conv_data, iter_num)
        input_for_iter  = self._load_inputs()

        # Parameter propagation and linearization
        prop_time_ms    = self._load_parameters(input_for_iter)

        # Solve subproblem
        solver_name = self.method.solver_opts.get("solver", "CLARABEL")
        ignore_dpp  = self.method.flags.ignore_dpp
        self.cp_subproblem.solve( 
                solver=solver_name, warm_start=False, ignore_dpp=ignore_dpp
            )  # ignore_dpp=True if desired
        print(self.cp_subproblem.status)

        # Create unified record for this iteration and append
        iter_record         = self._load_outputs(input_for_iter, prop_time_ms)
        iter_record         = convergence.check_convergence_tolerance(self.problem, self.method, iter_record)
        iter_record_prev    = self.iter_data[-1]
        iter_record         = self._baseline_autotune(self.problem, self.method, iter_record, iter_record_prev)
        self.iter_data.append(iter_record)

    # ============================================================
    # PARAMETER UPDATES AND SOLVE (UNIFIED HISTORY)
    # ============================================================
    def _set_param(self, param: Optional[cp.Parameter], val: np.ndarray) -> None:
        if param is None:
            return
        arr = np.asarray(val)
        if param is self.z_m and arr.shape != (self.N.time_grid, self.n.z):
            arr = arr.reshape(self.N.time_grid, self.n.z)
        param.value = arr

    def _load_inputs(self) -> Dict[str, Any]:
        last_rec = self.iter_data[-1]
        k_prev = int(last_rec.get("iter_num", 0))

        refs = tools.AttrDict({
                "z_ref": last_rec["z_opt"],
                "nu_ref": last_rec["nu_opt"],
            })
        
        W = copy.deepcopy(last_rec.get("W"))
        dual = copy.deepcopy(last_rec.get("dual"))
        penalty = copy.deepcopy(last_rec.get("penalty", self.method.penalty))
        conv_data = last_rec.get("conv_data", {})

        next_inputs = tools.AttrDict({
            "iter_num": k_prev + 1,
            **refs,
            "W": W,
            "dual": dual,
            "penalty": penalty,
            "conv_data": conv_data,
        })

        return tools.AttrDict(next_inputs)

    def _load_parameters(self, inputs: Dict[str, Any]) -> float:

        problem, method = self.problem, self.method


        # assign cost-related CP parameters
        self.z_ref.value  = inputs.z_ref
        self.nu_ref.value = inputs.nu_ref

        # Unpack numpy reference trajectories (mirrors _create_parameters for numpy arrays)
        self.x_ref, self.t_ref, self.beta_ref, self.u_ref, self.s_ref = self.index_map.unpack_znu(
            inputs.z_ref, inputs.nu_ref
        )

        start = time.time()
        Ak, Bk, Bkp, z_minus = discretize.compute_linsys_discrete(
            inputs.z_ref, inputs.nu_ref, problem, method
        )

        # inputs.z_ref = z_minus

        # Ak_bwd, Bk_bwd, Bkp_bwd, Sk_bwd, z_minus_bwd = discretize.compute_linsys_discrete_bwd(
        #     inputs.z_ref, inputs.nu_ref, inputs.dt_ref, problem, method
        # )

        prop_time_ms = (time.time() - start) * 1000.0

        start = time.time()

        # compute linearized terminal and running costs
        cost, dcostdx, dcostdu = discretize.compute_nonconvex_costs(inputs.z_ref, inputs.nu_ref, problem, method)

        if problem.constraints.has(ct=0, type="nonconvex_inequality"):
            g, dgdx, dgdu = discretize.compute_nonconvex_constraints(inputs.z_ref, inputs.nu_ref, problem, method)
        else:
            g = dgdx = dgdu = None

        if dgdx is not None:
            self.dgdx.value     = dgdx
            self.dgdu.value     = dgdu
            self.g0.value       = g

        ncvx_cnstr_time = (time.time() - start) * 1000.0
        print(f"ncvx_cnstr_time: {ncvx_cnstr_time}")

        # Dynamics & references
        self._set_param(self.Ak,   Ak)
        self._set_param(self.Bk,   Bk)
        self._set_param(self.Bkp,  Bkp)
        self._set_param(self.z_m, z_minus)

        # # backwards shooting dynamics
        # self._set_param(self.Ak_bwd,   Ak_bwd)
        # self._set_param(self.Bk_bwd,   Bk_bwd)
        # self._set_param(self.Bkp_bwd,  Bkp_bwd)
        # self._set_param(self.Sk_bwd,   Sk_bwd)
        # self._set_param(self.z_m_bwd, z_minus_bwd)

        if inputs.get("W") is None or inputs.get("dual") is None:
            W_stack, dual_stack = configure_penalty_weights(self.problem, self.method, subconstraints=self.constraints)
        else:
            # Use the already-stacked values from previous iteration (already updated by autotune)
            W_stack     = inputs.W
            dual_stack  = inputs.dual
        
        # Update subproblem state
        self.W_stack    = W_stack
        self.dual_stack = dual_stack
        inputs.W     = W_stack
        inputs.dual  = dual_stack

        # update scalar weights in-subproblem
        penalty_to_use = inputs.get("penalty", tools.AttrDict(self.method.penalty))
        # self.w.cost.value   = penalty_to_use.get("w_cost", 1.0)
        self.w.tr_z.value   = penalty_to_use.get("wtr_z", 1e-2)
        self.w.tr_u.value   = penalty_to_use.get("wtr_u", 1e-2)
        inputs.penalty = copy.deepcopy(penalty_to_use)
        
        self.w_cost_times_dcostdx.value     = self.w.cost.value * dcostdx
        self.w_cost_times_dcostdu.value    = self.w.cost.value  * dcostdu
        self.w_cost_times_cost0.value       = self.w.cost.value * cost

        for constraint in problem.constraints.get(ct=0, type="equality_bc"):
            if constraint.name in self.constraint_params:
                self.constraint_params[constraint.name]["value"].value = constraint.value

        self.nu_ref_sq.value = np.sum(inputs.nu_ref * inputs.nu_ref, axis=1)
        
        if bool(self.flags.free_final_time):
            self.dt_min.value  = float(method.dt_min)
            self.dt_max.value  = float(method.dt_max)
            self.ddt_max.value = float(method.ddt_max)
            self.T_min.value   = float(method.Ts_min)
            self.T_max.value   = float(method.Ts_max)

        # 1. Refresh W_sqrt CVXPY parameter values from stacked constraint weights
        self.W_sqrt.nonconvex_inequality.value = tools.ensure_shape(np.sqrt(W_stack.nonconvex_inequality), self.W_sqrt.nonconvex_inequality.shape)
        self.W_sqrt.final_state.value          = tools.ensure_shape(np.sqrt(W_stack.final_state),             self.W_sqrt.final_state.shape)
        self.W_sqrt.dynamics.value             = tools.ensure_shape(np.sqrt(W_stack.dynamics),             self.W_sqrt.dynamics.shape)
        self.W_sqrt.plus_real.value            = tools.ensure_shape(np.sqrt(W_stack.plus_real),            self.W_sqrt.plus_real.shape)
        self.W_sqrt.minus_real.value           = tools.ensure_shape(np.sqrt(W_stack.minus_real),           self.W_sqrt.minus_real.shape)
        self.W_sqrt.plus_ctcs.value            = tools.ensure_shape(np.sqrt(W_stack.plus_ctcs),            self.W_sqrt.plus_ctcs.shape)
        self.W_sqrt.minus_ctcs.value           = tools.ensure_shape(np.sqrt(W_stack.minus_ctcs),           self.W_sqrt.minus_ctcs.shape)

        # 2. Update dual CVXPY parameter values from stacked constraint duals
        self.dual.nonconvex_inequality.value   = tools.ensure_shape(dual_stack.nonconvex_inequality, self.dual.nonconvex_inequality.shape)
        self.dual.final_state.value            = tools.ensure_shape(dual_stack.final_state,               self.dual.final_state.shape)
        self.dual.dynamics.value               = tools.ensure_shape(dual_stack.dynamics,               self.dual.dynamics.shape)
        self.dual.plus_real.value              = tools.ensure_shape(dual_stack.plus_real,              self.dual.plus_real.shape)
        self.dual.minus_real.value             = tools.ensure_shape(dual_stack.minus_real,             self.dual.minus_real.shape)
        self.dual.plus_ctcs.value              = tools.ensure_shape(dual_stack.plus_ctcs,              self.dual.plus_ctcs.shape)
        self.dual.minus_ctcs.value             = tools.ensure_shape(dual_stack.minus_ctcs,             self.dual.minus_ctcs.shape)


        # ctcs eps
        self.eps_ctcs.value = float(method.conv.eps_ctcs)

        # cache for optional debug
        inputs._linsys_cache = (Ak, Bk, Bkp, z_minus)
        # inputs._linsys_cache_bwd = (Ak_bwd, Bk_bwd, Bkp_bwd, z_minus_bwd)
        return prop_time_ms

    # ============================================================
    # OUTPUT PACKING (UNIFIED RECORD)
    # ============================================================
    def _load_outputs(self, input_for_iter: Dict[str, Any], prop_time_ms: float) -> Dict[str, Any]:
        # mission, model, method = self.problem.mission, self.problem.model, self.method
        N, n_x, n_u, n_z, n_nu, n_ctcs = self.N.time_grid, self.n.state, self.n.control, self.n.z, self.n.nu, self.n.ctcs

        dz_val, dnu_val = self.dz.value, self.dnu.value
        dt_val = None

        rec = tools.AttrDict(copy.deepcopy(input_for_iter))
        rec.cp_subprob = self.cp_subproblem

        if self.cp_subproblem is not None:
            compilation_time = getattr(self.cp_subproblem, "compilation_time", None)
            rec.parse_time = float(compilation_time or 0.0) * 1000.0
        else:
            rec.parse_time = None

        if self.cp_subproblem.solver_stats is not None:
            solve_time = getattr(self.cp_subproblem.solver_stats, "solve_time", None)
            rec.solve_time = float(solve_time or 0.0) * 1000.0
        else:
            rec.solve_time = None

        # raw solver variables (useful for diagnostics)
        rec.dz_s    = dz_val
        rec.dnu_s   = dnu_val

        # outputs (absolute trajectories)
        rec.z_opt  = tools.safe_val(dz_val, rows=N, cols=self.n.z) + input_for_iter.z_ref
        rec.nu_opt = tools.safe_val(dnu_val, rows=N, cols=self.n.nu) + input_for_iter.nu_ref

        # Unpack physical components from optimal trajectories
        rec.x_opt, rec.t_opt, rec.beta_opt, rec.u_opt, rec.s_opt \
            = self.index_map.unpack_znu(rec.z_opt, rec.nu_opt)

        rec.dt_opt = np.diff(tools.reshape_1d(rec.t_opt))
        rec.T_opt  = float(tools.reshape_1d(rec.t_opt)[-1]) if tools.reshape_1d(rec.t_opt).size > 0 else 0.0

        # Discretization model (expose for debug/analysis)
        Ak, Bk, Bkp, z_minus = input_for_iter.get("_linsys_cache", (None, None, None, None))
        rec.z_minus             = self.z_m.value if z_minus is None else z_minus
        rec.Ak                  = self.Ak.value if Ak is None else Ak
        rec.Bk                  = self.Bk.value if Bk is None else Bk
        rec.Bkp                 = self.Bkp.value if Bkp is None else Bkp

        # Ak_bwd, Bk_bwd, Bkp_bwd, z_minus_bwd = input_for_iter.get("_linsys_cache_bwd", (None, None, None, None))
        # rec.z_minus_bwd = self.z_m_bwd.value if z_minus_bwd is None else z_minus_bwd
        # rec.Ak_bwd = self.Ak_bwd.value if Ak_bwd is None else Ak_bwd
        # rec.Bk_bwd = self.Bk_bwd.value if Bk_bwd is None else Bk_bwd
        # rec.Bkp_bwd = self.Bkp_bwd.value if Bkp_bwd is None else Bkp_bwd

        # Path residuals and reference cost
        g, _, _     = discretize.compute_nonconvex_constraints(rec.z_opt, rec.nu_opt, self.problem, self.method)
        cost, _, _  = discretize.compute_nonconvex_costs(rec.z_opt, rec.nu_opt, self.problem, self.method)

        rec.cnst_path = g

        # Handle case where self.TRUE is either a float or cvxpy expression
        if hasattr(self.TRUE, 'value'):
            true_cost_value = self.TRUE.value.item()
        else:
            true_cost_value = float(self.TRUE)
        rec.cost = true_cost_value / self.w.cost.value

        # Convergence data (buffers, defects, TR cost, ref cost)
        conv = tools.AttrDict()
        conv.vb_ineq         = tools.get_val(self.vb_ineq,  rows=self.N.time_grid, cols=self.n.nonconvex_inequality) if self.vb_ineq  is not None else np.zeros((self.N.time_grid,self.n.nonconvex_inequality))
        conv.vb_terminal     = tools.get_val(self.vb_term,  rows=1, cols=self.n.term_total) if self.vb_term  is not None else np.zeros((1, self.n.term_total))
        conv.vb_dyn          = tools.get_val(self.vb_dyn_p, rows=self.N.time_grid-1,  cols=self.n.dynamics) - tools.get_val(self.vb_dyn_m, rows=self.N.time_grid-1, cols=self.n.dynamics)
        conv.vb_plus_real    = tools.get_val(self.vb_plus_real, rows=self.N.pm_real, cols=self.n.plus_real) if self.vb_plus_real  is not None else np.zeros((self.N.pm_real, self.n.plus_real))
        conv.vb_minus_real   = tools.get_val(self.vb_minus_real, rows=self.N.pm_real, cols=self.n.minus_real) if self.vb_minus_real  is not None else np.zeros((self.N.pm_real, self.n.minus_real))
        conv.vb_plus_ctcs    = tools.get_val(self.vb_plus_ctcs, rows=self.N.pm_ctcs, cols=self.n.plus_ctcs) if self.vb_plus_ctcs  is not None else np.zeros((self.N.pm_ctcs, self.n.plus_ctcs))
        conv.vb_minus_ctcs   = tools.get_val(self.vb_minus_ctcs, rows=self.N.pm_ctcs, cols=self.n.minus_ctcs) if self.vb_minus_ctcs  is not None else np.zeros((self.N.pm_ctcs, self.n.minus_ctcs))

        conv.defect  = tools.safe_val(self.dz, rows=N, cols=n_x) + input_for_iter.z_ref - self.z_m.value
        conv.Jtr     = ( self.w.tr_z.value * np.sum(tools.safe_val(self.dz, rows=N, cols=n_x)**2)
                        + self.w.tr_u.value * np.sum(tools.safe_val(self.dnu, rows=N, cols=n_nu)**2) )
        ref_cost = discretize.compute_nonconvex_costs(input_for_iter.z_ref, input_for_iter.nu_ref, self.problem, self.method)[0].sum().item()
        conv.cost_ref = ref_cost

        rec.conv_data  = conv
        rec.prop_time  = prop_time_ms

        return rec
    
    # ===========================
    # Baseline autotune wrapper
    # ===========================
    def _baseline_autotune(self, problem, method, rec: Dict[str, Any], rec_prev) -> Dict[str, Any]:
        """Autotune updates W_stack and dual_stack on the subproblem directly."""
        flag = method.flags.flag_autotune
        conv_data = rec.conv_data
        conv_data_prev = rec_prev.get("conv_data")
        iter_num = rec.iter_num
        
        if flag == "1":
            dual_update_info = hp.autotune1(self, conv_data, iter_num)
            rec.dual_update = dual_update_info
        elif flag == "2":
            weight_update_info = hp.autotune2(self, conv_data, conv_data_prev, iter_num)
            rec.weight_update = weight_update_info
        elif flag == "3":
            update_info = hp.autotune3(self, conv_data, conv_data_prev, iter_num)
            rec.autotune_update = update_info

        # Copy updated W_stack and dual_stack to iter_record for history
        rec.W = tools.AttrDict({
            "nonconvex_inequality": self.W_stack.nonconvex_inequality.copy(),
            "final_state": self.W_stack.final_state.copy(),
            "dynamics": self.W_stack.dynamics.copy(),
            "plus_real": self.W_stack.plus_real.copy(),
            "minus_real": self.W_stack.minus_real.copy(),
            "plus_ctcs": self.W_stack.plus_ctcs.copy(),
            "minus_ctcs": self.W_stack.minus_ctcs.copy(),
        })
        rec.dual = tools.AttrDict({
            "nonconvex_inequality": self.dual_stack.nonconvex_inequality.copy(),
            "final_state": self.dual_stack.final_state.copy(),
            "dynamics": self.dual_stack.dynamics.copy(),
            "plus_real": self.dual_stack.plus_real.copy(),
            "minus_real": self.dual_stack.minus_real.copy(),
            "plus_ctcs": self.dual_stack.plus_ctcs.copy(),
            "minus_ctcs": self.dual_stack.minus_ctcs.copy(),
        })

        return rec