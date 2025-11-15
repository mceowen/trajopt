"""
Subproblem module: SCP with DPP updates 
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import importlib
import time
import numpy as np
import cvxpy as cp
import jax.numpy as jnp

# trajopt imports
from trajopt.core.modules.method import discretize as discretize
from trajopt.core.modules.method import convexify as convexify
from trajopt.core.modules.method import hyperparameters as hp
from trajopt.core.modules.method import convergence
from trajopt.utils import tools


# =====================================================================================
# Public API: full Sequential Convex Programming (SCP) loop with DPP reuse
# =====================================================================================
def run_scp(problem):
    """
    Run the full Sequential Convex Programming (SCP) loop for a given problem.

    Subproblem maintains iteration history internally:
      - subprob.iter_data[0]: initialization inputs only
      - subprob.iter_data[k>=1]: inputs used at iter k + outputs from that iter
    """

    # Start full problem convergence timer
    time_start = time.perf_counter()

    # Get or create the compiled Subproblem (DPP)
    subprob: Optional[Subproblem] = getattr(problem.method, "subprob", None)
    if subprob is None:
        subprob = Subproblem(problem)
        problem.method.subprob = subprob

    # START SUBPROBLEM CONSTRUCTION / HEADER
    print("-" * 152)
    print(f"                                              ..:: {problem.mission.name}: PTR with Virtual Buffer ::..")
    print("-" * 152)
    print("  Iteration |  Propagation |   Solve   |    Parse   |  log(dz)  |      log(VB)    |   log(VB)   |  log(VB)    | Solve status |  Time of    |   Cost    ")
    print("            |   time [ms]  | time [ms] |  time [ms] |           |  (path + NFZ)   |  (terminal) |  (dynamics) |              |  Flight [s] |           ")
    print("-" * 152)

    max_iter = int(problem.method.conv["iter_max"])

    for _ in range(max_iter + 1):
        subprob.solve_iteration()  # appends a new unified record for this iteration

        latest = subprob.iter_data[-1]
        display_subprob_status(problem, latest)

        if latest.get("converged", False):
            print("Terminated from convergence criteria!")
            break

    if not subprob.iter_data[-1].get("converged", False):
        print("Terminated from hitting maximum iterations!")

    problem.solution = problem.method.subprob.iter_data[-1]

    # Save off convergence time (full time - parse time)
    problem.solution['t_full'] = time.perf_counter() - time_start

    return problem


# ===================================
# Module flags (gating via Parameters)
# ===================================
@dataclass
class ModuleFlags:
    path:  float = 1.0
    nfz:   float = 1.0
    custom:   float = 1.0
    term:  float = 1.0
    dyn:   float = 1.0
    tr:    float = 1.0
    true:  float = 1.0
    dual:  float = 1.0
    vb:    float = 1.0


# =========================
# Subproblem (build-once)
# =========================
class Subproblem:
    """Reusable convex SCP with full baseline functionality & DPP updates.
    """

    def __init__(self, problem) -> None:
        self.problem = problem
        mission = self.problem.mission
        model   = self.problem.model
        method  = self.problem.method

        # derive canonical sizes for quick reuse
        self.N  = int(method.N)
        self.n  = int(model.n)
        self.m  = int(model.m)
        self.nz = int(model.nz)

        self.flags         = method.flags
        self.free_T        = bool(self.flags["free_final_time"])
        self.equal_dt      = bool(self.flags["equal_dt"])
        self.buff_dyn      = self.flags["buff_dyn"]                  # e.g., "l1", "l2", "term", "quad-1", "quad-2"
        self.flag_autotune = self.flags["flag_autotune"]

        # module sizes
        self.n_path  = int(mission.n_path)
        self.n_nfz   = int(mission.n_nfz)
        self.n_custom   = int(getattr(mission, "n_custom", 0))
        self.n_ineq  = int(mission.n_ineq)
        self.n_term  = int(mission.n_term + mission.n_term_ineq)
        self.n_dyn   = int(getattr(mission, "n_dyn", model.nz))
        self.Npm     = int(getattr(method, "Npm", 0))
        self.n_plus  = int(getattr(method, "n_plus", 0))
        self.n_minus = int(getattr(method, "n_minus", 0))

        # Optional module flags as Parameters (enable gating)
        self.flag_path = method.flags.get("flag_path", 1.0)
        self.flag_nfz  = method.flags.get("flag_nfz", 1.0)
        self.flag_custom  = method.flags.get("flag_custom", 1.0)
        self.flag_term = method.flags.get("flag_term", 1.0)
        self.flag_dyn  = method.flags.get("flag_dyn", 1.0)
        self.flag_tr   = method.flags.get("flag_tr", 1.0)
        self.flag_true = method.flags.get("flag_true", 1.0)
        self.flag_dual = method.flags.get("flag_dual", 1.0)
        self.flag_vb   = method.flags.get("flag_vb", 1.0)

        # Bounds (only create when nonzero-length)
        self._init_bounds(mission)

        # Build the DPP graph once
        self._create_variables()
        self._create_parameters()
        self.constraints: List[cp.constraints.constraint.Constraint] = []
        self._build_constraint_once()
        self.cost_expr = self._build_cost_once()

        # apply custom constraints and cost
        mission.custom_constraints(self)
        mission.custom_cost(self)

        # Compile CVXPY problem once
        self.subproblem = cp.Problem(cp.Minimize(self.cost_expr), self.constraints)

        # --------------------------
        # Initialize unified history
        # --------------------------
        self.iter_data: List[Dict[str, Any]] = [{
            "iter_num": 0,  # init only (no outputs yet)
            "z_ref": problem.method.z_init,
            "us_ref": problem.method.nu_init,
            "dt_ref": problem.method.dt_init,
            "t_ref": problem.method.t_init,
            "conv_data": {
                "vb_ineq": np.zeros((self.N, mission.n_ineq)),
                "vb_dyn":  np.zeros((self.N - 1, self.nz)),
                "vb_term": np.zeros((self.n_term, 1)),
            },
            "weights": problem.method.weights,
        }]

    # -------------------------
    # Helper: init bounds
    # -------------------------
    def _init_bounds(self, mission):
        self.z1     = mission.zi if mission.n_init > 0 else None
        if mission.n_init_ineq > 0:
            self.z1_min = mission.zi_min
            self.z1_max = mission.zi_max
        else:
            self.z1_min = self.z1_max = None

        self.zN     = mission.zf if mission.n_term > 0 else None
        if mission.n_term_ineq > 0:
            self.zN_min = mission.zf_min
            self.zN_max = mission.zf_max
        else:
            self.zN_min = self.zN_max = None

        self.u1 = mission.ui if mission.n_init_ctrl > 0 else None
        self.uN = mission.uf if mission.n_term_ctrl > 0 else None

        self.z_min = mission.z_min if hasattr(mission, "z_min") and len(mission.z_min) > 0 else None
        self.z_max = mission.z_max if hasattr(mission, "z_max") and len(mission.z_max) > 0 else None

        if mission.n_ctrl > 0:
            self.u_min  = mission.u_min
            self.u_max  = mission.u_max
        else:
            self.u_min = self.u_max = None

        if mission.n_udot > 0:
            self.udot_max = mission.udot_max
        else:
            self.udot_max = None

    # ============================================================
    # VARIABLE & PARAMETER CREATION
    # ============================================================
    def _create_variables(self) -> None:
        mission, model, method = self.problem.mission, self.problem.model, self.problem.method
        N, n, m, nz = self.N, self.n, self.m, self.nz

        # Core optimization variables
        self.dz = cp.Variable((N, nz), name="dz")
        self.dnu= cp.Variable((N, m), name="du")

        # Time variable(s)
        if self.free_T:
            if self.equal_dt:
                self.dT = cp.Variable(name="dT")
                self.dt = (1 / (N - 1)) * self.dT * np.ones((N - 1, 1))
            else:
                self.dT = None
                self.dt = cp.Variable((N - 1, 1), name="dt")
        else:
            self.dT = None
            self.dt = cp.Constant(np.zeros((N - 1, 1)))  # CVXPY constant, safe in constraints


        # Virtual buffers (None if zero-sized)
        self.vb_ineq    = cp.Variable((N, mission.n_ineq), name="vb_ineq")   if mission.n_ineq  > 0 else None 
        self.vb_term    = (
                            cp.Variable(self.n_term, name="vb_term")
                            if method.flags.get("buff_dyn") == "term" and self.n_term > 0
                            else cp.Constant(np.zeros(self.n_term)) if self.n_term > 0
                            else None
                        )


        # --- Physical dynamics (first n states) ---
        self.vb_dyn_real_p = (
            cp.Variable((N - 1, n), name="vb_dyn_real_plus")
            if method.flags["buff_dyn"] != "term" and n > 0
            else cp.Constant(np.zeros((N - 1, n))) if n > 0
            else None
        )
        self.vb_dyn_real_m = (
            cp.Variable((N - 1, n), name="vb_dyn_real_minus")
            if method.flags["buff_dyn"] != "term" and n > 0
            else cp.Constant(np.zeros((N - 1, n))) if n > 0
            else None
        )

        
        # --- CTCS dynamics (augmented states n : nz) ---
        self.vb_dyn_ctcs_p = (
            cp.Variable((N - 1, max(nz - n, 0)), name="vb_dyn_ctcs_plus")
            if method.flags["ctcs"] != "none" not in {"none", "term"} and nz > n
            else cp.Constant(np.zeros((N - 1, max(nz - n, 0)))) if nz > n
            else None
        )
        self.vb_dyn_ctcs_m = (
            cp.Variable((N - 1, max(nz - n, 0)), name="vb_dyn_ctcs_minus")
            if method.flags["ctcs"] != "none" not in {"none", "term"} and nz > n
            else cp.Constant(np.zeros((N - 1, max(nz - n, 0)))) if nz > n
            else None
        )

        # --- Unified composite buffers (always same shape for DPP) ---
        self.vb_dyn_p = (
            cp.hstack([self.vb_dyn_real_p, self.vb_dyn_ctcs_p])
            if nz > n else self.vb_dyn_real_p
        )
        self.vb_dyn_m = (
            cp.hstack([self.vb_dyn_real_m, self.vb_dyn_ctcs_m])
            if nz > n else self.vb_dyn_real_m
        )

        # Aggregate buffers (optional)
        self.vb_plus  = cp.Variable((self.Npm, self.n_plus),  name="vb_plus")  if self.n_plus  > 0 else None
        self.vb_minus = cp.Variable((self.Npm, self.n_minus), name="vb_minus") if self.n_minus > 0 else None

    def _create_parameters(self) -> None:
        mission, model, method = self.problem.mission, self.problem.model, self.problem.method
        N, n, m, nz = self.N, self.n, self.m, self.nz

        # Linearized dynamics & trajectory references
        self.Ak     = cp.Parameter((N - 1, nz, nz), name="Ak")
        self.Bk     = cp.Parameter((N - 1, nz, m), name="Bk")
        self.Bkp    = cp.Parameter((N - 1, nz, m), name="Bkp")
        self.Sk     = cp.Parameter((N - 1, nz),    name="Sk")
        self.z_m   = cp.Parameter((N, nz),        name="z_minus")

        self.w_cost_times_dcostdz = cp.Parameter((N, n), name="dcostdz")
        self.w_cost_times_dcostdnu = cp.Parameter((N, m), name="dcostdnu")
        self.w_cost_times_cost0   = cp.Parameter(N,      name="cost0")

        self.z_ref  = cp.Parameter((N, nz),    name="z_ref")
        self.nu_ref  = cp.Parameter((N, m),    name="us_ref")
        self.dt_ref = cp.Parameter((N - 1, 1),name="dt_ref", nonneg=True)

        # Path/NFZ/AUX linearized constraints
        if mission.n_ineq > 0:
            self.dgdz = cp.Parameter((N, mission.n_ineq, n), name="dgdz")
            self.dgdnu = cp.Parameter((N, mission.n_ineq, m), name="dgdnu")
            self.g0   = cp.Parameter((N, mission.n_ineq),    name="g0")
        else:
            self.dgdz = self.dgdnu = self.g0 = None

        # time-step scalar bounds
        if self.free_T:
            self.dt_min  = cp.Parameter(nonneg=True, name="dt_min")
            self.dt_max  = cp.Parameter(nonneg=True, name="dt_max")
            self.ddt_max = cp.Parameter(nonneg=True, name="ddt_max")
        else:
            self.dt_min = self.dt_max = self.ddt_max = None

        # Weights & Trust Region weights
        self.w_cost = cp.Parameter(nonneg=True, name="w_cost")
        self.wtr_z  = cp.Parameter(nonneg=True, name="wtr_z")
        self.wtr_u  = cp.Parameter(nonneg=True, name="wtr_u")

        # TODO(Skye): refactor these weight parameters to be less redundant later
        self.W_ineq  = cp.Parameter((N,  max(self.n_ineq, 1)),  nonneg=True, name="W_ineq")
        self.W_term  = cp.Parameter((max(self.n_term, 1),),     nonneg=True, name="W_term")
        self.W_dyn   = cp.Parameter((N - 1, max(nz, 1)),        nonneg=True, name="W_dyn")
        self.W_plus  = cp.Parameter((max(self.Npm, 1), max(self.n_plus, 1)),  nonneg=True, name="W_plus")
        self.W_minus = cp.Parameter((max(self.Npm, 1), max(self.n_minus, 1)), nonneg=True, name="W_minus")
        # ----------------------------------------------------------------------------------------------------------
        self.W_ineq_sqrt  = cp.Parameter((N,  max(self.n_ineq, 1)),  nonneg=True, name="W_ineq_sqrt")
        self.W_term_sqrt  = cp.Parameter((max(self.n_term, 1),),     nonneg=True, name="W_term_sqrt")
        self.W_dyn_sqrt   = cp.Parameter((N - 1, max(nz, 1)),        nonneg=True, name="W_dyn_sqrt")
        self.W_plus_sqrt  = cp.Parameter((max(self.Npm, 1), max(self.n_plus, 1)),  nonneg=True, name="W_plus_sqrt")
        self.W_minus_sqrt = cp.Parameter((max(self.Npm, 1), max(self.n_minus, 1)), nonneg=True, name="W_minus_sqrt")
        # ----------------------------------------------------------------------------------------------------------
        self.w_ineq_row = cp.Parameter(N,  nonneg=True, name="w_ineq_row")  if self.n_ineq > 0 else None
        self.w_dyn_row  = cp.Parameter(N - 1, nonneg=True, name="w_dyn_row")
        # ----------------------------------------------------------------------------------------------------------
        self.W_ineq.value  = np.ones((N,  max(self.n_ineq, 1)))
        self.W_term.value  = np.ones((max(self.n_term, 1),))
        self.W_dyn.value   = np.ones((N - 1, max(nz, 1)))
        self.W_plus.value  = np.ones((max(self.Npm, 1), max(self.n_plus, 1)))
        self.W_minus.value = np.ones((max(self.Npm, 1), max(self.n_minus, 1)))
        # ----------------------------------------------------------------------------------------------------------
        self.W_ineq_sqrt.value  = np.sqrt(self.W_ineq.value)
        self.W_term_sqrt.value  = np.sqrt(self.W_term.value)
        self.W_dyn_sqrt.value   = np.sqrt(self.W_dyn.value)
        self.W_plus_sqrt.value  = np.sqrt(self.W_plus.value)
        self.W_minus_sqrt.value = np.sqrt(self.W_minus.value)
        # ----------------------------------------------------------------------------------------------------------
        if self.w_ineq_row is not None:
            self.w_ineq_row.value = np.max(self.W_ineq.value, axis=1)
        self.w_dyn_row.value = np.max(self.W_dyn.value, axis=1)

        # duals (same ≥1-column pattern, unified inequality structure)
        self.dual_ineq  = cp.Parameter((N,  max(self.n_ineq, 1)), name="dual_ineq")
        self.dual_dyn   = cp.Parameter((N - 1, nz),              name="dual_dyn")
        self.dual_plus  = cp.Parameter((max(self.Npm, 1), max(self.n_plus, 1)),  name="dual_plus")
        self.dual_minus = cp.Parameter((max(self.Npm, 1), max(self.n_minus, 1)), name="dual_minus")
        self.dual_term  = cp.Parameter((max(self.n_term, 1),),                  name="dual_term")

        # CTCS epsilon (scalar)
        self.eps_ctcs = cp.Parameter(nonneg=True, name="eps_ctcs")

    # ============================================================
    # CONSTRAINTS (build-once)
    # ============================================================
    def _build_constraint_once(self) -> None:
        mission, model, method = self.problem.mission, self.problem.model, self.problem.method
        N, n, m, nz = self.N, self.n, self.m, self.nz
        flags = method.flags

        C: List[cp.Constraint] = []

        # Initial control (optional)
        if mission.flags.get("init_ctrl", False) and mission.n_init_ctrl > 0:
            C.append(self.dnu[0,mission.ui_idx] + self.nu_ref[0, mission.ui_idx] == self.u1)

        # Terminal control (optional)
        if mission.flags.get("final_ctrl", False) and mission.n_term_ctrl > 0:
            C.append(self.dnu[-1,mission.uf_idx] + self.nu_ref[-1, mission.uf_idx] == self.uN)

        # Initial equalities / inequalities
        if mission.n_init > 0 and self.z1 is not None:
            C.append(self.dz[0, mission.zi_idx] + self.z_ref[0, mission.zi_idx] == self.z1)

        if mission.n_init_ineq > 0 and self.z1_min is not None and self.z1_max is not None:
            M_sel = tools.constraint_index_selector(mission.zi_min_idx, mission.zi_max_idx, n)
            C.append(M_sel @ (self.dz[0, :n] + self.z_ref[0, :n]) <= cp.hstack([-self.z1_min, self.z1_max]))

        # Terminal equalities / inequalities
        if mission.n_term > 0 and self.zN is not None:
            vbN = self.vb_term[:mission.n_term] if self.vb_term is not None else 0.0
            C.append(self.dz[-1, mission.zf_idx] + self.z_ref[-1, mission.zf_idx] - vbN == self.zN)

        if mission.n_term_ineq > 0 and self.zN_min is not None and self.zN_max is not None:
            nterm = mission.n_term
            vbNiq = self.vb_term[nterm:nterm + mission.n_term_ineq] if self.vb_term is not None else 0.0
            M_sel = tools.constraint_index_selector(mission.zf_min_idx, mission.zf_max_idx, n)
            C.append(M_sel @ (self.dz[-1, :n] + self.z_ref[-1, :n]) - vbNiq <= cp.hstack([-self.zN_min, self.zN_max]))

        # Per-stage constraints
        for k in range(N):
            if k < N - 1:
                # Discrete dynamics with dyn buffers
                rhs = (
                    self.Ak[k] @ self.dz[k]
                    + self.Bk[k] @ self.dnu[k]
                    + self.Bkp[k] @ self.dnu[k + 1]
                    + cp.multiply(self.Sk[k], self.dt[k])
                    + (self.vb_dyn_p[k] - self.vb_dyn_m[k])
                )
                C.append(self.dz[k + 1] + self.z_ref[k + 1] - self.z_m[k + 1] == rhs)

                if self.buff_dyn != "term":
                    C.append(self.vb_dyn_p[k] >= 0)
                    C.append(self.vb_dyn_m[k] >= 0)

                # CTCS coupling on extra components
                if method.flags["ctcs"] != "none" and n < nz:
                    C.append(
                        self.z_ref[k + 1, n:nz] + self.dz[k + 1, n:nz]
                        - (self.z_ref[k, n:nz] + self.dz[k, n:nz]) <= self.eps_ctcs
                    )

                # Free-final-time bounds
                if self.free_T:
                    C.append(self.dt_ref[k] + self.dt[k] <= self.dt_max)
                    C.append(self.dt_ref[k] + self.dt[k] >= self.dt_min)
                    C.append(cp.abs(self.dt[k]) <= self.ddt_max)

                # Control slew (udot)
                if mission.n_udot > 0 and self.udot_max is not None and k < N - 2:
                    M_sel = tools.constraint_index_selector(mission.udot_max_idx, mission.udot_max_idx, m)
                    C.append(
                        M_sel @ (self.nu_ref[k + 1] + self.dnu[k + 1] - (self.nu_ref[k] + self.dnu[k]))
                        <= (self.dt_ref[k] + self.dt[k]) * cp.hstack([self.udot_max, self.udot_max])
                    )

            # State box constraints
            if mission.n_state > 0 and self.z_min is not None and self.z_max is not None:
                M_sel = tools.constraint_index_selector(mission.z_min_idx, mission.z_max_idx, n)
                C.append(
                    M_sel @ (self.z_ref[k, :n] + self.dz[k, :n])
                    <= cp.hstack([-self.z_min, self.z_max])
                )

            # Control box constraints
            if mission.n_ctrl > 0 and self.u_min is not None and self.u_max is not None:
                M_sel = tools.constraint_index_selector(mission.u_min_idx, mission.u_max_idx, m)
                C.append(
                    M_sel @ (self.nu_ref[k] + self.dnu[k])
                    <= cp.hstack([-self.u_min, self.u_max])
                )

            # Linearized inequality constraints (path + nfz + custom)
            if mission.n_ineq > 0 and method.flags["ctcs"] == "none" and self.dgdz is not None:
        
                C.append(self.dgdz[k] @ self.dz[k] + self.dgdnu[k] @ self.dnu[k] + self.g0[k] - self.vb_ineq[k] <= 0)
                if str(self.flag_autotune) in {"1", "3", "al-scvx"} and self.vb_ineq[k]:
                    C.append(self.vb_ineq[k] >= 0)

        # Fixed-time tying
        if not self.free_T:
            C.append(self.dt == 0)

        # Equal dt tying
        if self.free_T and self.equal_dt and self.dT is not None:
            one = np.ones((self.N - 1, 1)) / (self.N - 1)
            C.append(self.dt == one * self.dT)

        self.constraints += C

    # ============================================================
    # COST FUNCTION (DCP-safe)
    # ============================================================
    def _build_cost_once(self) -> cp.Expression:
        mission, model, method = self.problem.mission, self.problem.model, self.problem.method
        """Full baseline cost: TRUE + TR + 0.5*VIRTUAL + DUAL; gated via flags & autotune."""

        # === TRUE cost (linearized objective) ===
        TRUE = self.flag_true * (
            cp.sum(cp.multiply(self.w_cost_times_dcostdz, self.dz[:,:model.n]))
          + cp.sum(cp.multiply(self.w_cost_times_dcostdnu, self.dnu))
          + cp.sum(self.w_cost_times_cost0)
        )

        # === Trust-region penalties ===
        TR = self.flag_tr * (self.wtr_z * cp.sum_squares(self.dz[:, :model.n]) + self.wtr_u * cp.sum_squares(self.dnu))

        # === Virtual buffer penalties ===
        VB = 0.0
        DUAL = 0.0

        # Quadratic penalties 
        if self.flag_autotune in {"0", "2", "3", "al-scvx"}:
            VB = hp.build_virtual_buffer_cost(self)

        # Dual penalties
        if self.flag_autotune in {"1", "3", "al-scvx"}:
            DUAL = hp.build_dual_buffer_cost(self)

        return TRUE + TR + VB + DUAL

    # ============================================================
    # PARAMETER UPDATES AND SOLVE (UNIFIED HISTORY)
    # ============================================================
    def _set_param(self, param: Optional[cp.Parameter], val: np.ndarray) -> None:
        if param is None:
            return
        arr = np.asarray(val)
        if param is self.z_m and arr.shape != (self.N, self.nz):
            arr = arr.reshape(self.N, self.nz)
        param.value = arr

    def _load_inputs(self) -> Dict[str, Any]:
        last_rec = self.iter_data[-1]
        k_prev = int(last_rec.get("iter_num", 0))

        if all(key in last_rec for key in ("zs", "us", "dt", "ts")):
            refs = {
                "z_ref": last_rec["zs"],
                "us_ref": last_rec["us"],
                "dt_ref": last_rec["dt"],
                "t_ref": last_rec["ts"],
            }
            weights = last_rec.get("weights", self.problem.method.weights)
            conv_data = last_rec.get("conv_data", {})
        else:
            refs = {
                "z_ref": last_rec["z_ref"],
                "us_ref": last_rec["us_ref"],
                "dt_ref": last_rec["dt_ref"],
                "t_ref": last_rec["t_ref"],
            }
            weights = last_rec.get("weights", self.problem.method.weights)
            conv_data = last_rec.get("conv_data", {})

        next_inputs = {
            "iter_num": k_prev + 1,
            **refs,
            "weights": weights,
            "conv_data": conv_data,
        }
        return next_inputs

    def _load_parameters(self, inputs: Dict[str, Any]) -> float:
        mission, model, method = self.problem.mission, self.problem.model, self.problem.method

        start = time.time()
        Ak, Bk, Bkp, Sk, z_minus = discretize.compute_linsys_discrete(
            inputs["z_ref"], inputs["us_ref"], inputs["dt_ref"], self.problem
        )
        prop_time_ms = (time.time() - start) * 1000.0

        # compute linearized terminal and running costs
        cost, dcostdz, dcostdnu = discretize.compute_linearized_costs(inputs["t_ref"], inputs["z_ref"], inputs["us_ref"], self.problem)

        g, dgdz, dgdnu = discretize.compute_nodal_inequality_constraints(inputs["t_ref"], inputs["z_ref"], inputs["us_ref"], self.problem)

        # Dynamics & references
        self._set_param(self.Ak,   Ak)
        self._set_param(self.Bk,   Bk)
        self._set_param(self.Bkp,  Bkp)
        self._set_param(self.Sk,   Sk)
        self._set_param(self.z_m, z_minus)

        # Weights/duals (ensure shapes, fill any scalars/empties)
        W = inputs["weights"]
        self.w_cost = W.get("w_cost", 1.0)
        self.wtr_z.value  = W.get("wtr_z", 1e-2)
        self.wtr_u.value  = W.get("wtr_u", 1e-2)

        self.w_cost_times_dcostdz.value = self.w_cost * dcostdz[:, 0, :]
        self.w_cost_times_dcostdnu.value = self.w_cost * dcostdnu[:, 0, :]
        self.w_cost_times_cost0.value   = self.w_cost * cost[:, 0, 0]
        self.z_ref.value  = inputs["z_ref"]
        self.nu_ref.value  = inputs["us_ref"]
        self.dt_ref.value = inputs["dt_ref"].reshape(self.N - 1, 1)

        if dgdz is not None:
            self.dgdz.value = dgdz
            self.dgdnu.value = dgdnu
            self.g0.value   = g
        
        if self.free_T:
            self.dt_min.value  = float(method.dt_min)
            self.dt_max.value  = float(method.dt_max)
            self.ddt_max.value = float(method.ddt_max)

        # TODO(Skye): refactor weight loading to reduce code duplication with autotune        
        W_ineq_arr = tools.ensure_shape(W.get("W_ineq", 0.0), (self.N, max(self.n_ineq, 1)))
        W_term_arr = tools.ensure_shape(W.get("W_term", 0.0), (max(self.n_term, 1),))
        W_dyn_arr  = tools.ensure_shape(W.get("W_dyn",  0.0), (self.N - 1, max(self.nz, 1)))
        W_plus_arr = tools.ensure_shape(W.get("W_plus", 0.0), (max(self.Npm, 1), max(self.n_plus,  1)))
        W_minus_arr= tools.ensure_shape(W.get("W_minus",0.0), (max(self.Npm, 1), max(self.n_minus, 1)))
        # ------------------------------------------------------------------
        # Assign to CVXPY parameters
        self.W_ineq.value  = W_ineq_arr
        self.W_term.value  = W_term_arr
        self.W_dyn.value   = W_dyn_arr
        self.W_plus.value  = W_plus_arr
        self.W_minus.value = W_minus_arr
        # ------------------------------------------------------------------
        # Square-rooted parameters (for quadratic penalties)
        self.W_ineq_sqrt.value  = np.sqrt(W_ineq_arr)
        self.W_term_sqrt.value  = np.sqrt(W_term_arr)
        self.W_dyn_sqrt.value   = np.sqrt(W_dyn_arr)
        self.W_plus_sqrt.value  = np.sqrt(W_plus_arr)
        self.W_minus_sqrt.value = np.sqrt(W_minus_arr)
        # ------------------------------------------------------------------
        # Rowwise scalar reductions (for DCP-safe convex weighting)
        if self.w_ineq_row is not None:
            self.w_ineq_row.value = np.max(W_ineq_arr, axis=1)
        if self.w_dyn_row is not None:
            self.w_dyn_row.value  = np.max(W_dyn_arr, axis=1)
        # ------------------------------------------------------------------
        # Dual variables (unified inequality structure)
        self.dual_ineq.value  = tools.ensure_shape(W.get("dual_ineq", 0.0), (self.N, max(self.n_ineq, 1)))
        self.dual_dyn.value   = tools.ensure_shape(W.get("dual_dyn",  0.0), (self.N - 1, self.nz))
        self.dual_plus.value  = tools.ensure_shape(W.get("dual_plus", 0.0), (max(self.Npm, 1), max(self.n_plus,  1)))
        self.dual_minus.value = tools.ensure_shape(W.get("dual_minus",0.0), (max(self.Npm, 1), max(self.n_minus, 1)))
        self.dual_term.value  = tools.ensure_shape(W.get("dual_term", 0.0), (max(self.n_term,1),))


        # ctcs eps
        self.eps_ctcs.value = float(self.problem.method.conv["eps_ctcs"])

        # cache for optional debug
        inputs["_linsys_cache"] = (Ak, Bk, Bkp, Sk, z_minus)
        return prop_time_ms

    def solve_iteration(self) -> None:
        """
        Single SCP iteration (no return):
          - builds inputs for the iteration from self.iter_data[-1],
          - loads Parameters and solves,
          - assembles a unified record (inputs used + outputs),
          - appends it to self.iter_data.
        """
        # Build inputs for this iteration (refs/weights/conv_data, iter_num)
        input_for_iter = self._load_inputs()

        # Parameter propagation and linearization
        prop_time_ms = self._load_parameters(input_for_iter)

        # Solve subproblem
        solver_name = self.problem.method.solver_opts.get("solver", "ECOS")
        self.subproblem.solve(solver=solver_name, warm_start=True)  # ignore_dpp=True if desired

        # Create unified record for this iteration and append
        iter_record = self._load_outputs(input_for_iter, prop_time_ms)
        iter_record = convergence.check_convergence_tolerance(self.problem, self, iter_record)
        iter_record = baseline_autotune(self.problem, iter_record)
        self.iter_data.append(iter_record)

    # ============================================================
    # OUTPUT PACKING (UNIFIED RECORD)
    # ============================================================
    def _load_outputs(self, input_for_iter: Dict[str, Any], prop_time_ms: float) -> Dict[str, Any]:
        mission, model, method = self.problem.mission, self.problem.model, self.problem.method
        N, n, m = self.N, self.n, self.m

        dz_val, dnu_val = self.dz.value, self.dnu.value
        dt_val = self.dt.value if isinstance(self.dt, cp.expressions.expression.Expression) else self.dt

        rec: Dict[str, Any] = dict(input_for_iter)  # include exact inputs used this iteration
        rec["subprob"] = self.subproblem

        if self.subproblem is not None:
            compilation_time = getattr(self.subproblem, "compilation_time", None)
            rec["parse_time"] = float(compilation_time or 0.0) * 1000.0
        else:
            rec["parse_time"] = None
        

        if self.subproblem.solver_stats is not None:
            solve_time = getattr(self.subproblem.solver_stats, "solve_time", None)
            rec["solve_time"] = float(solve_time or 0.0) * 1000.0
        else:
            rec["solve_time"] = None

        # raw solver variables (useful for diagnostics)
        rec["dz_s"] = dz_val
        rec["dnu_s"] = dnu_val
        rec["dt_s"] = dt_val

        # outputs (absolute trajectories)
        rec["zs"]  = tools.safe_val(dz_val, rows=N, cols=n) + input_for_iter["z_ref"]
        rec["us"]  = tools.safe_val(dnu_val, rows=N, cols=m) + input_for_iter["us_ref"]
        rec["dt"] = tools.safe_val(dt_val).squeeze() + input_for_iter["dt_ref"].squeeze()
        rec["ts"]  = np.concatenate(([0], np.cumsum(rec["dt"])))
        rec["Ts"]  = float(np.sum(rec["dt"]))

        # Discretization model (expose for debug/analysis)
        Ak, Bk, Bkp, Sk, z_minus = input_for_iter.get("_linsys_cache", (None, None, None, None, None))
        rec["z_minus"] = self.z_m.value if z_minus is None else z_minus
        rec["Ak"] = self.Ak.value if Ak is None else Ak
        rec["Bk"] = self.Bk.value if Bk is None else Bk
        rec["Bkp"] = self.Bkp.value if Bkp is None else Bkp
        rec["Sk"]  = self.Sk.value if Sk is None else Sk

        # Path residuals and reference cost
        g, _, _ = discretize.compute_nodal_inequality_constraints(rec["ts"], rec["zs"], rec["us"], self.problem)

        rec["cnst_path"] = g
        rec["cost"]      = discretize.compute_linearized_costs(rec["ts"], rec["zs"], rec["us"], self.problem)[0].sum().item()
 
        # Convergence data (buffers, defects, TR cost, ref cost)
        conv = {}
        conv["soln"]    = self.subproblem
        conv["vb_ineq"] = tools.get_val(self.vb_ineq,  rows=self.n_ineq, cols=self.N) if self.vb_ineq  is not None else np.zeros((self.n_ineq,  self.N))
        conv["vb_term"] = tools.get_val(self.vb_term,  rows=self.n_term, cols=1)      if self.vb_term  is not None else np.zeros((self.n_term,  1))
        conv["vb_dyn"]  = tools.get_val(self.vb_dyn_p, rows=self.n_dyn,  cols=self.N-1) - tools.get_val(self.vb_dyn_m, rows=self.n_dyn, cols=self.N-1)

        conv["defect"]  = tools.safe_val(self.dz, rows=N, cols=n) + input_for_iter["z_ref"] - self.z_m.value
        conv["Jtr"]     = ( float(self.wtr_z.value) * np.sum(tools.safe_val(self.dz, rows=N, cols=n)**2)
                          + float(self.wtr_u.value) * np.sum(tools.safe_val(self.dnu, rows=N, cols=m)**2) )
        ref_cost = discretize.compute_linearized_costs(input_for_iter["t_ref"], input_for_iter["z_ref"], input_for_iter["us_ref"], self.problem)[0].sum().item()
        conv["cost_ref"] = ref_cost

        rec["conv_data"]  = conv
        rec["prop_time"]  = prop_time_ms

        return rec


# ===========================
# Baseline autotune wrapper
# ===========================
def baseline_autotune(problem, rec: Dict[str, Any]) -> Dict[str, Any]:
    flag = problem.method.flags["flag_autotune"]
    if flag == "1":
        rec = hp.autotune1(problem, rec)
    elif flag == "2":
        rec = hp.autotune2(problem, rec)
    elif flag == "3":
        rec = hp.autotune3(problem, rec)
    return rec


# ==========================================
# Iteration status printout (unified record)
# ==========================================
def display_subprob_status(problem, rec: Dict[str, Any]) -> None:
    conv = rec.get("conv_data", {})

    chk_feas_path = conv.get("chk_feas_path", 0.0)
    chk_feas_nfz  = conv.get("chk_feas_nfz", 0.0)
    ineq_vb       = chk_feas_path + (chk_feas_nfz if chk_feas_nfz != 0 else 0.0)

    chk_dz        = conv.get("chk_dz", 1e-12)
    chk_feas_term = conv.get("chk_feas_term", 1e-12)
    chk_feas_dyn  = conv.get("chk_feas_dyn", 1e-12)

    log_dz      = np.log10(max(chk_dz, 1e-12))
    log_vb_ineq = np.log10(max(ineq_vb, 1e-12))
    log_vb_term = np.log10(max(chk_feas_term, 1e-12))
    log_vb_dyn  = np.log10(max(chk_feas_dyn, 1e-12))

    solve_stat  = conv.get("status", "UNKNOWN")
    iter_num    = int(rec.get("iter_num", -1))

    nt    = float(problem.method.nondim.get("nt", 1.0))
    ncost = float(problem.method.nondim.get("ncost", 1.0))

    Ts   = float(rec.get("Ts", 0.0))
    cost = float(rec.get("cost", 0.0))

    prop_ms = float(rec.get("prop_time", 0.0) or 0.0)
    solve_ms= float(rec.get("solve_time", 0.0) or 0.0)
    parse_ms= float(rec.get("parse_time", 0.0) or 0.0)

    print(
        "     {:02d}     |    {:07.1f}   |   {:06.1f}  |   {:06.1f}   |   {:+04.1f}    |      {:+05.1f}      |    {:+05.1f}    |     {:+05.1f}   |    {:s}    |   {:4.2f}   |  {:4.1f}".format(
            iter_num,
            prop_ms,
            solve_ms,
            parse_ms,
            log_dz,
            log_vb_ineq,
            log_vb_term,
            log_vb_dyn,
            str(solve_stat),
            Ts * nt,
            cost * ncost
        )
    )
