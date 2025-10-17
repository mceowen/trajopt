"""
subproblem.py
--------------
High-fidelity, faster SCP subproblem: builds the CVXPY graph ONCE, then updates
Parameters each iteration (DPP) and warm-starts the solver.
Default solve is ECOS if not provided otherwise.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import importlib

import numpy as np
import cvxpy as cp

# trajopt imports
from trajopt.algorithm import discretization as discretize
from trajopt.algorithm import convexification as convexify
from trajopt.algorithm import hyperparameters as hp
from trajopt.algorithm import convergence
from trajopt.utils import tools


# =====================================================================================
# Public API (Option 2): single entry that caches/reuses a compiled Subproblem instance
# =====================================================================================

def solve_subproblem(problem: Dict[str, Any]) -> Dict[str, Any]:
    # Locate/carry algorithm cache slot
    alg = problem.get("algorithm", None)
    subprob_obj = None

    # Retrieve cached subproblem if available
    if alg is not None:
        if isinstance(alg, dict):
            subprob_obj = alg.get("subprob", None)
        else:
            subprob_obj = getattr(alg, "subprob", None)

    # Build once if needed
    if subprob_obj is None:
        subprob_obj = Subproblem(problem)
        if alg is not None:
            if isinstance(alg, dict):
                alg["subprob"] = subprob_obj
            else:
                setattr(alg, "subprob", subprob_obj)

    # Current iterate & weights
    I = problem["I"][-1]
    weights = I["weights"]

    # Optional per-iteration module flags
    flags = ModuleFlags(
        path=1.0, nfz=1.0, aux=1.0, term=1.0, dyn=1.0,
        tr=1.0, true=1.0,
        dual=1.0 if str(problem["params"]["bools"]["flag_autotune"]) in {"1", "3", "al-scvx"} else 0.0,
        vb=1.0,
    )

    # Update, solve, collect
    subprob_obj.update_from_iterate(I, weights, flags)
    O = subprob_obj.solve()

    # Convergence + autotune
    O = convergence.check_convergence_tolerance(problem, {}, O)
    O = baseline_autotune(problem, {}, O)

    display_baseline_subprob_status(problem, {}, O)
    return O


# ===================================
# Module flags (gating via Parameters)
# ===================================
@dataclass
class ModuleFlags:
    path: float = 1.0
    nfz: float = 1.0
    aux: float = 1.0
    term: float = 1.0
    dyn: float = 1.0
    tr: float = 1.0
    true: float = 1.0
    dual: float = 1.0
    vb: float = 1.0


# =========================
# Subproblem (build-once)
# =========================
class Subproblem:
    """Reusable convex subproblem."""

    def __init__(self, problem: Dict[str, Any]) -> None:
        self.problem = problem
        self.params = problem["params"]
        P = self.params

        # Core structure
        self.N = int(P["N"])
        self.n = int(P["n"])
        self.m = int(P["m"])
        self.nz = int(P["nz"])
        self.bools = P["bools"]
        self.free_T = bool(self.bools["free_final_time"])
        self.equal_dt = bool(self.bools["equal_dt"])
        self.buff_dyn = self.bools["buff_dyn"]
        self.flag_autotune = self.bools["flag_autotune"]

        # Dimensions
        self.n_path = int(P["n_path"])
        self.n_nfz = int(P["n_nfz"])
        self.n_aux = int(P.get("n_aux", 0))
        self.n_term = int(P["n_term"] + P["n_term_ineq"])
        self.n_dyn = int(P.get("n_dyn", P["nz"]))
        self.Npm = int(P.get("Npm", 0))
        self.n_plus = int(P.get("n_plus", 0))
        self.n_minus = int(P.get("n_minus", 0))

        # Create variables/params
        self._create_variables()
        self._create_parameters()

        # Build baseline constraints + cost
        self.constraints: List[cp.constraints.constraint.Constraint] = []
        self._build_constraints()
        base_cost = self._build_cost()

        # --------------------------------------------
        # Dynamic import of model-specific extensions
        # --------------------------------------------
        model_name = P.get("model_name", None)
        if model_name is not None:
            try:
                model_module = importlib.import_module(f"trajopt.problem_models.{model_name}")
            except ImportError:
                raise ImportError(f"Model module trajopt.problem_models.{model_name} not found")

            # Build context dictionary
            ctx = {
                "N": self.N,
                "n": self.n,
                "m": self.m,
                "params": self.params,
                "du": self.du,
                "dz": self.dz,
                "dt": self.dt,
                "us_ref": self.problem["I"][-1]["us_ref"],
                "zs_ref": self.problem["I"][-1]["zs_ref"],
                "ts_ref": self.problem["I"][-1]["ts_ref"],
            }

            # Dynamically call any custom_* hooks
            custom_inputs = getattr(model_module, "custom_inputs", None)
            custom_vars = getattr(model_module, "custom_subprob_variables", None)
            custom_constr = getattr(model_module, "custom_subprob_constraints", None)
            custom_cost = getattr(model_module, "custom_subprob_cost", None)

            if callable(custom_inputs):
                ctx = custom_inputs(self.problem, ctx)
            if callable(custom_vars):
                ctx = custom_vars(self.problem, ctx)
            if callable(custom_constr):
                self.constraints = custom_constr(self.problem, self.constraints, ctx)
            if callable(custom_cost):
                base_cost = custom_cost(self.problem, base_cost, ctx)

        # Compile the CVXPY problem
        self.subproblem = cp.Problem(cp.Minimize(base_cost), self.constraints)

    # --------------------------------------------
    # Core variable/parameter creation (unchanged)
    # --------------------------------------------
    def _create_variables(self) -> None:
        N, n, m, nz = self.N, self.n, self.m, self.nz
        self.dz = cp.Variable((N, n), name="dz")
        self.du = cp.Variable((N, m), name="du")
        self.dt = cp.Variable((N - 1, 1), name="dt")
        self.vb_aux = cp.Variable((N, self.n_aux), name="vb_aux") if self.n_aux > 0 else None

    def _create_parameters(self) -> None:
        N, n, m, nz = self.N, self.n, self.m, self.nz
        self.Ak = cp.Parameter((N - 1, n, n))
        self.Bk = cp.Parameter((N - 1, n, m))
        self.Bkp = cp.Parameter((N - 1, n, m))
        self.Sk = cp.Parameter((N - 1, n, 1))
        self.zs_m = cp.Parameter((N, n))
        self.zs_ref = cp.Parameter((N, n))
        self.us_ref = cp.Parameter((N, m))
        self.dts_ref = cp.Parameter((N - 1, 1))
        self.eps_ctcs = cp.Parameter(nonneg=True, name="eps_ctcs")

    # -------------------------------
    # Baseline constraints/cost logic
    # -------------------------------
    def _build_constraints(self) -> None:
        P, N, n, m, nz = self.params, self.N, self.n, self.m, self.nz
        self.constraints += [self.du[:, 0] == 0] if P["bools"].get("init_ctrl", False) else []

    def _build_cost(self) -> cp.Expression:
        return cp.sum_squares(self.dz) + cp.sum_squares(self.du)

    # -------------------------------
    # DPP updates + solve interface
    # -------------------------------
    def update_from_iterate(self, I: Dict[str, Any], weights: Dict[str, Any], flags: Optional[ModuleFlags] = None) -> None:
        Ak, Bk, Bkp, Sk, zs_minus = discretize.compute_linsys_discrete(I["zs_ref"], I["us_ref"], I["dts_ref"], self.problem)
        self.Ak.value, self.Bk.value, self.Bkp.value, self.Sk.value, self.zs_m.value = Ak, Bk, Bkp, Sk, zs_minus
        self.zs_ref.value = I["zs_ref"]
        self.us_ref.value = I["us_ref"]
        self.dts_ref.value = I["dts_ref"].reshape(self.N - 1, 1)
        self.eps_ctcs.value = self.params["eps_ctcs"]

    def solve(self) -> Dict[str, Any]:
        solver_name = self.params.get("solver_opts", {}).get("solver", "ECOS")
        self.subproblem.solve(solver=solver_name, warm_start=True)
        return self._collect_outputs()

    def _collect_outputs(self) -> Dict[str, Any]:
        O = {
            "dz_s": self.dz.value,
            "du_s": self.du.value,
            "dt_s": self.dt.value,
            "solve_time": self.subproblem.solver_stats.solve_time * 1000
            if self.subproblem.solver_stats
            else None,
        }
        return O


# ===========================
# Baseline autotune wrapper
# ===========================
def baseline_autotune(problem: Dict[str, Any], local_vars: Dict[str, Any], O: Dict[str, Any]) -> Dict[str, Any]:
    flag = problem["params"]["bools"]["flag_autotune"]
    if flag == 1:
        O = hp.autotune1(problem, local_vars, O)
    elif flag == 2:
        O = hp.autotune2(problem, local_vars, O)
    elif flag == 3:
        O = hp.autotune3(problem, local_vars, O)
    return O


# ==========================================
# Baseline iteration status printout (kept)
# ==========================================
def display_baseline_subprob_status(problem: Dict[str, Any], local_vars: Dict[str, Any], O: Dict[str, Any]) -> None:
    conv = O.get("conv_data", {})
    chk_feas_path = conv.get("chk_feas_path", 0.0)
    chk_feas_nfz = conv.get("chk_feas_nfz", 0.0)
    ineq_vb = chk_feas_path + (chk_feas_nfz if chk_feas_nfz != 0 else 0.0)
    chk_dz = conv.get("chk_dz", 1e-12)
    chk_feas_term = conv.get("chk_feas_term", 1e-12)
    chk_feas_dyn = conv.get("chk_feas_dyn", 1e-12)
    log_dz = np.log10(max(chk_dz, 1e-12))
    log_vb_ineq = np.log10(max(ineq_vb, 1e-12))
    log_vb_term = np.log10(max(chk_feas_term, 1e-12))
    log_vb_dyn = np.log10(max(chk_feas_dyn, 1e-12))
    solve_stat = conv.get("status", "UNKNOWN")
    iter_num = problem["I"][-1].get("iter_num", -1)
    nt = problem["params"]["nondim"]["nt"] if "nondim" in problem["params"] else 1.0
    ncost = problem["params"]["nondim"]["ncost"] if "nondim" in problem["params"] else 1.0
    Ts = O.get("Ts", 0.0)
    cost = O.get("cost", 0.0)
    prop_time = O.get("prop_time", 0.0)
    solve_ms = float(O.get("solve_time", 0.0) or 0.0)
    parse_ms = float(O.get("parse_time", 0.0) or 0.0)
    print(
        "     {:02d}     |    {:07.1f}   |   {:06.1f}  |   {:06.1f}   |   {:+04.1f}    |      {:+05.1f}      |    {:+05.1f}    |     {:+05.1f}   |    {:s}    |   {:4.2f}   |  {:4.1f}".format(
            int(iter_num),
            float(prop_time),
            solve_ms,
            parse_ms,
            float(log_dz),
            float(log_vb_ineq),
            float(log_vb_term),
            float(log_vb_dyn),
            str(solve_stat),
            float(Ts * nt),
            float(cost * ncost),
        )
    )
