import numpy as np
import cvxpy as cp
from trajopt.utils import tools

def configure_penalty_weights(problem, method, subconstraints=None):
    """
    Configure all scalar and matrix weights for the optimization method,
    using YAML-defined parameters under method.penalty.
    """

    idx = method.index_map
    penalty = tools.AttrDict(method.penalty)

    # --- Default weights ---
    n_ineq = problem.index_map.n.nonconvex_inequality
    W_stack = tools.AttrDict()
    dual_stack = tools.AttrDict()

    W_stack.nonconvex_inequality        = np.zeros((method.index_map.N.time_grid, n_ineq))
    W_stack.final_state                 = np.zeros(problem.index_map.n.final_state + problem.index_map.n.term_ineq + problem.index_map.n.term_ctcs)
    W_stack.dynamics                    = np.zeros((method.index_map.N.time_grid - 1, problem.index_map.n.z))
    W_stack.plus_real                   = np.zeros((method.index_map.N.pm_real, method.index_map.n.plus_real))
    W_stack.minus_real                  = np.zeros((method.index_map.N.pm_real, method.index_map.n.minus_real))
    W_stack.plus_ctcs                   = np.zeros((method.index_map.N.pm_ctcs, method.index_map.n.plus_ctcs))
    W_stack.minus_ctcs                  = np.zeros((method.index_map.N.pm_ctcs, method.index_map.n.minus_ctcs))

    dual_stack.nonconvex_inequality     = np.zeros((method.index_map.N.time_grid, n_ineq))
    dual_stack.final_state              = np.zeros(problem.index_map.n.final_state + problem.index_map.n.term_ineq + problem.index_map.n.term_ctcs)
    dual_stack.dynamics                 = np.zeros((method.index_map.N.time_grid - 1, problem.index_map.n.z))
    dual_stack.plus_real                = np.zeros((method.index_map.N.pm_real, method.index_map.n.plus_real))
    dual_stack.minus_real               = np.zeros((method.index_map.N.pm_real, method.index_map.n.minus_real))
    dual_stack.plus_ctcs                = np.zeros((method.index_map.N.pm_ctcs, method.index_map.n.plus_ctcs))
    dual_stack.minus_ctcs               = np.zeros((method.index_map.N.pm_ctcs, method.index_map.n.minus_ctcs))

    # local block arrays
    W_ineq                            = np.zeros((method.index_map.N.time_grid, problem.index_map.n.nonconvex_inequality))
    dual_ineq                         = np.zeros((method.index_map.N.time_grid, problem.index_map.n.nonconvex_inequality))

    penalty.wtr_z = 1 / (2 * penalty.alpha_z) * (1 / (method.index_map.N.time_grid * (method.index_map.n.z)))
    penalty.wtr_u = 1 / (2 * penalty.alpha_u) * (1 / (method.index_map.N.time_grid * (method.index_map.n.control)))

    # === Autotune modes (flag_autotune ∈ {0,2,3,al-scvx}) ===
    if str(method.flags.flag_autotune) in {"0", "2", "3", "al-scvx"}:

        # --- Buffer weights ---
        if str(method.flags.flag_autotune) in {"0", "al-scvx"}:
            if str(method.flags.flag_autotune) == "0":

                w_ineq      = penalty.config.scale_w.ineq   * penalty.config.scale_w.default / method.index_map.N.time_grid
                w_dyn       = penalty.config.scale_w.dyn    * penalty.config.scale_w.default / (method.index_map.N.time_grid - 1)
                w_term      = penalty.config.scale_w.term   * penalty.config.scale_w.default

            else:
                w_ineq      = penalty.config.scale_w.default / method.index_map.N.time_grid
                w_dyn       = penalty.config.scale_w.default / (method.index_map.N.time_grid - 1)
                w_term      = penalty.config.scale_w.default
        else:
            penalty.config.scale_w.default = 1
            w_ineq         = penalty.config.scale_w.default / method.index_map.N.time_grid
            w_dyn           = penalty.config.scale_w.default / (method.index_map.N.time_grid - 1)
            w_term          = penalty.config.scale_w.default

        W_ineq += w_ineq

        # Stack into the master W_nonconvex_inequality
        W_stack.nonconvex_inequality = W_ineq

        if method.flags.dynamics_nonconvex or method.flags.ctcs != "none":

            z_state_idx = idx.indices.z.state
            z_ctcs_idx  = idx.indices.z.ctcs
            term_idx    = idx.indices.constraints.final_state 
            
            # real dynamics portion weights
            if method.flags.buff_dyn in {"l1", "l2"}:
                W_stack.dynamics[:, z_state_idx] += w_dyn

            elif method.flags.buff_dyn in {"quad-1", "quad-2", "quad-3"}:
                W_stack.plus_real += w_dyn
                W_stack.minus_real += w_dyn
                W_stack.final_state[term_idx["eq"]] += w_term
                W_stack.final_state[term_idx["ineq"]] += w_term

            else:
                if len(term_idx.eq) > 0:
                    W_stack.final_state[term_idx.eq] += w_term

                if len(term_idx.ineq) > 0:
                    W_stack.final_state[term_idx.ineq] += w_term

            # ctcs portion weights
            if method.flags.ctcs in {"l1", "l2"}:
                W_stack.dynamics[:, z_ctcs_idx] += w_dyn
                W_stack.final_state[term_idx["ctcs"]] += w_term
    
            elif method.flags.ctcs in {"quad-1", "quad-2", "quad-3"}:
                W_stack.plus_ctcs += w_dyn
                W_stack.minus_ctcs += w_dyn
                W_stack.final_state[term_idx["ctcs"]] += w_term
            
            else:
                W_stack.final_state[term_idx.ctcs] += w_term


    # === Autotune mode: {1,3,al-scvx} ===
    if str(method.flags.flag_autotune) in {"1", "3", "al-scvx"}:

        dual_ineq += penalty.eps_nonzero1

        dual_stack.nonconvex_inequality = dual_ineq

        if method.flags.dynamics_nonconvex or method.flags.ctcs != "none":

            buff_dyn       = str(method.flags.buff_dyn)
            buff_dyn_dual  = str(method.flags.buff_dyn_dual)
            ctcs_flag      = str(method.flags.ctcs)
            ctcs_dual      = str(method.flags.ctcs_dual)

            z_state_idx = idx.indices.z.state
            z_ctcs_idx  = idx.indices.z.ctcs
            term_idx    = idx.indices.constraints.final_state

            eps = penalty.eps_nonzero1

            # real dynamics duals
            if buff_dyn in {"l1", "l2"}:
                dual_stack.dynamics[:, z_state_idx] += eps

            elif buff_dyn in {"quad-1", "quad-2", "quad-3"}:
                if buff_dyn_dual == "l1":
                    dual_stack.plus_real  += eps
                    dual_stack.minus_real += eps
                else:
                    dual_stack.dynamics[:, z_state_idx] += eps

            else:
                if len(term_idx.eq) > 0:
                    dual_stack.final_state[term_idx.eq] += eps
                if len(term_idx.ineq) > 0:
                    dual_stack.final_state[term_idx.ineq] += eps

            # ctcs duals
            if ctcs_flag in {"l1", "l2"}:
                dual_stack.dynamics[:, z_ctcs_idx] += eps

            elif ctcs_flag in {"quad-1", "quad-2", "quad-3"}:
                if ctcs_dual == "l1":
                    dual_stack.plus_ctcs  += eps
                    dual_stack.minus_ctcs += eps
                else:
                    dual_stack.dynamics[:, z_ctcs_idx] += eps

            else:
                dual_stack.final_state[term_idx.ctcs] += eps

    method.penalty = penalty

    if subconstraints is not None:
        subconstraints.apply_stacked_W_and_dual(W_stack, dual_stack, method)

    return W_stack, dual_stack


# -------------- PENALTIES ----------------------------------------------------------------------------------------

def build_virtual_buffer_cost(subprob) -> cp.Expression:
    """
    Virtual-buffer cost VB with SEPARATE buffering logic for:
        • real dynamics  (buff_dyn flag)
        • CTCS dynamics  (ctcs flag)
    """
    method  = subprob.method
    problem = subprob.problem

    N      = subprob.N.time_grid
    idx_real = subprob.indices.z.real
    n_real   = subprob.n.real
    n_ctcs = problem.index_map.n.ctcs

    VB = 0.0

    # ------------------------------------------------------------
    # TERMINAL TERM (REAL + CTCS)
    # ------------------------------------------------------------
    if subprob.vb_term is not None and subprob.vb_term.size > 0:
        VB += cp.sum_squares(cp.diag(subprob.W_sqrt.final_state) @ subprob.vb_term)

    # ------------------------------------------------------------
    # STACKED NONLINEAR INEQUALITY BUFFERS
    # ------------------------------------------------------------
    if subprob.vb_ineq is not None and subprob.n.nonconvex_inequality > 0:
        for k in range(N):
            VB += cp.sum_squares(
                cp.diag(subprob.W_sqrt.nonconvex_inequality[k, :]) @ subprob.vb_ineq[k, :]
            )

    # ============================================================
    # REAL DYNAMICS BUFFERING   (state + time)
    # ============================================================
    mode_real = method.flags.buff_dyn   # {"none","term","l1","l2","quad-1","quad-2"}

    if subprob.vb_dyn_real_p is not None and n_real > 0:
        diff_real = subprob.vb_dyn_p[:, idx_real] - subprob.vb_dyn_m[:, idx_real]

        # --------------------------------------------------------
        # L1 penalty
        # --------------------------------------------------------
        if mode_real == "l1":
            for k in range(N - 1):
                VB += 10000 * cp.norm1(diff_real[k, :])

        # --------------------------------------------------------
        # L2 penalty
        # --------------------------------------------------------
        elif mode_real == "l2":
            for k in range(N - 1):
                VB += cp.sum_squares(
                    cp.diag(subprob.W_sqrt.dynamics[k, idx_real]) @ diff_real[k, :]
                )

        # --------------------------------------------------------
        # QUAD-1 or QUAD-3: k = 1 only
        # --------------------------------------------------------
        elif mode_real == "quad-1" or mode_real == "quad-3":
            if subprob.vb_plus_real is not None:
                VB += cp.sum_squares(
                    cp.diag(subprob.W_sqrt.plus_real[0, :]) @ subprob.vb_plus_real[0, :]
                )
            if subprob.vb_minus_real is not None:
                VB += cp.sum_squares(
                    cp.diag(subprob.W_sqrt.minus_real[0, :]) @ subprob.vb_minus_real[0, :]
                )

        # --------------------------------------------------------
        # QUAD-2 : Per-time-step quadratic penalties
        # --------------------------------------------------------
        elif mode_real == "quad-2":
            if subprob.vb_plus_real is not None:
                for k in range(subprob.N.pm_real):
                    VB += cp.sum_squares(
                        cp.diag(subprob.W_sqrt.plus_real[k, :]) @ subprob.vb_plus_real[k, :]
                    )
            if subprob.vb_minus_real is not None:
                for k in range(subprob.N.pm_real):
                    VB += cp.sum_squares(
                        cp.diag(subprob.W_sqrt.minus_real[k, :]) @ subprob.vb_minus_real[k, :]
                    )

    # ============================================================
    # CTCS DYNAMICS BUFFERING   (last n_ctcs states)
    # ============================================================
    mode_ctcs = method.flags.ctcs       # {"none","term","l1","l2","quad-1","quad-2"}

    if subprob.vb_dyn_p is not None and n_ctcs > 0:
        diff_ctcs = subprob.vb_dyn_p[:, subprob.indices.z.ctcs] - subprob.vb_dyn_m[:, subprob.indices.z.ctcs]

        # --------------------------------------------------------
        # L1 penalty
        # --------------------------------------------------------
        if mode_ctcs == "l1":
            for k in range(N - 1):
                VB += subprob.w_dyn_row[k] * cp.norm1(diff_ctcs[k, :])

        # --------------------------------------------------------
        # L2 penalty
        # --------------------------------------------------------
        elif mode_ctcs == "l2":
            for k in range(N - 1):
                VB += cp.sum_squares(
                    cp.diag(subprob.W_sqrt.dynamics[k, subprob.indices.z.ctcs]) @ diff_ctcs[k, :]
                )

        # --------------------------------------------------------
        # QUAD-1 or QUAD-3: k = 1 only
        # --------------------------------------------------------
        elif mode_ctcs == "quad-1" or mode_ctcs == "quad-3":
            if subprob.vb_plus_ctcs is not None:
                VB += cp.sum_squares(
                    cp.diag(subprob.W_sqrt.plus_ctcs[0, :]) @ subprob.vb_plus_ctcs[0, :]
                )
            if subprob.vb_minus_ctcs is not None:
                VB += cp.sum_squares(
                    cp.diag(subprob.W_sqrt.minus_ctcs[0, :]) @ subprob.vb_minus_ctcs[0, :]
                )

        # --------------------------------------------------------
        # QUAD-2 : Per-time-step quadratic penalties
        # --------------------------------------------------------
        elif mode_ctcs == "quad-2":
            if subprob.vb_plus_ctcs is not None:
                for k in range(subprob.N.pm_ctcs):
                    VB += cp.sum_squares(
                        cp.diag(subprob.W_sqrt.plus_ctcs[k, :]) @ subprob.vb_plus_ctcs[k, :]
                    )
            if subprob.vb_minus_ctcs is not None:
                for k in range(subprob.N.pm_ctcs):
                    VB += cp.sum_squares(
                        cp.diag(subprob.W_sqrt.minus_ctcs[k, :]) @ subprob.vb_minus_ctcs[k, :]
                    )

    # ------------------------------------------------------------
    # Final scaling (flag multiplies entire VB block)
    # ------------------------------------------------------------
    return 0.5 * subprob.method.flags.flag_vb * VB

def build_dual_buffer_cost(subprob) -> cp.Expression:
    """
    Dual penalty term DUAL for inequality constraints, dynamic buffers,
    and aggregate quadratic-buffer (quad-1, quad-2) modes.
    """
    method = subprob.method
    mode_real     = method.flags.buff_dyn        # {'none','term','l1','l2','quad-1','quad-2','quad-3'}
    mode_real_dual = method.flags.buff_dyn_dual  # MATLAB: buff_dyn_dual
    mode_ctcs     = method.flags.ctcs            # {'none','term','l1','l2','quad-1','quad-2','quad-3'}
    mode_ctcs_dual = method.flags.ctcs_dual

    DUAL = 0.0

    # ============================================================
    # Unified NONCONVEX_INEQUALITY DUAL COST
    # ============================================================
    if subprob.vb_ineq is not None and subprob.n.nonconvex_inequality > 0:
        DUAL += cp.sum(cp.multiply(subprob.vb_ineq, subprob.dual.nonconvex_inequality))

    # ============================================================
    # Dynamic dual: dual_dynamics .* (vb_dyn_plus - vb_dyn_minus)
    # ============================================================
    diff = subprob.vb_dyn_p - subprob.vb_dyn_m
    DUAL += cp.sum(cp.multiply(diff, subprob.dual.dynamics))

    # ============================================================
    # QUAD-1, QUAD-2, QUAD-3 dual components
    # ============================================================
    # These exist ONLY if buff_dyn_dual == 'l1'
    if mode_real_dual == "l1":

        # -----------------------
        # QUAD-1 or QUAD-3: k == 1 only
        # -----------------------
        if mode_real == "quad-1" or mode_real == "quad-3":
            if subprob.vb_plus_real is not None and subprob.dual.plus_real is not None:
                DUAL += subprob.dual.plus_real[0, :] @ subprob.vb_plus_real[0, :]
            if subprob.vb_minus_real is not None and subprob.dual.minus_real is not None:
                DUAL += subprob.dual.minus_real[0, :] @ subprob.vb_minus_real[0, :]

        # -----------------------
        # QUAD-2  (per time index)
        # -----------------------
        elif mode_real == "quad-2":
            if subprob.vb_plus_real is not None and subprob.dual.plus_real is not None:
                for k in range(subprob.N.pm_real):
                    DUAL += subprob.dual.plus_real[k, :] @ subprob.vb_plus_real[k, :]
            if subprob.vb_minus_real is not None and subprob.dual.minus_real is not None:
                for k in range(subprob.N.pm_real):
                    DUAL += subprob.dual.minus_real[k, :] @ subprob.vb_minus_real[k, :]

    # ============================================================
    # CTCS DUAL COST
    # ============================================================
    # These exist ONLY if ctcs_dual == 'l1'
    if mode_ctcs_dual == "l1":

        # -----------------------
        # QUAD-1 or QUAD-3: k == 1 only
        # -----------------------
        if mode_ctcs == "quad-1" or mode_ctcs == "quad-3":
            if subprob.vb_plus_ctcs is not None and subprob.dual.plus_ctcs is not None:
                DUAL += subprob.dual.plus_ctcs[0, :] @ subprob.vb_plus_ctcs[0, :]
            if subprob.vb_minus_ctcs is not None and subprob.dual.minus_ctcs is not None:
                DUAL += subprob.dual.minus_ctcs[0, :] @ subprob.vb_minus_ctcs[0, :]

        # -----------------------
        # QUAD-2  (per time index)
        # -----------------------
        elif mode_ctcs == "quad-2":
            if subprob.vb_plus_ctcs is not None and subprob.dual.plus_ctcs is not None:
                for k in range(subprob.N.pm_ctcs):
                    DUAL += subprob.dual.plus_ctcs[k, :] @ subprob.vb_plus_ctcs[k, :]
            if subprob.vb_minus_ctcs is not None and subprob.dual.minus_ctcs is not None:
                for k in range(subprob.N.pm_ctcs):
                    DUAL += subprob.dual.minus_ctcs[k, :] @ subprob.vb_minus_ctcs[k, :]

    # ============================================================
    # Terminal dual cost
    # ============================================================
    if subprob.vb_term is not None and subprob.n.final_state > 0:
        DUAL += subprob.dual.final_state @ subprob.vb_term

    return DUAL


# -------------- AUTOTUNING SCHEMES ----------------------------------------------------------------------------------------

def autotune1(subproblem, conv_data, conv_data_prev, iter_num):
    """
    Unified version of autotune1 including dual_plus and dual_minus.
    Updates subproblem.dual_stack directly.
    """
    method = subproblem.method
    problem = subproblem.problem

    # Extract primal buffers
    vb_ineq  = np.array(conv_data.vb_ineq)
    vb_term  = np.array(conv_data.vb_terminal)
    vb_dyn   = np.array(conv_data.vb_dyn)
    
    vb_plus_real  = np.array(conv_data.vb_plus_real)
    vb_minus_real = np.array(conv_data.vb_minus_real)
    
    vb_plus_ctcs  = np.array(conv_data.vb_plus_ctcs)
    vb_minus_ctcs = np.array(conv_data.vb_minus_ctcs)

    # Extract current duals from subproblem
    dual_ineq  = subproblem.dual_stack.nonconvex_inequality
    dual_dyn   = subproblem.dual_stack.dynamics
    dual_term  = subproblem.dual_stack.final_state

    dual_plus_real  = subproblem.dual_stack.plus_real
    dual_minus_real = subproblem.dual_stack.minus_real

    dual_plus_ctcs  = subproblem.dual_stack.plus_ctcs
    dual_minus_ctcs = subproblem.dual_stack.minus_ctcs

    # Hyperparameters
    if method.flags.stepsize_auto_dual:
        beta = gamma = 1 / iter_num
    else:
        penalty_rec = method.penalty
        beta  = penalty_rec.beta
        gamma = penalty_rec.gamma

    # ==========================================
    # Dual updates
    # ==========================================

    W_ineq = subproblem.W_stack.nonconvex_inequality

    # inequality
    dual_ineq_plus = np.maximum(0, W_ineq * vb_ineq + dual_ineq)

    W_dyn = subproblem.W_stack.dynamics

    # NOTE: testing the augmented lagrangian update rule for duals
    dual_dyn_plus = W_dyn * vb_dyn + dual_dyn

    W_term = subproblem.W_stack.final_state

    # terminal
    dual_term_plus = W_term * vb_term + dual_term

    # plus/minus (quadratic 1-norm decomposition)
    W_plus_real  = subproblem.W_stack.plus_real
    W_minus_real = subproblem.W_stack.minus_real

    dual_plus_plus_real  = np.maximum(0, W_plus_real * vb_plus_real  + dual_plus_real)
    dual_minus_plus_real = np.maximum(0, W_minus_real * vb_minus_real + dual_minus_real)

    W_plus_ctcs  = subproblem.W_stack.plus_ctcs
    W_minus_ctcs = subproblem.W_stack.minus_ctcs

    dual_plus_plus_ctcs  = np.maximum(0, W_plus_ctcs * vb_plus_ctcs  + dual_plus_ctcs)
    dual_minus_plus_ctcs = np.maximum(0, W_minus_ctcs * vb_minus_ctcs + dual_minus_ctcs)

    # ==========================================
    # Saturation thresholds
    # ==========================================
    conv = method.conv
    eps_ineq = conv.get("eps_ineq", 1e-6)
    eps_term = conv["eps_term"]
    eps_dyn  = conv["eps_dyn"]
    eps_quad = conv.get("eps_quad", eps_dyn)   # for vb_plus/vb_minus

    # ==========================================
    # Update subproblem duals directly
    # ========================================== 
    subproblem.dual_stack.nonconvex_inequality = dual_ineq_plus
    subproblem.dual_stack.dynamics             = dual_dyn_plus
    subproblem.dual_stack.final_state          = dual_term_plus
    subproblem.dual_stack.plus_real            = dual_plus_plus_real
    subproblem.dual_stack.minus_real           = dual_minus_plus_real
    subproblem.dual_stack.plus_ctcs            = dual_plus_plus_ctcs
    subproblem.dual_stack.minus_ctcs           = dual_minus_plus_ctcs

    # Return dual update info for logging
    return {
        "dmu_ineq": dual_ineq_plus - dual_ineq,
        "dmu_eq":   dual_term_plus - dual_term,
        "dmu_plus_real": dual_plus_plus_real  - dual_plus_real,
        "dmu_minus_real": dual_minus_plus_real - dual_minus_real,
        "dmu_plus_ctcs": dual_plus_plus_ctcs  - dual_plus_ctcs,
        "dmu_minus_ctcs": dual_minus_plus_ctcs - dual_minus_ctcs,
    }


def autotune2(subproblem, conv_data, conv_data_prev, iter_num):
    """
    Unified stacked-inequality version of autotune2.
    Updates subproblem.W_stack directly.
    """
    method = subproblem.method
    problem = subproblem.problem
    
    # Extract variables from conv_data
    N = method.index_map.N.time_grid
    vb_ineq = np.array(conv_data.vb_ineq)
    vb_dyn  = np.array(conv_data.vb_dyn)
    vb_term = np.array(conv_data.vb_terminal)

    vb_plus_real = np.array(conv_data.vb_plus_real)
    vb_minus_real = np.array(conv_data.vb_minus_real)
    vb_plus_ctcs = np.array(conv_data.vb_plus_ctcs)
    vb_minus_ctcs = np.array(conv_data.vb_minus_ctcs)

    # Get current weights from subproblem
    W_ineq = np.array(subproblem.W_stack.nonconvex_inequality)
    W_dyn  = np.array(subproblem.W_stack.dynamics)
    W_term = np.array(subproblem.W_stack.final_state)

    W_plus_real  = np.array(subproblem.W_stack.plus_real)
    W_minus_real = np.array(subproblem.W_stack.minus_real)
    W_plus_ctcs  = np.array(subproblem.W_stack.plus_ctcs)
    W_minus_ctcs = np.array(subproblem.W_stack.minus_ctcs)

    # ==========================================
    # Saturation thresholds
    # ==========================================
    conv = method.conv
    eps_feas_ineq = conv.eps_ineq
    eps_feas_term = conv.eps_term
    eps_feas_dyn  = conv.eps_dyn

    eps_target_term =  np.maximum(1.0*eps_feas_term, conv.fac_eps * np.abs(conv_data.vb_terminal))
    eps_target_ineq =  np.maximum(1.0*eps_feas_ineq, conv.fac_eps * np.abs(conv_data.vb_ineq))
    eps_target_dyn  =  np.maximum(1.0*eps_feas_dyn , conv.fac_eps * np.abs(conv_data.vb_dyn))

    conv_data.eps_target_term = eps_target_term.copy()
    conv_data.eps_target_ineq = eps_target_ineq.copy()
    conv_data.eps_target_dyn  = eps_target_dyn.copy()

    penalty_rec = method.penalty
    eps_nonzero2 = penalty_rec.eps_nonzero2

    buff_dyn = method.flags.buff_dyn
    ctcs = method.flags.ctcs

    Wh_ineq = np.zeros((N, problem.index_map.n.nonconvex_inequality))
    Wh_dyn  = np.zeros((N - 1, problem.index_map.n.z))
    Wh_term = np.zeros(problem.index_map.n.term_total)

    Wh_plus_real  = np.zeros((method.index_map.N.pm_real, method.index_map.n.plus_real))
    Wh_minus_real = np.zeros((method.index_map.N.pm_real, method.index_map.n.minus_real))
    Wh_plus_ctcs  = np.zeros((method.index_map.N.pm_ctcs, method.index_map.n.plus_ctcs))
    Wh_minus_ctcs = np.zeros((method.index_map.N.pm_ctcs, method.index_map.n.minus_ctcs))

    z_state_idx = method.index_map.indices.z.state
    z_ctcs_idx = method.index_map.indices.z.ctcs

    # ==========================================
    # COMPUTE AUTOTUNE UPDATES
    # ==========================================

    if buff_dyn == "quad-1":
        Wh_plus_real = np.abs(W_plus_real @ vb_plus_real) / eps_feas_dyn[z_state_idx]
        Wh_minus_real = np.abs(W_minus_real @ vb_minus_real) / eps_feas_dyn[z_state_idx]

    if ctcs == "quad-1":
        Wh_plus_ctcs = np.abs(W_plus_ctcs @ vb_plus_ctcs) / eps_feas_dyn[z_ctcs_idx]
        Wh_minus_ctcs = np.abs(W_minus_ctcs @ vb_minus_ctcs) / eps_feas_dyn[z_ctcs_idx]

    # TODO: add quad-3 case

    if buff_dyn == "quad-3":
        for j in range(problem.index_map.n.state):
            Wh_plus_real[:, j] = np.sum(np.abs(np.diag(W_plus_real[:, j]) @ vb_plus_real[:, j] / np.min(eps_feas_dyn)))
            Wh_minus_real[:, j] = np.sum(np.abs(np.diag(W_minus_real[:, j]) @ vb_minus_real[:, j] / np.min(eps_feas_dyn)))
    if ctcs == "quad-3":
        for j in range(problem.index_map.n.ctcs):
            Wh_plus_ctcs[:, j] = np.sum(np.abs(np.diag(W_plus_ctcs[:, j]) @ vb_plus_ctcs[:, j] / np.min(eps_feas_dyn)))
            Wh_minus_ctcs[:, j] = np.sum(np.abs(np.diag(W_minus_ctcs[:, j]) @ vb_minus_ctcs[:, j] / np.min(eps_feas_dyn)))

    for k in range(N):
        dual_ineq_buff = np.diag(W_ineq[k, :]) @ vb_ineq[k, :].flatten()

        if problem.index_map.n.nonconvex_inequality > 0:
            Wh_ineq[k, :] = np.minimum(np.abs(dual_ineq_buff / eps_target_ineq[k]), 1e4)
        else:
            Wh_ineq[k, :] = np.abs(dual_ineq_buff) # rho*np.ones_like(Wh_ineq[k, :])

        if k < N - 1:
            dual_dyn_buff = np.diag(W_dyn[k, :]) @ vb_dyn[k, :]
            
            if buff_dyn == "l1":
                Wh_dyn[k, z_state_idx] = np.minimum(np.sum(np.abs(dual_dyn_buff[z_state_idx]) / eps_feas_dyn[z_state_idx]), 1e5)
            if buff_dyn != "none":
                Wh_dyn[k, z_state_idx] = np.minimum(np.abs(dual_dyn_buff[z_state_idx] / eps_target_dyn[k, z_state_idx]), 1e5) # rho*np.ones_like(W_dyn[k, z_state_idx]) #
            # TODO(Skye): REVISIT
            #if buff_dyn == "l2":
            #     Wh_dyn[k, z_state_idx] = np.abs(dual_dyn_buff[z_state_idx] / eps_feas_dyn[z_state_idx])

            if ctcs == "l1":
                Wh_dyn[k, z_ctcs_idx] = np.sum(np.abs(dual_dyn_buff[z_ctcs_idx]) / eps_target_dyn[k, z_ctcs_idx])
            elif ctcs != "none":
                Wh_dyn[k, z_ctcs_idx] = np.minimum(np.abs(dual_dyn_buff[z_ctcs_idx] / eps_target_dyn[k, z_ctcs_idx]), 1e5) #rho*np.ones_like(W_dyn[k, z_ctcs_idx])

            # TODO: THINK ABOUT THIS (MAYBE ONE IF ELSE) COME BACK TO THIS, SINGLE EPSILON ETC
            if buff_dyn == "quad-2":
                Wh_plus_real[k]  = np.minimum(np.sum(np.abs(np.diag(W_plus_real[k, :]) @ vb_plus_real[k, :] / eps_target_dyn[k, z_state_idx])), 1e5)
                Wh_minus_real[k] = np.minimum(np.sum(np.abs(np.diag(W_minus_real[k, :]) @ vb_minus_real[k, :] / eps_target_dyn[k, z_state_idx])), 1e5)
            
            if ctcs == "quad-2":
                Wh_plus_ctcs[k]  = np.minimum(np.sum(np.abs(np.diag(W_plus_ctcs[k, :]) @ vb_plus_ctcs[k, :] / eps_target_dyn[k, z_ctcs_idx])), 1e5)
                Wh_minus_ctcs[k] = np.minimum(np.sum(np.abs(np.diag(W_minus_ctcs[k, :]) @ vb_minus_ctcs[k, :] / eps_target_dyn[k, z_ctcs_idx])), 1e5)

    if problem.index_map.n.term_total > 0:
        dual_term_buff = np.diag(W_term) @ vb_term
        Wh_term = np.minimum(np.abs(dual_term_buff / eps_target_term).flatten(), 1e5) # rho*np.ones_like(W_term)

    # ==========================================
    # UPDATE WEIGHTS WITH COMPUTED AUTOTUNE UPDATES
    # ==========================================

    if np.sum(W_plus_real) > 0: Wh_plus_real[Wh_plus_real <= eps_nonzero2] = eps_nonzero2 
    if np.sum(W_minus_real) > 0: Wh_minus_real[Wh_minus_real <= eps_nonzero2] = eps_nonzero2
    if np.sum(W_plus_ctcs) > 0: Wh_plus_ctcs[Wh_plus_ctcs <= eps_nonzero2] = eps_nonzero2
    if np.sum(W_minus_ctcs) > 0: Wh_minus_ctcs[Wh_minus_ctcs <= eps_nonzero2] = eps_nonzero2

    if np.sum(W_ineq) > 0: Wh_ineq[Wh_ineq <= eps_nonzero2] = eps_nonzero2  
    if np.sum(W_dyn) > 0: Wh_dyn[Wh_dyn <= eps_nonzero2] = eps_nonzero2
    if np.sum(W_term) > 0: Wh_term[Wh_term <= eps_nonzero2] = eps_nonzero2

    # subproblem.W_stack.nonconvex_inequality = np.maximum(Wh_ineq, W_ineq)
    # subproblem.W_stack.dynamics             = np.maximum(Wh_dyn, W_dyn)
    # subproblem.W_stack.final_state          = np.maximum(Wh_term, W_term)
    # subproblem.W_stack.plus_real            = np.maximum(Wh_plus_real, W_plus_real)
    # subproblem.W_stack.minus_real           = np.maximum(Wh_minus_real, W_minus_real)
    # subproblem.W_stack.plus_ctcs            = np.maximum(Wh_plus_ctcs, W_plus_ctcs)
    # subproblem.W_stack.minus_ctcs           = np.maximum(Wh_minus_ctcs, W_minus_ctcs)
    # else:
    subproblem.W_stack.plus_real = Wh_plus_real
    subproblem.W_stack.minus_real = Wh_minus_real
    subproblem.W_stack.plus_ctcs = Wh_plus_ctcs
    subproblem.W_stack.minus_ctcs = Wh_minus_ctcs
    subproblem.W_stack.nonconvex_inequality = Wh_ineq
    subproblem.W_stack.dynamics = Wh_dyn
    subproblem.W_stack.final_state = Wh_term

    # Return diagnostics for logging
    return {
        "eps_feas": eps_feas_ineq,
        "term": {
            "Wxq": np.diag(W_term) @ vb_term if problem.index_map.n.term_total > 0 else None,
            "dual": dual_term_buff if problem.index_map.n.term_total > 0 else None
        }
    }


def autotune3(subproblem, conv_data, conv_data_prev, iter_num):
    """Combined autotune1 and autotune2."""
    dual_info = autotune1(subproblem, conv_data, conv_data_prev, iter_num)
    weight_info = autotune2(subproblem, conv_data, conv_data_prev, iter_num)
    

    return {"dual_update": dual_info, "weight_update": weight_info}