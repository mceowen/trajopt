import trajopt.utils.set_defaults           as defaults
import trajopt.utils.tools                  as tools
import trajopt.algorithm.initial_guess      as guess
import trajopt.algorithm.convergence        as convergence
import trajopt.algorithm.convexification    as convexify
import trajopt.utils.nondim                 as nondim
import numpy as np

def load_params(example_name):

    params = {}

    params["mission"] = {}
    params["model"] = {}
    params["method"] = {}

    # example configs
    example_pkg = f"trajopt.examples.{example_name}"
    example = {
        k: tools.load_yaml(example_pkg, f"{k}.yaml")
        for k in [
            "mission", "model", "method"
            ]
        }

    # base configs
    base_pkgs = {
        "mission": "trajopt.base_configs.missions",
        "model":  "trajopt.base_configs.models",
        "method": "trajopt.base_configs.methods",
    }

    base = {
        k: tools.load_yaml(base_pkgs[k], f"{name}.yaml")
        for k, name in {
            "mission": example["mission"]["mission_name"],
            "model":   example["model"]["model_name"],
            "method":  example["method"]["method_name"],
        }.items()
    }


    # general default configs
    default = {
        k: tools.load_yaml(base_pkgs[k], "default.yaml")
        for k in base_pkgs
        }

    # update params with defaults -> base -> example params
    for param_type in ['mission', 'model', 'method']:
        for update_type in [default, base, example]:
            if f'{update_type}' == 'default':
                params[param_type].update(default[param_type])
            else:
                params[param_type] = tools.deep_update(params[param_type], update_type[param_type])

    # update mission/method/model params based on configs
    params = nondim.set_nondim_params(params["model"]["z_types"],
                            params["model"]["u_types"],
                            params["model"]["anchor_types"],
                            params["model"]["anchor_scales"],
                            params,
                            base_unit_labels=params["model"]["base_unit_labels"])

    params = update_mission_params(params)
    params = defaults.set_params_constraint_default(params)
    params = update_method_params(params)

    return params

def update_mission_params(params):

    #======================
    # Path /NFZ constraints
    #======================

    nd = params["method"]["nondim"]["nd"]
    nt = params["method"]["nondim"]["nt"]
    nf = params["method"]["nondim"]["nf"]

    # no fly zones, specified by position and radius [rad]
    num_nfz = params["mission"]["bools"]["flag_nfz"]
    xc = params["mission"][f"nfz_{num_nfz}"]["xc"] / nd
    yc = params["mission"][f"nfz_{num_nfz}"]["yc"] / nd
    rc = params["mission"][f"nfz_{num_nfz}"]["rc"] / nd

    params["mission"]["nfz_idx"]       = np.arange(0, xc.size)
    params["mission"]["n_nfz"]         = len(params["mission"]["nfz_idx"])

    params["mission"].setdefault("obs", {})["posc"] = np.array([xc, yc]) # xc and yc may be vectors
    params["mission"]["obs"]["rc"]     = rc

    # set nondim for cost and constraints
    np_ineq = np.ones(params["mission"]["n_nfz"]) * nd**2
    ncost = params["method"]["nondim"]["nf"]**2 * nt

    params = nondim.set_cost_cnst_nondim_params(np_ineq, ncost, params)

    #====================
    # Boundary Conditions
    #====================
    # initial conditions

    # equality initial conditions
    params["mission"]["zi"]            = params["method"]["nondim"]["M"]["state"]["d2nd"] @ params["mission"]["zi"]
    params["mission"]["zi_idx"]        = np.arange(0, params["model"]["n"])

    # inequality initial conditions
    # none

    # equality terminal conditions
    params["mission"]["zf"]            = params["method"]["nondim"]["M"]["state"]["d2nd"] @ params["mission"]["zf"]  
    params["mission"]["zf_idx"]        = np.arange(0,params["model"]["n"])

    # control boundary conditions
    params["mission"]["ui"]            = -params["mission"]["ge"]*params["mission"]["mass"] / params["method"]["nondim"]["nf"]
    params["mission"]["uf"]            = -params["mission"]["ge"]*params["mission"]["mass"] / params["method"]["nondim"]["nf"]

    #==============================
    # Control and state constraints
    #==============================
    # no state constraints
    params["mission"]["z_min"]         = np.array([0, 0, 0.25]) / nd
    params["mission"]["z_min_idx"]     = np.arange(0,3)
    params["mission"]["z_max"]         = np.array([12, 12, 10]) / nd
    params["mission"]["z_max_idx"]     = np.arange(0,3)
    params["mission"]["u_norm_min"]    = 0.21 / nf
    params["mission"]["u_norm_max"]    = 8.12 / nf
    params["mission"]["udot_max"]      = 5*np.ones(3) / (nf / nt)# [N/s]
    params["mission"]["udot_max_idx"]  = np.arange(0,3)

    return params

def update_method_params(params):

    ### Time of flight constraints ###
    Ts_min                  = 1. / params["method"]["nondim"]["nt"]  # 50
    Ts_max                  = 20. / params["method"]["nondim"]["nt"]
    params["method"]["ddts_max"]      = 5. / ((params["method"]["N"] - 1) * params["method"]["nondim"]["nt"])  # 0.025
    params["method"]["dts_min"]       = Ts_min / (params["method"]["N"] - 1)
    params["method"]["dts_max"]       = Ts_max / (params["method"]["N"] - 1)

    #============================================
    # Optimization parameters and hyperparameters
    #============================================
    # PTR penalty weights
        # Wtr: weight for trust region cost                        
        # w_term: weight for terminal constraint buffer cost
        # w_path: weight for path constraint buffer cost
        # w_nfz: weight for path constraint buffer cost

    # === Baseline cost + trust region weights ===
    params["method"]["weights"]["w_cost"]         = 0.
    params["method"]["weights"]["eps_nonzero1"]   = 2e-1
    params["method"]["weights"]["eps_nonzero2"]   = 1e-10

    M_state  = params["method"]["nondim"]["M"]["state"]["nd2d"]
    avg_state_nd_sq  = np.mean(np.diag(M_state)**2)

    # === Trust region weights ===
    params["method"]["weights"].setdefault("alpha_z", 0.5)
    params["method"]["weights"].setdefault("alpha_u", np.inf)

    params["method"]["weights"]["wtr_z"]          = avg_state_nd_sq  * 1 / (2 * params["method"]["weights"]["alpha_z"])
    params["method"]["weights"]["wtr_u"]          = 0 if np.isinf(params["method"]["weights"]["alpha_u"]) else 1 / (2 * params["method"]["weights"]["alpha_u"])

    # === Autotune modes (flag_autotune ∈ {0,2,3,al-scvx}) ===
    if str(params["method"]["bools"]["flag_autotune"]) in {"0", "2", "3", "al-scvx"}:

        params["method"]["weights"].setdefault("beta", 1)
        params["method"]["weights"].setdefault("gamma", 1e-1)

        # --- Buffer weights ---
        if str(params["method"]["bools"]["flag_autotune"]) in {"0", "al-scvx"}:
            if "wbuff" not in params["method"]["weights"]:
                wbuff = 1e2
                if str(params["method"]["bools"]["flag_autotune"]) == "0":

                    w_nfz_dim  = wbuff / params["method"]["weights"]["w_fac_N"]
                    w_dyn_dim  = 1e5 * wbuff / params["method"]["weights"]["w_fac_Nm1"]
                    w_term_dim = 1e2 * wbuff

                    # scaled nondim weights to approximately preserve relative scaling between cost terms
                    M_nfz  = params["method"]["nondim"]["M"]["nfz"]["nd2d"]
                    M_dyn  = params["method"]["nondim"]["M"]["dyn"]["nd2d"]
                    M_term = params["method"]["nondim"]["M"]["term"]["nd2d"]

                    avg_nfz_nd_sq  = np.mean(np.diag(M_nfz)**2)
                    avg_dyn_nd_sq  = np.mean(np.diag(M_dyn)**2)
                    avg_term_nd_sq = np.mean(np.diag(M_term)**2)

                    w_nfz   = avg_nfz_nd_sq  * w_nfz_dim
                    w_dyn   = avg_dyn_nd_sq  * w_dyn_dim
                    w_term  = avg_term_nd_sq * w_term_dim
            else:
                wbuff = params["method"]["weights"]["wbuff"]
                w_nfz = wbuff / params["method"]["weights"]["w_fac_N"]
                w_dyn = wbuff / params["method"]["weights"]["w_fac_Nm1"]
                w_term = wbuff
        else:
            wbuff = 1
            w_nfz = wbuff / params["method"]["weights"]["w_fac_N"]
            w_dyn = wbuff / params["method"]["weights"]["w_fac_Nm1"]
            w_term = wbuff

        params["method"]["weights"]["W_nfz"] += w_nfz

        if params["method"]["bools"]["free_final_time"] or params["method"]["bools"]["ctcs"]:
            buff_dyn = str(params["method"]["bools"].get("buff_dyn", ""))
            if buff_dyn in {"l1", "l2"}:
                params["method"]["weights"]["W_dyn"] += w_dyn
            elif buff_dyn in {"quad-1", "quad-2", "quad-3"}:
                params["method"]["weights"]["W_plus"] += w_dyn
                params["method"]["weights"]["W_minus"] += w_dyn
            else:
                params["method"]["weights"]["W_term"] += w_term

    # === Autotune mode: {1,3,al-scvx} ===
    if str(params["method"]["bools"]["flag_autotune"]) in {"1", "3", "al-scvx"}:

        params["method"]["weights"].setdefault("beta", 1)
        params["method"]["weights"].setdefault("gamma", 1e-1)

        params["method"]["weights"]["dual_nfz"] += params["method"]["weights"]["eps_nonzero1"]

        if params["method"]["bools"]["free_final_time"]:
            buff_dyn = str(params["method"]["bools"].get("buff_dyn", ""))
            if buff_dyn == "term":
                params["method"]["weights"]["dual_term"] += params["method"]["weights"]["eps_nonzero1"]
            else:
                params["method"]["weights"]["dual_dyn"] += params["method"]["weights"]["eps_nonzero1"]

                if str(params["method"]["bools"].get("buff_dyn_dual", "")) == "l1":
                    params["method"]["weights"]["dual_plus"] += params["method"]["weights"]["eps_nonzero1"]
                    params["method"]["weights"]["dual_minus"] += params["method"]["weights"]["eps_nonzero1"]

    ### ctcs convergence adjustments ###
    ctcs_mult_state         = 1e0
    ctcs_mult_cnst          = 1e0 
    eps_ctcs                = 1e-4

    params["method"]["conv"]["setup"]["ctcs_mult_state"]                  = ctcs_mult_state
    params["method"]["conv"]["setup"]["ctcs_mult_cnst"]                   = ctcs_mult_cnst

    params["method"]["eps_ctcs"]                                          = eps_ctcs
    params["method"]["weights"]["w_ctcs"]                                 = params["method"]["nondim"]["nd"]**2

    ### State convergence ###
    eps_d_state             = 1e-1  # [m]
    eps_v_state             = 1e-1   # [m/s]
    params["method"]["conv"]["setup"]["eps_state"]                        = np.concatenate((eps_d_state * np.ones(params["model"]["n"] // 2), 
                                                                    eps_v_state * np.ones(params["model"]["n"] // 2)))

    params["method"]["conv"]["setup"].setdefault("state", {})["eps_d"]    = eps_d_state
    params["method"]["conv"]["setup"]["state"]["eps_v"]                   = eps_v_state

    ### Cost convergence ###
    eps_F_cost              = 1e0 # N

    # Assign to cost eps and store data
    params["method"]["conv"]["setup"]["eps_cost"] = eps_F_cost
    params["method"]["conv"]["setup"].setdefault("cost", {})["eps_v"]     = eps_F_cost

    ### NFZ convergence values ###
    eps_nfz_dim             = 1e-1 # [m]
    rc_dim = params["mission"]["obs"]["rc"] * params["method"]["nondim"]["nd"]

    eps_nfz_cnst            = 2 * rc_dim * eps_nfz_dim - eps_nfz_dim**2
    params["method"]["conv"]["setup"]["eps_nfz"]                          = eps_nfz_cnst * np.ones(params["mission"]["n_nfz"])
    params["method"]["conv"]["setup"].setdefault("cnst", {})["eps_nfz"]   = eps_nfz_cnst

    ### Terminal constraint values ###
    eps_d_term              = 1e-1 # [m]
    eps_v_term              = 1e-1 # [m/s]

    # Create eps_vector for full terminal state equality, min, max constraints
    eps_term                = np.array([eps_d_term, eps_d_term, eps_d_term, eps_v_term, eps_v_term, eps_v_term])
    eps_term_min            = eps_term.copy()
    eps_term_max            = eps_term.copy()

    # Extract only those terminal constraints used
    params["method"]["conv"]["setup"]["eps_term"]                         = np.concatenate((eps_term[params["mission"]["zf_idx"]], 
                                                                    eps_term_min[params["mission"]["zf_min_idx"]], 
                                                                    eps_term_max[params["mission"]["zf_max_idx"]]))

    # Store data
    params["method"]["conv"]["setup"].setdefault("term", {})["eps_d"]     = eps_d_term

    #### Configure multiple shooting dynamics defect convergence values ###
    params["method"]["conv"]["setup"]["eps_defect"]                       = np.array([1e-2])

    ### Dynamics convergence ###
    eps_d_dyn               = 1e-1  # [m]
    eps_v_dyn               = 1e-1   # [m/s]
    params["method"]["conv"]["setup"]["eps_dyn"]                          = np.concatenate((eps_d_dyn * np.ones(params["model"]["n"] // 2), 
                                                                    eps_v_dyn * np.ones(params["model"]["n"] // 2)))

    # Store data
    params["method"]["conv"]["setup"].setdefault("dyn", {})["eps_d"]      = eps_d_dyn
    params["method"]["conv"]["setup"]["dyn"]["eps_v"]                     = eps_v_dyn

    ### Configure generic convergence criterion and max iterations ###
    params = convergence.set_convergence_tolerance(params)

    # Iterations
    params["method"]["conv"]["iter_max"]  = 20

    # Save variable names
    params["save_var_names"]    = ["ts_opt", "zs_opt", "us_opt", "params", "O"]

    return params
