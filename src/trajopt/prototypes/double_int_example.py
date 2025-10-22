# basic imports
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import time

# trajopt imports
import trajopt.problem_models.double_integrator_2dof    as double_int
import trajopt.algorithm.subproblem                     as subproblem

# Step 1: Define configuration
config = double_int.config_main()

# Step 2: Create the full problem dictionary using ocp()
problem = double_int.ocp(config)

# Step 3: Inject the first SCvx iteration into problem['I']
N = problem['params']['N']

problem["I"] = [{
    "iter_num": 1,
    "zs_ref": problem["params"]["zs_init"],
    "us_ref": problem["params"]["us_init"],
    "dts_ref": np.full(N - 1, problem["params"]["T_init"] / (N - 1)),
    "ts_ref": np.linspace(0, problem["params"]["T_init"], N),
    "conv_data": {
        "vb_path": np.zeros((problem["params"]["n_path"], N)),
        "vb_nfz": np.zeros((problem["params"]["mission"]["n_nfz"], N)),
        "vb_aux": np.zeros((problem["params"].get("n_aux", 0), N)),
        "vb_dyn": np.zeros((problem["params"]["model"]["nz"], N - 1)),
        "vb_term": np.zeros((problem["params"]["mission"]["n_term"] + problem["params"]["mission"]["n_term_ineq"], 1)),
    },
    "weights": problem["params"]["method"]["weights"]
}]
problem["O"] = []

print("-" * 152)
print(f"                                              ..:: {problem['name']}: PTR with Virtual Buffer ::..")
print("-" * 152)
print("  Iteration |  Propagation |   Solve   |    Parse   |  log(dz)  |      log(VB)    |   log(VB)   |  log(VB)    | Solve status |  Time of    |   Cost    ")
print("            |   time [ms]  | time [ms] |  time [ms] |           |  (path + NFZ)   |  (terminal) |  (dynamics) |              |  Flight [s] |           ")
print("-" * 152)

for ii in range( problem['params']['conv']['iter_max']+1 ):

    output = subproblem.solve_subproblem( problem )

    problem["O"].append(output)
    problem["O"][ii]["iter_num"]    = ii + 1

    if problem["O"][-1]["converged"]:
        break

    problem["I"].append(problem["I"][ii]) 
    problem["I"][ii+1]["iter_num"]  = ii + 2
    problem["I"][ii+1]["ts_ref"]    = problem["O"][ii]["ts"]
    problem["I"][ii+1]["dts_ref"]   = problem["O"][ii]["dts"]
    problem["I"][ii+1]["zs_ref"]    = problem["O"][ii]["zs"]
    problem["I"][ii+1]["us_ref"]    = problem["O"][ii]["us"]
    problem["I"][ii+1]["Ts_ref"]    = problem["O"][ii]["Ts"]
    problem["I"][ii+1]["method"]["weights"]   = problem["O"][ii]["method"]["weights"]
    problem["I"][ii+1]["conv_data"] = problem["O"][ii]["conv_data"]


# # # STORING DATA 
import trajopt.prototypes.test_plot.store_data as store_data
store_data.dump_filtered_dict(problem, '~/ACL/entry/python/scp_sandbox/trajopt/src/trajopt/prototypes/test_plot/Problem.json')

# make plots 
fig     = plt.figure()
ax      = fig.add_subplot(111, projection="3d")
ax.plot(problem["I"][-1]["zs_ref"][0, :], problem["I"][-1]["zs_ref"][1, :], problem["I"][-1]["zs_ref"][2, :])

fig     = plt.figure()
ax_1    = fig.add_subplot(311)
ax_2    = fig.add_subplot(312)
ax_3    = fig.add_subplot(313)
ax_1.plot(problem["I"][-1]["ts_ref"], problem["I"][-1]["us_ref"][0, :])
ax_2.plot(problem["I"][-1]["ts_ref"], problem["I"][-1]["us_ref"][1, :])
ax_3.plot(problem["I"][-1]["ts_ref"], problem["I"][-1]["us_ref"][2, :])
plt.show()

