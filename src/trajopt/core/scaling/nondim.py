import numpy as np
from trajopt.utils.tools import AttrDict

class Nondim:
    def __init__(self, problem):

        """
        Initializes all nondimensional parameters
        """

        n_x                 = problem.index_map.n.state
        n_u                 = problem.index_map.n.get('control')

        self.state_scales   = np.ones(n_x)
        self.control_scales = np.ones(n_u)
        self.time_scale     = 1.0

        for state_group_name, state_group in problem.config.problem.state.items():

            provided_scale = state_group.get("scale", None)
            if provided_scale is None:
                print(f"Warning: no scale provided for state group '{state_group_name}', defaulting to 1.0.")
                group_scale = 1.0
            else:
                group_scale = provided_scale

            self.state_scales[state_group["idx"]] = group_scale

        for control_group_name, control_group in problem.config.problem.control.items():
            provided_scale = control_group.get("scale", None)
            
            if provided_scale is None:
                print(f"Warning: no scale provided for control group '{control_group_name}', defaulting to 1.0.")
                group_scale = 1.0
            else:
                group_scale = provided_scale

            self.control_scales[control_group["idx"]] = group_scale

        provided_scale = problem.config.problem.time.get("scale", None)
        if provided_scale is None:
            print(f"Warning: no time scale provided in 'model.nondim.t_scale', defaulting to 1.0.")
            self.time_scale = 1.0
        else:
            self.time_scale = provided_scale

        self.M              = AttrDict({})
        self.M.state        = AttrDict({})
        self.M.control      = AttrDict({})
        self.M.time         = AttrDict({})

        self.M.state.nd2d   = np.diag(self.state_scales).copy()
        self.M.state.d2nd   = np.diag(1 / self.state_scales).copy()
        self.M.control.nd2d = np.diag(self.control_scales).copy()
        self.M.control.d2nd = np.diag(1 / self.control_scales).copy()
        
        self.M.time.d2nd    = 1 / self.time_scale
        self.M.time.nd2d    = self.time_scale

        print("\n")
        print("nondim scales: ")
        print("------------------------------------------------------------")

        print(f"state scales: {self.state_scales}")
        print(f"control scales: {self.control_scales}")
        print(f"time scale: {self.time_scale}")
        print("------------------------------------------------------------")
        print("\n")

    def nondim_function(self, fcn, M_state_nd2d, M_ctrl_nd2d, M_out_d2nd):
        def wrapped_fcn(t, z, nu, params, *args, **kwargs):
            return M_out_d2nd @ fcn(t, M_state_nd2d @ z, M_ctrl_nd2d @ nu, params, *args, **kwargs)
        return wrapped_fcn
    
# old nondim:

# TODO (Carlos): revisit this for systematic, physically meaningful scaling

# # this solves the following linear system to backout base scales for
# # distance, time, and mass:
# # A @ ln([d, t, m]^T) = ln([anchor0, anchor1, anchor2]^T)
# # then ([d, t, m]^T) = exp(log([d, t, m]^T))

# exponents = AttrDict({
#     "d": np.array([1,  0,  0]),
#     "t": np.array([0,  1,  0]),
#     "m": np.array([0,  0,  1]),
#     "v": np.array([1, -1,  0]),
#     "a": np.array([1, -2,  0]),
#     "f": np.array([1, -2,  1]),
# })

# A = np.vstack([exponents[key] for key in problem.config.problem.model.nondim.anchor_types])
# b = np.log(np.array([val for val in problem.config.problem.model.nondim.anchor_scales]))

# log_base_scales = np.linalg.solve(A, b)
# base_scales = np.exp(log_base_scales)

# # retrieve remaining scales from base scales
# nd = base_scales[0]
# nt = base_scales[1]
# nm = base_scales[2]

# self.scales = AttrDict({
#     "d"    : nd,
#     "t"    : nt,
#     "m"    : nm,
#     "v"    : nd / nt,
#     "a"    : nd / (nt**2),
#     "f"    : nm * nd / (nt**2),
#     "fdot" : nm * nd / (nt**3),
#     "mom"  : nm * (nd**2) / (nt**2),
#     "momdot"  : nm * (nd**2) / (nt**3),
#     "ang"  : 180 / np.pi,
#     "angv" : (180 / np.pi) / nt,
#     "none": 1.0 
# })

# d_lbl = "m"
# t_lbl = "s"
# m_lbl = "kg"

# self.scale_labels = AttrDict({
#     "d"    : d_lbl,
#     "t"    : t_lbl,
#     "m"    : m_lbl,
#     "v"    : f"{d_lbl} / {t_lbl}" ,
#     "a"    : f"{d_lbl} / ({t_lbl}^2)",
#     "f"    : f"{m_lbl} * {d_lbl} / ({t_lbl}^2)",
#     "fdot" : f"{m_lbl} * {d_lbl} / ({t_lbl}^3)",
#     "mom"  : f"{m_lbl} * ({d_lbl}^2) / ({t_lbl}^2)",
#     "momdot"  : f"{m_lbl} * ({d_lbl}^2) / ({t_lbl}^3)",
#     "ang"  : "deg",
#     "angv" : f"deg / {t_lbl}",
#     "none": ""
# })

# print("scales: ")
# print(", ".join(f"{k}: {v:.4f}" for k, v in self.scales.items()))

# self.z_types = problem.config.problem.model.nondim.z_types
# self.u_types = problem.config.problem.model.nondim.u_types

# self.nd_state = np.array([self.scales[self.z_types[i]] for i in range(n_x)])
# self.nd_ctrl  = np.array([self.scales[self.u_types[i]] for i in range(n_nu)])