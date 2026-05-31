import numpy as np

from trajopt.indexing import IndexMap
from trajopt.utils.tools import AttrDict


class Nondim:
    """Nondimensionalize utility class for use in TrajectoryAnalyzer."""

    def __init__(self, config: AttrDict, index_map: IndexMap) -> None:
        """Initialize all nondimensional parameters."""
        n_x                 = index_map.n.state
        n_u                 = index_map.n.get('control')

        self.state_scales   = np.ones(n_x)
        self.control_scales = np.ones(n_u)
        self.time_scale     = 1.0

        for state_group_name, state_group in config.problem.state.items():

            provided_scale = state_group.get("scale", None)
            if provided_scale is None:
                print(f"Warning: no scale provided for state group '{state_group_name}', defaulting to 1.0.")
                group_scale = 1.0
            else:
                group_scale = provided_scale

            self.state_scales[state_group["idx"]] = group_scale

        for control_group_name, control_group in config.problem.control.items():
            provided_scale = control_group.get("scale", None)

            if provided_scale is None:
                print(f"Warning: no scale provided for control group '{control_group_name}', defaulting to 1.0.")
                group_scale = 1.0
            else:
                group_scale = provided_scale

            self.control_scales[control_group["idx"]] = group_scale

        provided_scale = config.problem.time.get("scale", None)
        if provided_scale is None:
            print("Warning: no time scale provided in 'model.nondim.t_scale', defaulting to 1.0.")
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
        def wrapped_fcn(t, x, u, params, *args, **kwargs):
            return M_out_d2nd @ fcn(t, M_state_nd2d @ x, M_ctrl_nd2d @ u, params, *args, **kwargs)
        return wrapped_fcn