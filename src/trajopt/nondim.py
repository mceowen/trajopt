import numpy as np

from trajopt.index_map import IndexMap
from trajopt.utils.tools import AttrDict


class ScalingMatrix:
    """View that computes nd2d/d2nd diagonal scaling matrices from a scale."""

    def __init__(self, owner: "Nondim", scales_attr: str) -> None:
        self._owner       = owner
        self._scales_attr = scales_attr

    @property
    def _scales(self) -> np.ndarray:
        return np.atleast_1d(getattr(self._owner, self._scales_attr))

    @property
    def nd2d(self) -> np.ndarray:
        return np.diag(self._scales)

    @property
    def d2nd(self) -> np.ndarray:
        return np.diag(1 / self._scales)


class Nondim:
    """Nondimensionalize utility class for use in Trajectory."""

    def __init__(self, segment_config: AttrDict, index_map: IndexMap) -> None:
        """Initialize all nondimensional parameters."""
        n_x                 = index_map.n.state
        n_u                 = index_map.n.get('control')

        # initialize to one
        self.state_scales   = np.ones(n_x)
        self.control_scales = np.ones(n_u)
        self.time_scale     = 1.0

        for state_group_name, state_group in segment_config.state.items():

            provided_scale = state_group.get("scale", None)
            if provided_scale is None:
                print(f"Warning: no scale provided for state group '{state_group_name}', defaulting to 1.0.")
                group_scale = 1.0
            else:
                group_scale = provided_scale

            self.state_scales[state_group["idx"]] = group_scale

        for control_group_name, control_group in segment_config.control.items():
            provided_scale = control_group.get("scale", None)

            if provided_scale is None:
                print(f"Warning: no scale provided for control group '{control_group_name}', defaulting to 1.0.")
                group_scale = 1.0
            else:
                group_scale = provided_scale

            self.control_scales[control_group["idx"]] = group_scale

        provided_scale = segment_config.time.get("scale", None)
        if provided_scale is None:
            print("Warning: no time scale provided in 'model.nondim.t_scale', defaulting to 1.0.")
            self.time_scale = 1.0
        else:
            self.time_scale = provided_scale

        # built once; the scaling matrices are computed on access from the
        # current scales, so they can never go stale.
        self.M              = AttrDict({})
        self.M.state        = ScalingMatrix(self, "state_scales")
        self.M.control      = ScalingMatrix(self, "control_scales")
        self.M.time         = ScalingMatrix(self, "time_scale")

        print("\n")
        print("nondim scales: ")
        print("------------------------------------------------------------")

        print(f"state scales: {self.state_scales}")
        print(f"control scales: {self.control_scales}")
        print(f"time scale: {self.time_scale}")
        print("------------------------------------------------------------")
        print("\n")