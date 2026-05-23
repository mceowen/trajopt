import inspect
from functools import partial

from trajopt.indexing import IndexMap
from trajopt.trajectories import trajectory_types


class Trajectories:
    """Manage a collection of trajectory objects constructed from configuration."""

    def __init__(self, config, index_map: IndexMap, fcns=None):
        """Initialize trajectory collection from config.

        Args:
            config: Problem configuration containing trajectory definitions.
            index_map: Index map object.
            fcns: Resolved functions dictionary.

        """
        self.trajectories_list = []
        self.trajectory_type_list = []

        for i, (traj_name, traj_config_i) in enumerate(config.problem.trajectories.items()):
            self.register_trajectory(traj_config_i, index_map, fcns=fcns)

    def register_trajectory(self, traj_config, index_map, fcns=None):
        """"Register a trajectory object in the trajectories list given a trajectory configuration.

        Args:
            traj_config: Trajectory configuration dictionary.
            index_map: Index map object.
            fcns: Resolved functions dictionary.

        Returns:
            None.

        """
        traj_type = traj_config["type"]
        trajectoryClass = getattr(trajectory_types, traj_type)

        traj_object = trajectoryClass(traj_config, index_map, fcns=fcns)
        self.trajectories_list.append(traj_object)
        if traj_type not in self.trajectory_type_list:
            self.trajectory_type_list.append(traj_type)

    def get(self, **kwargs):
        """"Get all trajectories that match given keyword arguments.

        Args:
            **kwargs: Keyword arguments to match against trajectory attributes.

        Returns:
            List of trajectories that match the given keyword arguments.

        """
        return [trajectory for trajectory in self.trajectories_list if all(getattr(trajectory, k, None) == v for k, v in kwargs.items())]

    def has(self, **kwargs):
        """"Check if any trajectories match given keyword arguments.

        Args:
            **kwargs: Keyword arguments to match against trajectory attributes.

        Returns:
            True if any trajectories match all given keyword arguments, False otherwise.

        """
        return any(all(getattr(trajectory, k, None) == v for k, v in kwargs.items()) for trajectory in self.trajectories_list)

    def resolve_functions(self, fcns):
        """Bind user-provided functions to trajectory objects and wrap 'fcns' dictionary.

        Args:
            fcns: Dictionary of user-provided functions.

        Returns:
            None.

        """
        for trajectory in self.trajectories_list:
            if getattr(trajectory, 'fcn_dim', None) is not None:
                sig = inspect.signature(trajectory.fcn_dim)
                if 'fcns' in sig.parameters:
                    trajectory.fcn_dim = partial(trajectory.fcn_dim, fcns=fcns)

            for i, qfcn in enumerate(getattr(trajectory, 'quiver_fcn_dims', [])):
                sig = inspect.signature(qfcn)
                if 'fcns' in sig.parameters:
                    trajectory.quiver_fcn_dims[i] = partial(qfcn, fcns=fcns)

            for i, ofcn in enumerate(getattr(trajectory, 'quiver_origin_fcn_dims', [])):
                if ofcn is not None:
                    sig = inspect.signature(ofcn)
                    if 'fcns' in sig.parameters:
                        trajectory.quiver_origin_fcn_dims[i] = partial(ofcn, fcns=fcns)

            trajectory.compile_function()
