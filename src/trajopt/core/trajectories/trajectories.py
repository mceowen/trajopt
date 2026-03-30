import trajopt.core.trajectories.trajectory_library as trajectory_library
import inspect
from functools import partial
import trajopt.library.methods.convexify as convexify
import trajopt.utils.tools as tools

class Trajectories:
    def __init__(self, config, index_map, fcns=None):

        self.trajectories_list = []

        for i, (traj_name, traj_config_i) in enumerate(config.problem.trajectories.items()):
            self.register_trajectory(traj_config_i, index_map, fcns=fcns)

    def register_trajectory(self, traj_config, index_map, fcns=None):
        """"
        Register a trajectory object in the trajectories list given a trajectory configuration.

        Args:
            traj_config: Trajectory configuration dictionary.
            index_map: Index map object.
            fcns: Resolved functions dictionary.

        Returns:
            None.
        """
        traj_type = traj_config["type"]
        trajectoryClass = getattr(trajectory_library, traj_type)

        traj_object = trajectoryClass(traj_config, index_map, fcns=fcns)
        self.trajectories_list.append(traj_object)
        
    def get(self, **kwargs):
        """"
        Get all trajectories that match given keyword arguments.

        Args:
            **kwargs: Keyword arguments to match against trajectory attributes.

        Returns:
            List of trajectories that match the given keyword arguments.
        """
        selected_trajectories = [trajectory for trajectory in self.trajectories_list if all(getattr(trajectory, k, None) == v for k, v in kwargs.items())]
        return selected_trajectories

    def has(self, **kwargs):
        """"
        Check if any trajectories match given keyword arguments.

        Args:
            **kwargs: Keyword arguments to match against trajectory attributes.

        Returns:
            True if any trajectories match all given keyword arguments, False otherwise.
        """
        
        return any(all(getattr(trajectory, k, None) == v for k, v in kwargs.items()) for trajectory in self.trajectories_list)

    def resolve_functions(self, fcns):
        """
        Bind user-provided functions to trajectory objects and wrap 'fcns' dictionary.

        Args:
            fcns: Dictionary of user-provided functions.

        Returns:
            None.
        """
        
        for trajectory in self.trajectories_list:
            if getattr(trajectory, 'fcn_dim', None) is not None:
                sig = inspect.signature(trajectory.fcn_dim)
                param_names = sig.parameters.keys()

                kwargs_to_bind = {}
                if 'fcns' in param_names:
                    kwargs_to_bind = {"fcns": fcns}

                if kwargs_to_bind:
                    trajectory.fcn_dim = partial(trajectory.fcn_dim, **kwargs_to_bind)

                trajectory.compile_function()