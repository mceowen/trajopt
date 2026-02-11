import trajopt.core.constraints.constraints_library as constraints_library
import inspect
from functools import partial
import trajopt.library.methods.convexify as convexify
import trajopt.utils.tools as tools

class Constraints:
    def __init__(self, cnstr_config_list, config):
        self.constraints_list = []

        for i, cnstr_config in enumerate(cnstr_config_list):
            print(f"  {i}: {cnstr_config['name']}: {cnstr_config['type']}")
            self.register_constraint(cnstr_config, config)

    def register_constraint(self, cnstr_config, config):
        """"
        Regsiter a constraint object in the constraints list given a constraint configuration.

        Args:
            cnstr_config: Constraint configuration dictionary.
            config: Problem configuration dictionary.

        Returns:
            None.
        """
        cnstr_type = cnstr_config["type"]
        constraintClass = getattr(constraints_library, cnstr_type)

        cnstr_object = constraintClass(cnstr_config=cnstr_config, config=config)
        self.constraints_list.append(cnstr_object)
        
    def get(self, **kwargs):
        """"
        Get all constraints that match given keyword arguments.

        Args:
            **kwargs: Keyword arguments to match against constraint attributes.

        Returns:
            List of constraints that match the given keyword arguments.
        """
        selected_constraints = [constraint for constraint in self.constraints_list if all(getattr(constraint, k, None) == v for k, v in kwargs.items())]
        return selected_constraints

    def has(self, **kwargs):
        """"
        Check if any constraints match given keyword arguments.

        Args:
            **kwargs: Keyword arguments to match against constraint attributes.

        Returns:
            True if any constraints match all given keyword arguments, False otherwise.
        """
        
        return any(all(getattr(constraint, k, None) == v for k, v in kwargs.items()) for constraint in self.constraints_list)

    def resolve_functions(self, fcns):
        """
        Bind user-provided functions to constraint objects and wrap 'fcns' dictionary.

        Args:
            fcns: Dictionary of user-provided functions.

        Returns:
            None.
        """
        
        for constraint in self.constraints_list:
            if getattr(constraint, 'fcn_dim', None) is not None:
                sig = inspect.signature(constraint.fcn_dim)
                param_names = sig.parameters.keys()

                if 'fcns' in param_names:
                    kwargs_to_bind = {"fcns": fcns}

                if kwargs_to_bind:
                    constraint.fcn_dim = partial(constraint.fcn_dim, **kwargs_to_bind)
    
    def nondim_constraints(self, nondim):
        """
        Non-dimensionalize all constraints.

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.
        """
        
        for constraint in self.constraints_list:
            constraint.nondim_constraint(nondim)

    def convexify_constraints(self):
        """
        Convexify all constraints. If a constriant has a 'convexify_constraint' method, call it.
        Args:
            None.

        Returns:
            None.
        """
        
        for constraint in self.constraints_list:
            if getattr(constraint, 'convexify_constraint', None) is not None:
                constraint.convexify_constraint()

    def augment_ctcs_dynamics(self, n):
        """
        Augment CTCS dynamics with CTCS constraints.

        Args:
            n: Number of states.

        Returns:
            None.
        """
        
        if self.has(ct=1):
            dynamics_obj = self.get(type="dynamics")[0]

            f_ctcs, dfcn_dz_ctcs, dfcn_du_ctcs = convexify.linearize_jax_ctcs(dynamics_obj.fcn, self, n)
            
            lin_dyn_ctcs = lambda t, z, nu, params: (
                f_ctcs(t, z, nu, params),
                dfcn_dz_ctcs(t, z, nu, params),
                dfcn_du_ctcs(t, z, nu, params)
            )
            
            dynamics_obj.lin_dyn = lin_dyn_ctcs