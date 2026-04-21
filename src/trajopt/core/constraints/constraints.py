import trajopt.core.constraints.constraints_library as constraints_library
import inspect
import jax.numpy as jnp
from functools import partial

class Constraints:
    def __init__(self, config, index_map, fcns=None):

        print("constraints:")

        self.index_map = index_map
        self.constraints_list = []

        for i, (cnstr_name, cnstr_config_i) in enumerate(config.problem.constraints.items()):
            print(f"  {i}: {cnstr_name}: type: {cnstr_config_i.type}")
            self.register_constraint(cnstr_config_i, index_map, fcns=fcns)

    def register_constraint(self, cnstr_config, index_map, fcns=None):
        """"
        Regsiter a constraint object in the constraints list given a constraint configuration.

        Args:
            cnstr_config: Constraint configuration dictionary.
            index_map: Index map object.
            fcns: Resolved functions dictionary.

        Returns:
            None.
        """
        cnstr_type      = cnstr_config["type"]
        constraintClass = getattr(constraints_library, cnstr_type)

        cnstr_object = constraintClass(cnstr_config, index_map, fcns=fcns)
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

                kwargs_to_bind = {}
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

    # TODO(Skye): Verify nondim (specifically time)
    # Move this to subproblem constraints
    # Generalize dynamics augmentation so users can add arbitrary states/controls
    
    def augment_dynamics_jax(self, f_phys):
        """
        Build augmented dynamics zdot = [xdot_tau, s, dbeta_dtau].
        """
        # generate f(z, nu) function given f(t, x, u)
        def dynamics_z_nu(z, nu, params):
            x, t, beta  = self.index_map.unpack_z(z)
            u, s        = self.index_map.unpack_nu(nu)

            dx_dt       = self.index_map.evaluate_f_phys(f_phys, z, nu, params)

            dt_dt       = jnp.asarray([1.0], dtype=z.dtype)

            ctcs_constraints = tuple(self.get(ct=1))
            if ctcs_constraints:
                ctcs_values = jnp.concatenate([
                    jnp.atleast_1d(constraint.fcn(z, nu, params))
                    for constraint in ctcs_constraints
                ])

                dbeta_dt = jnp.maximum(ctcs_values, 0.0)
            else:
                dbeta_dt = jnp.zeros_like(beta)

            return s * jnp.concatenate([dx_dt, dt_dt, dbeta_dt])

        return dynamics_z_nu
    
    def augment_txu_to_znu(self, fcn):

        def fcn_znu(z, nu, params):
            x, t, beta  = self.index_map.unpack_z(z)
            u, s        = self.index_map.unpack_nu(nu)

            return fcn(t, x, u, params)

        return fcn_znu