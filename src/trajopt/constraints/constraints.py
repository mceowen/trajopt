import inspect
from collections.abc import Callable
from functools import partial
from typing import Any
import jax.numpy as jnp
from trajopt.constraints import constraint_types


class Constraints:
    def __init__(self, config: Any, index_map: Any, fcns: dict | None = None) -> None:
        """Initialize constraints from problem configuration.

        Args:
            config: Problem configuration object.
            index_map: Index map object.
            fcns: Resolved functions dictionary.

        """
        print("constraints:")

        self.index_map = index_map
        self.constraint_list = []
        self.constraint_type_list = []
        self.params = config.problem.params
        self.ctcs_flag = config.method.get("flags", {}).get("ctcs", 0)

        for i, (cnstr_name, cnstr_config_i) in enumerate(config.problem.constraints.items()):
            cnstr_config_i["name"] = cnstr_name
            print(f"  {i}: {cnstr_name}: type: {cnstr_config_i.type}")
            self.register_constraint(cnstr_config_i, index_map, fcns=fcns)

    def register_constraint(self, cnstr_config: Any, index_map: Any, fcns: dict | None = None) -> None:
        """Register a constraint object in the constraints list given a constraint configuration.

        Args:
            cnstr_config: Constraint configuration dictionary.
            index_map: Index map object.
            fcns: Resolved functions dictionary.

        Returns:
            None.

        """
        cnstr_type = cnstr_config["type"]
        constraintClass = getattr(constraint_types, cnstr_type)

        cnstr_object = constraintClass(cnstr_config, index_map, fcns=fcns, params=self.params)
        self.constraint_list.append(cnstr_object)
        if cnstr_type not in self.constraint_type_list:
            self.constraint_type_list.append(cnstr_type)

    def get(self, **kwargs: Any) -> list:
        """Get all constraints that match given keyword arguments.

        Args:
            **kwargs: Keyword arguments to match against constraint attributes.

        Returns:
            List of constraints that match the given keyword arguments.

        """
        selected_constraints = [
            constraint
            for constraint in self.constraint_list
            if all(getattr(constraint, k, None) == v for k, v in kwargs.items())
        ]

        return selected_constraints

    def has(self, **kwargs: Any) -> bool:
        """Check if any constraints match given keyword arguments.

        Args:
            **kwargs: Keyword arguments to match against constraint attributes.

        Returns:
            True if any constraints match all given keyword arguments, False otherwise.

        """
        return any(
            all(getattr(constraint, k, None) == v for k, v in kwargs.items()) for constraint in self.constraint_list
        )

    def resolve_functions(self, fcns: dict) -> None:
        """Bind user-provided functions to constraint objects and wrap 'fcns' dictionary.

        Args:
            fcns: Dictionary of user-provided functions.

        Returns:
            None.

        """
        for constraint in self.constraint_list:
            if getattr(constraint, "fcn_dim", None) is not None:
                sig = inspect.signature(constraint.fcn_dim)
                param_names = sig.parameters.keys()

                kwargs_to_bind = {}
                if "fcns" in param_names:
                    kwargs_to_bind = {"fcns": fcns}

                if kwargs_to_bind:
                    constraint.fcn_dim = partial(constraint.fcn_dim, **kwargs_to_bind)

    def nondim_constraints(self, nondim: Any) -> None:
        """Non-dimensionalize all constraints.

        Args:
            nondim: Non-dimensionalization object.

        Returns:
            None.

        """
        for constraint in self.constraint_list:
            constraint.nondim_constraint(nondim)

    def augment(self) -> None:
        """Finalize the augmented z layout and dynamics once every constraint knows its dimension.

        Returns:
            None.

        """
        index_map = self.index_map

        # method-level ctcs flag promotes all nonconvex inequalities to continuous-time constraints
        if self.ctcs_flag:
            for constraint in self.get(type="nonconvex_inequality"):
                constraint.ct = 1

        # the augmented state grows by the total dimension of the continuous-time (ct) constraints
        n_ctcs = sum(constraint.dimension for constraint in self.constraint_list if getattr(constraint, "ct", 0) == 1)
        index_map.set_augmented_dims(n_ctcs)

        # wrap every t-x-u constraint's non-dimensional function into augmented-vector form fcn_znu(z, nu, params)
        for constraint in self.constraint_list:
            if getattr(constraint, "augment_constraint", None) is not None:
                constraint.augment_constraint()

        # the dynamics defect spans the full augmented state z = [x, t, beta]
        dynamics = self.get(type="dynamics")[0]
        dynamics.dimension = index_map.n.z
        dynamics.fcn = self.augment_dynamics(dynamics.fcn_znu)

        # per-type sizes used to build the penalty / dual / virtual-buffer stacks
        for constraint in self.constraint_list:
            index_map.n[constraint.type] = index_map.n.get(constraint.type, 0) + constraint.dimension
            index_map.N[constraint.type] = len(constraint.nodes)

    # TODO(Skye): Verify nondim (specifically time)
    # Generalize dynamics augmentation so users can add arbitrary states/controls

    def augment_dynamics(self, f_znu: Callable) -> Callable:
        """Build augmented dynamics zdot = [xdot_tau, s, dbeta_dtau].

        Args:
            f_znu: Physical dynamics in augmented-vector form f(z, nu, params).

        Returns:
            Augmented dynamics function dynamics_z_nu(z, nu, params).

        """

        def dynamics_z_nu(z: Any, nu: Any, params: Any) -> Any:
            x, t, beta = self.index_map.unpack_z(z)
            u, s = self.index_map.unpack_nu(nu)

            dx_dt = f_znu(z, nu, params)

            dt_dt = jnp.asarray([1.0], dtype=z.dtype)

            ctcs_constraints = tuple(self.get(ct=1))
            if ctcs_constraints:
                ctcs_values = jnp.concatenate(
                    [jnp.atleast_1d(constraint.fcn_znu(z, nu, params)) for constraint in ctcs_constraints],
                )

                dbeta_dt = jnp.maximum(ctcs_values, 0.0)
            else:
                dbeta_dt = jnp.zeros_like(beta)

            return s * jnp.concatenate([dx_dt, dt_dt, dbeta_dt])

        return dynamics_z_nu
