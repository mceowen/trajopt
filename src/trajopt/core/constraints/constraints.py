import trajopt.core.constraints.constraints_library as constraints_library
import inspect
from functools import partial
import trajopt.core.scp.convexify as convexify
import trajopt.utils.tools as tools

# ------------------------------------------------------------
# constraints class contains a list of constraint objects with 
# a mapping of constraint type to a list of constraint ids for
# fast lookup
#
# example usage of constraints class:
# nonconvex_inequality_constraints = self.constraints.get('nodal', 'nonconvex_inequality')
# ctcs_constraints = self.constraints.get('ct', 'all')
#
# each constraint object is an instantiation from the constraints_library module
#
# example structure of constaint_ids
# (type: dict[str, dict[str, list[int]]]) 
# 
#
# constraints = [axis_angle_cone, axis_angle_cone,  quaternion_cone, quaternion_cone, max_norm_cone]
# constraint_ids = {
#   "ct": {"control_axis_angle_cone": [0, 1], "state_max_norm_cone": [4], "all"},
#   "nodal": {"quaternion_cone": [2, 3]},
#   "type": [],
#   "name": {"initial_state": [0], "final_state": [1]}
#   
# }
# ------------------------------------------------------------


# input type -- in the config file
# class type -- building constraint
# 'AFFINE','POLYTOPE','SOC',
# POLYTOPE - 'BOX','UPPER','LOWER', 
# SOC - SPHERE, ELLIPSOID, CYLINDER,...

# implementation type -- how to implement in the subproblem
# --- POLYTOPE_IN, POLYTOPE_OUT, SOC_IN, SOC_OUT, AFFINE_IN, AFFINE_OUT


GCONST_TYPES = ['AFFINE','POLYTOPE','SOC'];
GCONST_TYPES = GCONST_TYPES + ['BOX','UPPER','LOWER'];
GCONST_TYPES = GCONST_TYPES + ['ZONOTOPE'];
GCONST_TYPES = GCONST_TYPES + ['SPHERE','ELLIPSOID','CYLINDER','CONE','PROXIMITY'];


class Constraints:
    def __init__(self, constraint_config_list, params):

        self.constraints_list = []
        self.constraint_ids = {}
        print(f"constraints:")

        # build constraint_ids mapping
        for i, constraint_config in enumerate(constraint_config_list):
            constraint_type = constraint_config["type"]
            constraint_name = constraint_config["name"]
            print(f"  {i}: {constraint_name}: {constraint_type}")
            constraint_params = {k:v for k, v in constraint_config.items() if k != "type"}
            constraintClass = getattr(constraints_library,constraint_type)

            if constraint_type in GCONST_TYPES:
                self.constraints_list.append(constraintClass(ins = constraint_params,params=params))
            else: self.constraints_list.append(constraintClass(**constraint_params, params=params))

            implement_type = self.constraints_list[-1].implement_type;
            # self.constraints_list[-1].type == 'BOX';            
            # self.constraints_list[-1].specific_type == 'BOX_IN'

            # add constraint to constraint_id map for indexing into list
            ct_type = "ct" if constraint_config.get('ct', 0) else "nodal"
                
            if ct_type not in self.constraint_ids:
                self.constraint_ids[ct_type] = {}
                self.constraint_ids[ct_type]['all'] = []

            if implement_type not in self.constraint_ids[ct_type]:
                self.constraint_ids[ct_type][implement_type] = []

            if 'name' not in self.constraint_ids:
                self.constraint_ids['name'] = {}

            if constraint_name not in self.constraint_ids['name']:
                self.constraint_ids['name'][constraint_name] = []
            
            self.constraint_ids[ct_type][implement_type].append(i)
            self.constraint_ids[ct_type]['all'].append(i)
            self.constraint_ids['name'][constraint_name].append(i)
        
    def get(self, ct_type, constraint_type=None):
        
        if constraint_type is not None:
            constraint_ids = self.constraint_ids.get(ct_type, {}).get(constraint_type, [])
        else:
            constraint_ids = self.constraint_ids.get(ct_type, {}).get("all", [])

        constraints = [self.constraints_list[i] for i in constraint_ids]

        return constraints

    def has(self, ct_type, constraint_type=None):

        if constraint_type is not None:
            return constraint_type in self.constraint_ids.get(ct_type, {})
        
        else:
            return ct_type in self.constraint_ids.keys()

    def add_params(self, problem_params):

        for constraint in self.constraints_list:
            if "params" in constraint.__dict__:
                if constraint.params is not None:
                    problem_params = tools.deep_update(problem_params, constraint.params)

    def resolve_functions(self, params, fcns):
        for constraint in self.constraints_list:
            # Check fcn_dim (the raw function) since fcn may be None until nondim wrapping
            if getattr(constraint, 'fcn_dim', None) is not None:
                sig = inspect.signature(constraint.fcn_dim)
                param_names = sig.parameters.keys()

                kwargs_to_bind = {}
                if 'params' in param_names:
                    kwargs_to_bind['params'] = params
                if 'fcns' in param_names:
                    kwargs_to_bind['fcns'] = fcns

                if kwargs_to_bind:
                    constraint.fcn_dim = partial(constraint.fcn_dim, **kwargs_to_bind)

    def nondim_constraints(self, nondim):
        # apply scaling to each constraint so that they are nondim
        # by the time it gets to the discretization and solver
        for constraint in self.constraints_list:
            constraint.nondim_constraint(nondim)

        print("constraints nondimmed!")

    def convexify_constraints(self):
        for constraint in self.constraints_list:
            if getattr(constraint, 'fcn', None) is not None:
                constraint.fcn_jit, constraint.dfcn_dz_jit, constraint.dfcn_du_jit = convexify.linearize_jax(constraint.fcn)

        print("constraints convexified!")

    def augment_ctcs_dynamics(self, n):
        if self.has('ct'):
            dynamics = self.get('name', 'dynamics')[0].fcn
            f_ctcs, dfcn_dz_ctcs, dfcn_du_ctcs = convexify.linearize_jax_ctcs(dynamics, self, n)

            lin_dyn_ctcs = lambda t, z, nu: (f_ctcs(z, nu), dfcn_dz_ctcs(z, nu), dfcn_du_ctcs(z, nu))

            dynamics.lin_dyn = lin_dyn_ctcs

            print("ctcs dynamics augmented!")