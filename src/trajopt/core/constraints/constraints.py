import trajopt.core.constraints.constraints_library as constraints_library
import inspect
from functools import partial
import trajopt.library.methods.convexify as convexify
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
#   "ct": {"control_axis_angle_cone": [0, 1], "state_max_norm_cone": [4], "all": [0, 1, 4]},
#   "nodal": {"quaternion_cone": [2, 3], "all": [2, 3]},
#   "type": {"axis_angle_cone": [0, 1], "quaternion_cone": [2, 3], "max_norm_cone": [4] "all": [0, 1, 2, 3, 4]},
#   "name": {"initial_state": [0], "final_state": [1], "all": [0, 1]}
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
    def __init__(self, constraint_config_list, config):
        self.constraints_list = []
        self.constraint_ids = {
            'ct':{'all':[]},
            'nodal':{'all':[]},
            'type':{'all':[]},
            'name':{'all':[]},
            'all': {'all':[]}
        }

        print(f"constraints:")

        for i, constraint_config in enumerate(constraint_config_list):
            constraint_type = constraint_config["type"]
            constraint_name = constraint_config["name"]

            print(f"  {i}: {constraint_name}: {constraint_type}")
            constraint_params = {k:v for k, v in constraint_config.items() if k != "type"}
            constraintClass = getattr(constraints_library, constraint_type)

            if constraint_type in GCONST_TYPES:
                self.constraints_list.append(constraintClass(ins=constraint_params, config=config))
            else:
                if 'params' in constraint_params and constraint_type == 'nonconvex_inequality':
                    constraint_params.pop('params')
                self.constraints_list.append(constraintClass(**constraint_params, config=config))

            implement_type = self.constraints_list[-1].implement_type;

            # add constraint to constraint_id map for indexing into list
            ct_type = "ct" if constraint_config.get('ct', 0) else "nodal"

            if implement_type not in self.constraint_ids[ct_type]:
                self.constraint_ids[ct_type][implement_type] = []
            if constraint_name not in self.constraint_ids['name']:
                self.constraint_ids['name'][constraint_name] = []
            
            self.constraint_ids[ct_type][implement_type].append(i)
            self.constraint_ids[ct_type]['all'].append(i)
            self.constraint_ids['name'][constraint_name].append(i)
            self.constraint_ids['all']['all'].append(i)
        
    def get(self, level1, level2=None):
        
        if level2 is not None:
            ids = self.constraint_ids.get(level1, {}).get(level2, [])
        else:
            ids = self.constraint_ids.get(level1, {}).get("all", [])

        selected_constraints = [self.constraints_list[i] for i in ids]

        return selected_constraints

    def has(self, level1, level2=None):

        if level2 is not None:
            return level2 in self.constraint_ids.get(level1, {})
        else: return len(self.constraint_ids[level1]['all'])>0

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
                if 'fcns' in param_names:
                    kwargs_to_bind['fcns'] = fcns

                if kwargs_to_bind:
                    constraint.fcn_dim = partial(constraint.fcn_dim, **kwargs_to_bind)

    def nondim_constraints(self, nondim):
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
            dynamics_obj = self.get('name', 'dynamics')[0]
            f_ctcs, dfcn_dz_ctcs, dfcn_du_ctcs = convexify.linearize_jax_ctcs(dynamics_obj.fcn, self, n)

            lin_dyn_ctcs = lambda t, z, nu, params: (f_ctcs(t, z, nu, params), dfcn_dz_ctcs(t, z, nu, params), dfcn_du_ctcs(t, z, nu, params))

            dynamics_obj.lin_dyn = lin_dyn_ctcs

            print("ctcs dynamics augmented!")