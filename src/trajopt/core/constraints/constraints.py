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

class Constraints:
    def __init__(self, cnstr_config_list, config):
        self.constraints_list = []
        self.constraint_ids = {
            'ct':{'all':[]},
            'nodal':{'all':[]},
            'type':{'all':[]},
            'name':{'all':[]},
            'all': {'all':[]}
        }

        for cnstr_number, cnstr_config in enumerate(cnstr_config_list):
            self.register_constraint(cnstr_number, cnstr_config, config)

    def register_constraint(self, cnstr_number, cnstr_config, config):
        cnstr_type = cnstr_config["type"]
        constraint_name = cnstr_config["name"]

        print(f"  {cnstr_number}: {constraint_name}: {cnstr_type}")
        cnstr_config = {k:v for k, v in cnstr_config.items() if k != "type"}
        constraintClass = getattr(constraints_library, cnstr_type)

        cnstr_object = constraintClass(cnstr_config=cnstr_config, config=config)
        self.constraints_list.append(cnstr_object)

        implement_type = self.constraints_list[-1].implement_type;

        # add constraint to constraint_id map for indexing into list
        if cnstr_config.get('ct', 0) == 1:
            ct_type = "ct" 
        else:
            ct_type = "nodal"

        if implement_type not in self.constraint_ids[ct_type]:
            self.constraint_ids[ct_type][implement_type] = []
        if constraint_name not in self.constraint_ids['name']:
            self.constraint_ids['name'][constraint_name] = []
        
        self.constraint_ids[ct_type][implement_type].append(cnstr_number)
        self.constraint_ids[ct_type]['all'].append(cnstr_number)
        self.constraint_ids['name'][constraint_name].append(cnstr_number)
        self.constraint_ids['all']['all'].append(cnstr_number)
        
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

    def resolve_functions(self, fcns):
        for constraint in self.constraints_list:
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

    def convexify_constraints(self):
        for constraint in self.constraints_list:
            if getattr(constraint, 'convexify_constraint', None) is not None:
                constraint.convexify_constraint()

    def augment_ctcs_dynamics(self, n):
        if self.has('ct'):
            dynamics_obj = self.get('name', 'dynamics')[0]

            f_ctcs, dfcn_dz_ctcs, dfcn_du_ctcs = convexify.linearize_jax_ctcs(dynamics_obj.fcn, self, n)
            
            lin_dyn_ctcs = lambda t, z, nu, params: (
                f_ctcs(t, z, nu, params),
                dfcn_dz_ctcs(t, z, nu, params),
                dfcn_du_ctcs(t, z, nu, params)
            )
            
            dynamics_obj.lin_dyn = lin_dyn_ctcs