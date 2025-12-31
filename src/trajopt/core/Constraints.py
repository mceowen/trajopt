import trajopt.core.models.constraints_library as constraints_library
from pprint import pprint
import inspect
from functools import partial
import trajopt.core.methods.convexify as convexify

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
#
# ------------------------------------------------------------

class Constraints:
    def __init__(self, constraint_config_list):

        self.constraints_list = []
        self.constraint_ids = {}
        print(f"constraints:")

        # build constraint_ids mapping
        for i, constraint_config in enumerate(constraint_config_list):
            constraint_type = constraint_config["type"]
            constraint_name = constraint_config["name"]
            print(f"  {i}: {constraint_type}")
            constraint_params = {k:v for k, v in constraint_config.items() if k != "type"}
            constraintClass = getattr(constraints_library, constraint_type)
            self.constraints_list.append(constraintClass(**constraint_params))

            # add constraint to constraint_id map for indexing into list
            ct_type = "ct" if constraint_config.get('ct', 0) else "nodal"
                
            if ct_type not in self.constraint_ids:
                self.constraint_ids[ct_type] = {}
                self.constraint_ids[ct_type]['all'] = []

            if constraint_type not in self.constraint_ids[ct_type]:
                self.constraint_ids[ct_type][constraint_type] = []

            if 'name' not in self.constraint_ids:
                self.constraint_ids['name'] = {}

            if constraint_name not in self.constraint_ids['name']:
                self.constraint_ids['name'][constraint_name] = []

            
            self.constraint_ids[ct_type][constraint_type].append(i)
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

    def resolve_functions(self, params, fcns):
        for constraint in self.constraints_list:
            if getattr(constraint, 'fcn', None) is not None:
                sig = inspect.signature(constraint.fcn)
                param_names = sig.parameters.keys()

                kwargs_to_bind = {}
                if 'params' in param_names:
                    kwargs_to_bind['params'] = params
                if 'fcns' in param_names:
                    kwargs_to_bind['fcns'] = fcns

                constraint.fcn = partial(constraint.fcn, **kwargs_to_bind)

                constraint.fcn_jit, constraint.dfcn_dz_jit, constraint.dfcn_du_jit = convexify.linearize_jax(constraint.fcn)