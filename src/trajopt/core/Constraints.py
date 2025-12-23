import trajopt.core.modules.model.constraints_library as constraints_library

class Constraints:
    def __init__(self, constraint_config_list):

        self.constraints_list = []
        self.constraint_ids = {}
        print(f"loading constraints:")

        # build constraint_ids mapping
        for i, constraint_config in enumerate(constraint_config_list):
            constraint_type = constraint_config["type"]
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
            
            self.constraint_ids[ct_type][constraint_type].append(i)
            self.constraint_ids[ct_type]['all'].append(i)
        
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