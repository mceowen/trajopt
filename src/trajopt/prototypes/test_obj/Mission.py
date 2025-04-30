import numpy as np

# This will change the most
class Mission:
    def __init__(self, config: object):
        self.vechicle = config['vechicle']

        self.rotation_rate_of_planet = config['rotation_rate_of_planet']
        self.radius_of_planet = config['radius_of_planet']
        self.no_fly_zones = config['no_fly_zones']

    def model_aerodynamics(self, t, z, u, model_params: object):
        # This is a placeholder for the actual implementation
        return None
    
    def custom_path_constrains(self, inputs: object):
        # Placeholder for path constraints
        pass



