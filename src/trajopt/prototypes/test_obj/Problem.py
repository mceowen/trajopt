import Mission
import Model

# A problem has a set of parameters and many functions

class Problem:
    
    def __init__(self, name: str, model: Model, mission: Mission):
        self.model = model
        self.mission = mission

        self.model.mission = mission
