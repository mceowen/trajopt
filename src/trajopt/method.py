import numpy as np
import trajopt.algorithm.initial_guess as guess

class Method:

    def __init__(self, problem, config):

        self.problem = problem

        # ===============================================================
        # load config params
        # ===============================================================

        method_config = config["method"]
        self.method_name = method_config["method_name"]
        self.N           = method_config["N"]
        self.bools       = method_config["bools"]
        self.solver_opts = method_config["solver_opts"]
        self.T_init      = method_config["T_init"]
        self.T_min       = method_config["T_min"]
        self.T_max       = method_config["T_max"]
        self.dT_max      = method_config["dT_max"]

        self.conv        = method_config["conv"]
        self.weights     = method_config["weights"]

        self.nondim      = {}
        self.conv_data   = {}

        # ===============================================================
        # point to module and corresponding methods based on configs
        # ===============================================================

        # TODO:
        # will probably point to discretization, convergence, subprob functions etc

    def get_initial_guess(self):
        problem = self.problem
        mission = problem.mission

        if self.bools["free_final_time"] and (self.bools.get("buff_dyn")=="term"):
            us_range = np.ones((2, 1)) @ ((-mission.ge.reshape(1, -1) * mission.mass) + np.array([0.08, 0.08, 0.0])) / self.nondim["nf"]
            
            # need to manually set the left-hand side vector to a column vector for multiplacation to work
            params = guess.nonlinear_initial_guess(us_range, problem)

        else:
            params = guess.straight_line_initial_guess(params) 
            params["us_init"] =  np.tile(-mission.ge * mission.mass, (self.N, 1)) / self.nondim["nf"]

        if self.bools["ctcs"]:
            params = guess.ctcs_initial_guess(params)

        self.cost_init  = mission.cost(self.ts_init, self.zs_init, self.us_init)

    


