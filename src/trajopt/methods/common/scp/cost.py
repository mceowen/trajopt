class SCPCost:
    def __init__(self, cost, scp_subproblem) -> None:
        self.cost = cost
        self.type = cost.type
        self.name = cost.name

    def create_cvxpy_cost(self, scp_subproblem): pass
    def update_cvxpy_parameters(self, scp_subproblem): pass
