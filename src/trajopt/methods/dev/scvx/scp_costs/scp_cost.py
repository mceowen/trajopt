class SCPCost:
    def __init__(self, cost, scp_segment) -> None:
        self.cost = cost
        self.type = cost.type
        self.name = cost.name

    def create_cvxpy_cost(self, scp_segment): pass
    def update_cvxpy_parameters(self, scp_segment): pass
