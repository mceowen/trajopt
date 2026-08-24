class SCPCost:
    def __init__(self, cost, scp_segment) -> None:
        self.cost = cost
        self.type = cost.type
        self.name = cost.name
        self._has_merit = False

    def create_cvxpy_cost(self, scp_segment): pass
    def update_cvxpy_parameters(self, scp_segment): pass
    def merit_cost(self, scp_segment): return None
    def accumulate_hessian(self, scp_segment, H): pass
