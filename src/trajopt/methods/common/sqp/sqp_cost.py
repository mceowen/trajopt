class SQPCost:
    def __init__(self, cost, scp_phase) -> None:
        self.cost = cost
        self.type = cost.type
        self.name = cost.name
        self._has_merit = False

    def create_cvxpy_cost(self, scp_phase): pass
    def update_cvxpy_parameters(self, scp_phase): pass
    def merit_cost(self, scp_phase): return None
    def accumulate_hessian(self, scp_phase, H): pass
