class SubproblemLinkingCost:
    def __init__(self, cost) -> None:
        self.cost = cost
        self.type = cost.type

    def build(self, subproblems): pass
    def update_cvxpy_parameters(self): pass
    def apply_step(self, alpha): pass
    def update_W_dual(self): pass
    def merit_value_and_grad(self, z_ref, dz, nu_ref, dnu): return 0.0, 0.0
    def merit_value(self, z, nu): return 0.0
