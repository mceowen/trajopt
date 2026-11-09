from trajopt.methods.stable.algflows import AlgorithmFlow

flow = AlgorithmFlow("trajopt/methods/stable/configs/algflows/dt_to_ct.yaml")

result = flow.solve()
