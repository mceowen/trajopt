from trajopt.methods.stable.algflows import AlgorithmFlow

flow = AlgorithmFlow("trajopt/methods/stable/algflows/configs/dt_to_ct.yaml")

result = flow.solve()
