from trajopt.methods.stable.algflows import AlgorithmFlow

flow = AlgorithmFlow("trajopt/methods/stable/configs/algflows/p_to_ms.yaml")

result = flow.solve()
