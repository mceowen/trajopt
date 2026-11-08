from trajopt.methods.stable.algflows import AlgorithmFlow

flow = AlgorithmFlow("p_to_ms_to_adaptive_hp.yaml")

result = flow.solve()
