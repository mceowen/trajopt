import jax

from trajopt.methods.stable.algflows import AlgorithmFlow

jax.config.update("jax_enable_x64", True)

flow = AlgorithmFlow("trajopt/methods/stable/configs/algflows/p_to_ms_to_hp.yaml")

result = flow.solve()
