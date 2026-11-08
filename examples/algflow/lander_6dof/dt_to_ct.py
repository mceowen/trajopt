import jax

from trajopt.methods.stable.algflows import AlgorithmFlow

jax.config.update("jax_enable_x64", True)

flow = AlgorithmFlow("trajopt/methods/stable/algflows/configs/dt_to_ct.yaml")

result = flow.solve()
