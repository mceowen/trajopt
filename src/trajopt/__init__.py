"""TrajOpt package init.

Enable JAX's persistent compilation cache so jitted kernels aren't
recompiled from scratch on every cold start. Disable with
TRAJOPT_DISABLE_JAX_CACHE=1.
"""

import os

# Silence the "Assume version compatibility. PjRt-IFRT does not
# track XLA executable versions." warning from XLA.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _enable_jax_compilation_cache() -> None:
    if os.environ.get("TRAJOPT_DISABLE_JAX_CACHE"):
        return

    import jax

    cache_dir = os.environ.get(
        "TRAJOPT_JAX_CACHE_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "trajopt", "jax"),
    )
    jax.config.update("jax_compilation_cache_dir", cache_dir)
    # Cache every executable
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)


_enable_jax_compilation_cache()
