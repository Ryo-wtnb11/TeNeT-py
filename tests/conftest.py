"""Session-wide test configuration.

Enables JAX's float64 mode before any JAX array is created. This is
**process-global** and irreversible in practice, which is exactly what the numeric
tolerances here need: at float32 a central difference at ``h = 1e-5`` has no usable
digits. Any test that genuinely wants float32 semantics must say so explicitly.
"""

try:
    import jax
except ImportError:  # JAX is optional; the non-JAX suite must still run
    pass
else:
    jax.config.update("jax_enable_x64", True)
