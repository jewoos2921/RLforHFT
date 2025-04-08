import jax
from jax import config

config.update("jax_enable_nans", True)


@jax.jit
def f(x):
    jax.debug.print("Debugging {x}", x=x)


