import jax
import numpyro

jax.config.update('jax_enable_x64', True)
numpyro.set_host_device_count(4)
