import jax
import numpyro

from .__about__ import __version__ as __version__

jax.config.update('jax_enable_x64', True)
numpyro.set_host_device_count(4)
