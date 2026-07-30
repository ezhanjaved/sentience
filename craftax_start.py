import jax
import jax.numpy as jnp
from craftax.craftax_env import make_craftax_env_from_name
from utilities.obs_decoders import obs_decoder

rng = jax.random.PRNGKey(0)
env = make_craftax_env_from_name(
    "Craftax-Symbolic-v1",
    auto_reset=True
)
params = env.default_params
rng, reset_rng = jax.random.split(rng)

obs, state = env.reset(reset_rng, params)
obs_v1 = obs_decoder(obs)
