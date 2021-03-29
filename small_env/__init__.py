from gym.envs.registration import registry, register, make, spec
from ray.tune.registry import register_env
from gym.wrappers.time_limit import TimeLimit

from small_env.ant_v3_reduced_obs import AntEnvReducedObs

# Importing the different multiagent environments.
from small_env.ant_centralized_controller import AntReducedCentralizedEnv

# Register Gym environment. 
register(
	id='AntReduced-v3',
	entry_point='small_env.ant_v3_reduced_obs:AntEnvReducedObs',
	max_episode_steps=1000,
	reward_threshold=6000.0,
)

register_env("AntMultiEnvReduced_Centralized", lambda config: AntReducedCentralizedEnv(config))