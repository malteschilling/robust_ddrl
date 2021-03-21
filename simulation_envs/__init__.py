from gym.envs.registration import registry, register, make, spec
from ray.tune.registry import register_env
from gym.wrappers.time_limit import TimeLimit

# Importing the different multiagent environments.
from simulation_envs.ant_centralized_controller import AntCentralizedEnv #MultiEnv_MA
from simulation_envs.ant_decentralized_controller import AntDecentralizedEnv

register_env("AntMultiEnv_Centralized", lambda config: AntCentralizedEnv(config))
register_env("AntMultiEnv_Decentralized", lambda config: AntDecentralizedEnv(config))