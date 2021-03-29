import numpy as np
import gym
from gym import spaces

import ray
import ray.rllib.agents.ppo as ppo
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print
from ray import tune
from ray.tune import grid_search
import time

import small_env
import models

import argparse

from small_env.ant_centralized_controller import AntReducedCentralizedEnv

policy_scope = "AntMultiEnvReduced_Centralized"
 
from small_env.ant_centralized_controller import AntReducedCentralizedEnv as AntEnv

# Init ray: First line on server, second for laptop
#ray.init(num_cpus=30, ignore_reinit_error=True)
ray.init(ignore_reinit_error=True)

config = ppo.DEFAULT_CONFIG.copy()

config['env'] = policy_scope #"multima" #"Ant_Muj2-v4" #policy_scope

config['num_workers']=2
config['num_envs_per_worker']=4

config['train_batch_size'] = 8000
#config['rollout_fragment_length'] = 1000

config['gamma'] = 0.99
config['lambda'] = 0.95 

#NEW from baselines
config['vf_loss_coeff']=0.5

config['sgd_minibatch_size'] = 64
config['num_sgd_iter'] = 10

config['clip_param'] = 0.2

config['lr'] = 3e-4

#config['model']['custom_model'] = "fc_glorot_uniform_init"
config['model']['fcnet_hiddens'] = grid_search([[16,16], [64, 64], [256,256]])

config['grad_clip']=0.5
config['observation_filter'] = 'MeanStdFilter'

print("SELECTED ENVIRONMENT: ", policy_scope, " = ", AntEnv)

# For running tune, we have to provide information on 
# the multiagent which are part of the MultiEnvs
#policies = AntEnv.return_policies( spaces.Box(-np.inf, np.inf, (111,), np.float64) )

policies = AntEnv.return_policies()
config["multiagent"] = {
            "policies": policies,
            "policy_mapping_fn": AntEnv.policy_mapping_fn,
            "policies_to_train": AntEnv.policy_names,
    }

#config['env_config']['ctrl_cost_weight'] = 0.5#grid_search([5e-4,5e-3,5e-2])
#config['env_config']['contact_cost_weight'] =  5e-4 #grid_search([5e-4,5e-3,5e-2])

# Parameters for defining environment:
# Heightfield smoothness (between 0.6 and 1.0 are OK)
#config['env_config']['hf_smoothness'] = 1.0
# Defining curriculum learning
#config['env_config']['curriculum_learning'] =  False
#config['env_config']['range_smoothness'] =  [1., 0.6]
#config['env_config']['range_last_timestep'] =  10000000

# For curriculum learning: environment has to be updated every epoch
#def on_train_result(info):
 #   result = info["result"]
  #  trainer = info["trainer"]
   # timesteps_res = result["timesteps_total"]
    #trainer.workers.foreach_worker(
     #   lambda ev: ev.foreach_env( lambda env: env.update_environment_after_epoch( timesteps_res ) )) 
#config["callbacks"]={"on_train_result": on_train_result,}

# Call tune and run (for evaluation: 10 seeds up to 20M steps; only centralized controller
# required that much of time; decentralized controller should show very good results 
# after 5M steps.
analysis = tune.run(
      "PPO",
      name=("AntReduced_Robustness/" + policy_scope),
      num_samples=10,
      checkpoint_at_end=True,
      checkpoint_freq=250,
      stop={"timesteps_total": 10000000},
      config=config,
  )
