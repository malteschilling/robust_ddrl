from ray.rllib.env.multi_agent_env import MultiAgentEnv

from mujoco_py import functions
import numpy as np

from gym import spaces

import gym

"""
    Extending the Ant-v3 environment to work with Mujoco 2.
    In Mujoco_2 the contact forces are not directly calculated,
    but calculation must be explicitly invoked.
"""
class AntMultiEnv(MultiAgentEnv):

    policy_names = ["centr_A_policy"]

    def __init__(self, config):
        if 'contact_cost_weight' in config.keys():
            self.contact_cost_weight = config['contact_cost_weight']
        else: 
            self.contact_cost_weight = 5e-4
            
        if 'ctrl_cost_weight' in config.keys():
            self.ctrl_cost_weight = config['ctrl_cost_weight']
        else: 
            self.ctrl_cost_weight = 0.5 #5e-2
    
        self.env = gym.make("Ant-v3")
#            ctrl_cost_weight=self.ctrl_cost_weight,
 #           contact_cost_weight=self.contact_cost_weight)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
#        self.agents = [
 #           gym.make("Ant-v3") for _ in range(1)
  #      ]
        self.dones = set()
   #     self.observation_space = self.agents[0].observation_space
    #    self.action_space = self.agents[0].action_space

    def reset(self):
        obs_original = self.env.reset()
        return self.distribute_observations(obs_original)

    def distribute_observations(self, obs_full):
        """ Distribute observations in the multi agent environment.
        """
        return {
            self.policy_names[0]: obs_full,
        }    
    
    def distribute_contact_cost(self):
        """ Calculate contact costs and describe how to distribute them.
        """
        contact_cost = {}
        raw_contact_forces = self.env.sim.data.cfrc_ext
        contact_forces = np.clip(raw_contact_forces, -1., 1.)
        contact_cost[self.policy_names[0]] = self.contact_cost_weight * np.sum(
            np.square(contact_forces))
        return contact_cost

    def distribute_reward(self, reward_full, info, action_dict):
        """ Describe how to distribute reward.
        """
        fw_reward = info['reward_forward'] + info['reward_survive']
        rew = {}    
        contact_costs = self.distribute_contact_cost()  
        for policy_name in self.policy_names:
            rew[policy_name] = fw_reward / len(self.policy_names) \
                - self.ctrl_cost_weight * np.sum(np.square(action_dict[policy_name])) \
                - contact_costs[policy_name]
#        return {
 #           self.policy_names[0]: reward_full,
  #      }
        return rew

    def concatenate_actions(self, action_dict):
        """ Collect actions from all agents and combine them for the single 
            call of the environment.
        """
        return action_dict[self.policy_names[0]]#np.concatenate( (action_dict[self.policy_A],

    def step(self, action_dict):
        # Stepping the environment.
        
        # Use with mujoco 2.
        #functions.mj_rnePostConstraint(self.env.model, self.env.data)
        
        obs_full, rew_full, done_full, info_full = self.env.step( self.concatenate_actions(action_dict) )
        
        # Distribute observations and rewards to the individual agents.
        obs_dict = self.distribute_observations(obs_full)
        rew_dict = self.distribute_reward(rew_full, info_full, action_dict)
        
#        if done[i]:
 #           done["__all__"] = True
  #      else:
   #         done["__all__"] = False
        done_dict = {
            "__all__": done_full,
        }
        return obs_dict, rew_dict, done_dict, {}

    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        return AntMultiEnv.policy_names[0]
        
    @staticmethod
    def return_policies():
        # For each agent the policy interface has to be defined.
        policies = {
            AntMultiEnv.policy_names[0]: (None,
                spaces.Box(-np.inf, np.inf, (111,), np.float64), 
                spaces.Box(np.array([-1.,-1.,-1.,-1., -1.,-1.,-1.,-1.]), np.array([+1.,+1.,+1.,+1., +1.,+1.,+1.,+1.])), {
                }),
        }
        return policies

