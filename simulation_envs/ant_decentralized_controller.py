import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import mujoco_py
from gym import spaces

from simulation_envs.ant_multienv import AntMultiEnv

class AntDecentralizedEnv(AntMultiEnv):
    """ Derived environment for control of the four-legged agent.
        Allows to instantiate multiple agents for control.
        
        Uses four different, concurrent control units (policies) 
        each instantiated as a single agent. 
        
        Class defines 
        - policy_mapping_fn: defines names of the distributed controllers
        - distribute_observations: how to distribute observations towards these controllers
            Is defined in the obs_indices for each leg.
        - distribute_contact_cost: how to distribute (contact) costs individually to controllers 
        - concatenate_actions: how to integrate the control signals from the controllers
        
        Input scope of each controller: only the controlled leg.            
    """  
    
    # This is ordering of the policies as applied here:
    policy_names = ["policy_FL","policy_HL","policy_HR","policy_FR"]
    
    def __init__(self, config):
        self.obs_indices = {}
        # First global information: 
        # 0: height, 1-4: quaternion orientation torso
        # 5: hip FL angle, 6: knee FL angle
        # 7: hip HL angle, 8: knee HL angle
        # 9: hip HR angle, 10: knee HR angle
        # 11: hip FR angle, 12: knee FR angle
        # Velocities follow same ordering, but have in addition x and y vel.
        # 13-15: vel, 16-18: rotational velocity
        # 19: hip FL angle, 20: knee FL angle
        # 21: hip HL angle, 22: knee HL angle
        # 23: hip HR angle, 24: knee HR angle
        # 25: hip FR angle, 26: knee FR angle
        # Contact Forces:
        # 14 x 6: Main body = 0,1; FL = 2:5; HL = 5:8; HR = 8:11; FR = 11: 
        # *range( 27, 39 ) = Main body 
        # *range( 39, 57 ) = FL
        # *range( 57, 75 ) = HL
        # *range( 75, 93 ) = HR
        # *range( 93, 111 ) = FR
        self.obs_indices["policy_FL"] = [0,1,2,3,4, 5, 6,13,14,15,16,17,18,19,20,*range( 27, 39 ),*range( 39, 57 )]
        self.obs_indices["policy_HL"] = [0,1,2,3,4, 7, 8,13,14,15,16,17,18,21,22,*range( 27, 39 ),*range( 57,75 )]
        self.obs_indices["policy_HR"] = [0,1,2,3,4, 9,10,13,14,15,16,17,18,23,24,*range( 27, 39 ),*range( 75, 93 )]
        self.obs_indices["policy_FR"] = [0,1,2,3,4,11,12,13,14,15,16,17,18,25,26,*range( 27, 39 ),*range( 93,111 )]
        super().__init__(config)

    def distribute_observations(self, obs_full):
        """ 
        Construct dictionary that routes to each policy only the relevant
        local information.
        """
        obs_distributed = {}
        for policy_name in self.policy_names:
            obs_distributed[policy_name] = obs_full[self.obs_indices[policy_name],]
        return obs_distributed

    def distribute_contact_cost(self):
        contact_cost = {}
        #print("CONTACT COST")
        #from mujoco_py import functions
        #functions.mj_rnePostConstraint(self.env.model, self.env.data)
        #print("From Ant Env: ", self.env.contact_cost)
        raw_contact_forces = self.env.sim.data.cfrc_ext
        contact_forces = np.clip(raw_contact_forces, -1., 1.)
        contact_costs = self.contact_cost_weight * np.square(contact_forces)
        global_contact_costs = np.sum(contact_costs[0:2])/4.
        contact_cost[self.policy_names[0]] = global_contact_costs + np.sum(contact_costs[2:5])
        contact_cost[self.policy_names[1]] = global_contact_costs + np.sum(contact_costs[5:8])
        contact_cost[self.policy_names[2]] = global_contact_costs + np.sum(contact_costs[8:11])
        contact_cost[self.policy_names[3]] = global_contact_costs + np.sum(contact_costs[11:])
        #print(contact_cost)
        #sum_c = 0.
        #for i in self.policy_names:
         #   sum_c += contact_cost[i]
        #print("Calculated: ", sum_c)
        return contact_cost
        
    def concatenate_actions(self, action_dict):
        # Return actions in the (DIFFERENT in Mujoco) order FR - FL - HL - HR
        actions = np.concatenate( (action_dict[self.policy_names[3]],
            action_dict[self.policy_names[0]],
            action_dict[self.policy_names[1]],
            action_dict[self.policy_names[2]]) )
        return actions
        
    @staticmethod
    def policy_mapping_fn(agent_id):
        # Each derived class has to define all agents by name.
        if agent_id.startswith("policy_FL"):
            return "policy_FL"
        elif agent_id.startswith("policy_HL"):
            return "policy_HL"
        elif agent_id.startswith("policy_HR"):
            return "policy_HR"
        else:
            return "policy_FR" 
            
    @staticmethod
    def return_policies():
        # For each agent the policy interface has to be defined.
        obs_space = spaces.Box(-np.inf, np.inf, (45,), np.float64)
        policies = {
            AntDecentralizedEnv.policy_names[0]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            AntDecentralizedEnv.policy_names[1]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            AntDecentralizedEnv.policy_names[2]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
            AntDecentralizedEnv.policy_names[3]: (None,
                obs_space, spaces.Box(np.array([-1.,-1.]), np.array([+1.,+1.])), {}),
        }
        return policies

    

    

 