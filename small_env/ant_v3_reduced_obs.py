from gym.envs.mujoco.ant_v3 import AntEnv
from mujoco_py import functions
import numpy as np

"""
    Extending the Ant-v3 environment.
    Reduced observation space.
"""
class AntEnvReducedObs(AntEnv):
    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()
        contact_force = self.contact_forces.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observations = np.concatenate((position, velocity)) #contact_force))

        return observations