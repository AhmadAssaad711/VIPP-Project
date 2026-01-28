import gymnasium as gym
import numpy as np
from typing import Optional

class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        self.size = size

        self._agent_location = np.array([-1, -1], dtype = np.int32)
        self._target_location = np.array([-1, -1], dtype = np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size-1, shape=(2,), dtype = int),
            }
        )

        self.action_space = gym.spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([0,1]),
            1: np.array([-1, 0]),
            2: np.array([0, -1]),
            3: np.array([1, 0]),
        }

    def _get_obs(self):
        return{
            "agent": self._agent_location,
            "target": self._target_location,
        }
    
    def _get_info(self):
        return{
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord = 1
            )

        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        ## goal is to place agent and target in random and separate locations##
        super().reset(seed=seed)

        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype = np.int32)

        self._target_location = self._agent_location.copy()
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np.random.integers(0, self.size, size=2, dtype=int
                )
        
        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):

        direction = self._action_to_direction[action]

        self._agent_to_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = np.array_equal(self._agent_location, self._target_location)

        truncated = False

        reward = 1 if terminated else 0

        observation = self._get_info()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

gym.register(
    id = "GridWorld-v0",
    entry_point = GridWorldEnv,
    max_episode_steps = 300,
)

    


        
        
