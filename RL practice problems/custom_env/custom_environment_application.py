import gymnasium as gym
import numpy as np
import custom_environment # type: ignore

env = gym.make('GridWorld-v0', size=7)
observation, info = env.reset()
print("Initial Observation:", observation)

