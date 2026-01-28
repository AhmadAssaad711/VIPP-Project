import gymnasium
import torch
from flappy import DQN
import flappy_bird_gymnasium
from pkg_resources import resource_stream, resource_exists
from importlib.resources import files


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent:

    def run(self, is_training=True, render = True):

        env = gymnasium.make('FlappyBird-v0', render_mode = 'human' if render else None, use_lidar = False)

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_states, num_actions)

        obs, _ = env.reset()

        while True:

            action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated:
                break
    
        env.close()

flappy = Agent()
flappy.run()




