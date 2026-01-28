"""Policy Gradient (REINFORCE) for CartPole"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=128, action_dim=2):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, action_dim)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class PolicyGradientAgent:
    def __init__(self, learning_rate=0.01, discount_factor=0.99):
        self.discount_factor = discount_factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = PolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.reset_episode()
    
    def reset_episode(self):
        self.episode_rewards = []
        self.episode_log_probs = []
    
    def get_action(self, state, explore=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_net(state_tensor)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.episode_log_probs.append(log_prob)
        return action.item()
    
    def store_transition(self, state, action, reward):
        self.episode_rewards.append(reward)
    
    def update(self):
        if len(self.episode_rewards) == 0:
            return 0.0
        returns = []
        G = 0
        for reward in reversed(self.episode_rewards):
            G = reward + self.discount_factor * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        policy_loss = []
        for log_prob, G in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * G)
        policy_loss = torch.stack(policy_loss).sum()
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        loss_value = policy_loss.item()
        self.reset_episode()
        return loss_value

def train_policy_gradient(n_episodes=500, seed=42, verbose=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    agent = PolicyGradientAgent()
    rewards, steps, losses = [], [], []
    if verbose:
        print("\n" + "="*60)
        print("POLICY GRADIENT: Model-Free Policy-Based Learning")
        print("="*60)
    for episode in range(n_episodes):
        state, _ = env.reset()
        agent.reset_episode()
        episode_reward = 0
        episode_steps = 0
        for step in range(500):
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(state, action, reward)
            state = next_state
            episode_reward += reward
            episode_steps += 1
            if terminated or truncated:
                break
        loss = agent.update()
        rewards.append(episode_reward)
        steps.append(episode_steps)
        losses.append(loss)
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode+1}/{n_episodes} | Avg: {avg_reward:.1f}")
    env.close()
    if verbose:
        print(f"Final avg: {np.mean(rewards[-100:]):.1f}")
        print("="*60)
    return {"rewards": rewards, "steps": steps, "losses": losses, "agent": agent, "method": "Policy Gradient"}
