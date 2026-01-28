"""Simple Policy Gradient (REINFORCE) using NumPy - No PyTorch dependency"""
import numpy as np
import gymnasium as gym

class SimplePolicyNetwork:
    """Simple 2-layer neural network using only NumPy"""
    def __init__(self, state_dim=4, hidden_dim=64, action_dim=2):
        # Xavier initialization
        self.W1 = np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, action_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(action_dim)
        
    def forward(self, state):
        """Forward pass: state -> action probabilities"""
        z1 = state @ self.W1 + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = a1 @ self.W2 + self.b2
        # Softmax
        exp_z = np.exp(z2 - np.max(z2))  # Numerical stability
        probs = exp_z / np.sum(exp_z)
        return probs, a1, z1
    
    def backward(self, state, action, a1, z1, grad_probs, lr=0.001):
        """Backward pass: update weights using gradient"""
        # Output layer gradients
        dW2 = np.outer(a1, grad_probs)
        db2 = grad_probs
        
        # Hidden layer gradients
        grad_a1 = grad_probs @ self.W2.T
        grad_z1 = grad_a1 * (z1 > 0)  # ReLU derivative
        dW1 = np.outer(state, grad_z1)
        db1 = grad_z1
        
        # Update weights
        self.W1 += lr * dW1
        self.b1 += lr * db1
        self.W2 += lr * dW2
        self.b2 += lr * db2

class PolicyGradientAgent:
    def __init__(self, learning_rate=0.001, discount_factor=0.99):
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.policy = SimplePolicyNetwork()
        self.reset_episode()
    
    def reset_episode(self):
        self.episode_data = []  # Store (state, action, a1, z1, prob)
        self.episode_rewards = []
    
    def get_action(self, state):
        """Sample action from policy"""
        probs, a1, z1 = self.policy.forward(state)
        action = np.random.choice(2, p=probs)
        self.episode_data.append((state, action, a1, z1, probs[action]))
        return action
    
    def store_reward(self, reward):
        self.episode_rewards.append(reward)
    
    def update(self):
        """REINFORCE update at end of episode"""
        if len(self.episode_rewards) == 0:
            return 0.0
        
        # Compute discounted returns
        returns = []
        G = 0
        for r in reversed(self.episode_rewards):
            G = r + self.discount_factor * G
            returns.insert(0, G)
        
        # Normalize returns (reduces variance)
        returns = np.array(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Update policy for each timestep
        total_loss = 0
        for t, (state, action, a1, z1, prob) in enumerate(self.episode_data):
            # Compute gradient
            probs, _, _ = self.policy.forward(state)
            
            # Gradient of log probability
            grad_log_prob = np.zeros(2)
            grad_log_prob[action] = 1.0 / (probs[action] + 1e-8)
            
            # Policy gradient: grad = (probs - one_hot) for softmax
            grad_probs = probs.copy()
            grad_probs[action] -= 1
            
            # Scale by return (REINFORCE)
            grad_probs *= -returns[t]  # Negative for gradient ascent
            
            # Backprop and update
            self.policy.backward(state, action, a1, z1, grad_probs, self.learning_rate)
            
            total_loss += -np.log(prob + 1e-8) * returns[t]
        
        loss = total_loss / len(self.episode_data)
        self.reset_episode()
        return loss

def train_policy_gradient(n_episodes=500, seed=42, verbose=True):
    np.random.seed(seed)
    env = gym.make('CartPole-v1')
    env.reset(seed=seed)
    
    agent = PolicyGradientAgent(learning_rate=0.005, discount_factor=0.99)
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
            agent.store_reward(reward)
            
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
    
    return {
        'rewards': rewards,
        'steps': steps,
        'losses': losses,
        'agent': agent,
        'method': 'Policy Gradient'
    }
