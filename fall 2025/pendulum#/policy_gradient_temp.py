"""
Q-Learning Agent for CartPole

Model-Free, Value-Based Reinforcement Learning
==============================================

Q-Learning learns a value function Q(s,a) that estimates the expected cumulative
reward of taking action a in state s. It uses Temporal Difference (TD) learning
to update the Q-table based on observed transitions.

Key Features:
- State space discretization (continuous → discrete bins)
- Bellman optimality equation for Q-value updates
- Epsilon-greedy exploration with decay
- No model of environment dynamics required

Update Rule:
    Q[s,a] ← Q[s,a] + α [r + γ max_a' Q[s',a'] − Q[s,a]]

Assumptions:
- Markov Decision Process (MDP)
- Stationary environment
- Discrete state and action spaces (discretization required)
- Convergence requires visiting all state-action pairs infinitely often
"""

import numpy as np
import gymnasium as gym


class QLearningAgent:
    """
    Q-Learning agent with state discretization for continuous environments.
    """
    
    def __init__(self, 
                 n_bins=(6, 6, 12, 12),
                 learning_rate=0.1,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_min=0.01,
                 epsilon_decay=0.995):
        """
        Initialize Q-Learning agent.
        
        Args:
            n_bins: Tuple of bins for each state dimension (x, x_dot, theta, theta_dot)
            learning_rate: α - Step size for Q-value updates
            discount_factor: γ - Importance of future rewards
            epsilon_start: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate per episode
        """
        self.n_bins = n_bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # CartPole state bounds: [x, x_dot, theta, theta_dot]
        self.state_bounds = [
            (-4.8, 4.8),      # Cart position
            (-4.0, 4.0),      # Cart velocity
            (-0.418, 0.418),  # Pole angle (±24 degrees)
            (-4.0, 4.0)       # Pole angular velocity
        ]
        
        # Create bin edges for discretization
        self.bin_edges = []
        for i, n_bin in enumerate(n_bins):
            low, high = self.state_bounds[i]
            self.bin_edges.append(np.linspace(low, high, n_bin + 1))
        
        # Initialize Q-table: Q[state][action]
        # State shape: (n_bins[0], n_bins[1], n_bins[2], n_bins[3], 2 actions)
        q_shape = tuple(n_bins) + (2,)
        self.q_table = np.zeros(q_shape)
        
        # Track epsilon history
        self.epsilon_history = []
        
    def discretize_state(self, state):
        """
        Convert continuous state to discrete bin indices.
        
        Args:
            state: [x, x_dot, theta, theta_dot]
        
        Returns:
            Tuple of bin indices
        """
        discrete_state = []
        for i, value in enumerate(state):
            # Clip value to bounds
            clipped = np.clip(value, self.state_bounds[i][0], self.state_bounds[i][1])
            # Find bin index
            bin_idx = np.digitize(clipped, self.bin_edges[i]) - 1
            # Ensure within bounds
            bin_idx = np.clip(bin_idx, 0, self.n_bins[i] - 1)
            discrete_state.append(bin_idx)
        return tuple(discrete_state)
    
    def get_action(self, state, explore=True):
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Discrete state tuple
            explore: Whether to use epsilon-greedy (False = pure exploitation)
        
        Returns:
            Action index (0 or 1)
        """
        if explore and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, 2)
        else:
            # Exploit: best known action
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-value using TD learning (Bellman equation).
        
        Q[s,a] ← Q[s,a] + α [r + γ max_a' Q[s',a'] − Q[s,a]]
        
        Args:
            state: Current discrete state
            action: Action taken
            reward: Reward received
            next_state: Next discrete state
            done: Whether episode terminated
        """
        # Current Q-value
        current_q = self.q_table[state + (action,)]
        
        # Max Q-value for next state (0 if terminal)
        if done:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[next_state])
        
        # TD target: r + γ max_a' Q[s',a']
        td_target = reward + self.discount_factor * max_future_q
        
        # TD error: δ = target - current
        td_error = td_target - current_q
        
        # Update Q-value: Q ← Q + α δ
        self.q_table[state + (action,)] = current_q + self.learning_rate * td_error
    
    def decay_epsilon(self):
        """Decay exploration rate after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)
    
    def get_q_statistics(self):
        """Return statistics about Q-table for monitoring."""
        return {
            'mean_q': np.mean(self.q_table),
            'max_q': np.max(self.q_table),
            'min_q': np.min(self.q_table),
            'nonzero_entries': np.count_nonzero(self.q_table)
        }


def train_q_learning(n_episodes=500, seed=42, verbose=True):
    """
    Train Q-Learning agent on CartPole.
    
    Args:
        n_episodes: Number of training episodes
        seed: Random seed for reproducibility
        verbose: Whether to print progress
    
    Returns:
        dict: Training history (rewards, steps, epsilon)
    """
    # Set random seeds
    np.random.seed(seed)
    
    # Create environment
    env = gym.make('CartPole-v1')
    env.reset(seed=seed)
    
    # Initialize agent
    agent = QLearningAgent(
        n_bins=(6, 6, 12, 12),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # Training history
    rewards = []
    steps = []
    epsilons = []
    
    if verbose:
        print("\n" + "="*60)
        print("Q-LEARNING: Model-Free Value-Based Learning")
        print("="*60)
        print("\nMethod Description:")
        print("- Learns Q(s,a) value function via TD updates")
        print("- Uses discretized state space (6×6×12×12 bins)")
        print("- Epsilon-greedy exploration with decay")
        print("- Bellman optimality: Q ← Q + α[r + γ max Q' - Q]")
        print("\nTraining for {} episodes...".format(n_episodes))
    
    # Training loop
    for episode in range(n_episodes):
        state, _ = env.reset()
        discrete_state = agent.discretize_state(state)
        
        episode_reward = 0
        episode_steps = 0
        
        # Episode loop (max 500 steps)
        for step in range(500):
            # Select action
            action = agent.get_action(discrete_state, explore=True)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Discretize next state
            next_discrete_state = agent.discretize_state(next_state)
            
            # Update Q-table
            agent.update(discrete_state, action, reward, next_discrete_state, done)
            
            # Move to next state
            discrete_state = next_discrete_state
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Record metrics
        rewards.append(episode_reward)
        steps.append(episode_steps)
        epsilons.append(agent.epsilon)
        
        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    if verbose:
        final_avg = np.mean(rewards[-100:])
        success_rate = np.sum(np.array(rewards) > 200) / n_episodes * 100
        print(f"\nTraining Complete!")
        print(f"Final 100-episode average: {final_avg:.1f}")
        print(f"Success rate (>200): {success_rate:.1f}%")
        print("="*60)
    
    return {
        'rewards': rewards,
        'steps': steps,
        'epsilons': epsilons,
        'agent': agent,
        'method': 'Q-Learning'
    }


if __name__ == "__main__":
    # Test Q-Learning independently
    results = train_q_learning(n_episodes=500, seed=42)
    print(f"\nFinal average reward: {np.mean(results['rewards'][-100:]):.2f}")
