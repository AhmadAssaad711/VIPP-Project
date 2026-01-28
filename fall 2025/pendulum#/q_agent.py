"""
Q-Learning Agent for CartPole Environment

This module implements a Q-learning agent with discretized state space.
Q-learning is a model-free reinforcement learning algorithm that learns
the value of taking actions in states without requiring a model of the environment.

Key Concepts:
- State Discretization: Continuous states are bucketed into discrete bins
- Q-Table: Stores Q-values Q(s,a) for each state-action pair
- Epsilon-Greedy: Balances exploration (random actions) vs exploitation (best known action)
- Q-Update Rule: Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
"""

import numpy as np
import gymnasium as gym


class QLearningAgent:
    """
    Q-Learning agent with discretized state space for CartPole.
    
    CartPole has 4 continuous state variables:
    - Cart Position: -4.8 to 4.8
    - Cart Velocity: -Inf to Inf (we'll clip to reasonable bounds)
    - Pole Angle: -0.418 to 0.418 radians (~24 degrees)
    - Pole Angular Velocity: -Inf to Inf (we'll clip to reasonable bounds)
    
    Actions: 0 (push left) or 1 (push right)
    """
    
    def __init__(
        self,
        n_bins=(6, 6, 12, 12),  # Number of bins for each state dimension
        learning_rate=0.1,       # α: How much we update Q-values
        discount_factor=0.99,    # γ: How much we value future rewards
        epsilon_start=1.0,       # Initial exploration rate
        epsilon_min=0.01,        # Minimum exploration rate
        epsilon_decay=0.995      # Rate at which exploration decreases
    ):
        """
        Initialize the Q-learning agent.
        
        Args:
            n_bins: Tuple of bin counts for (position, velocity, angle, angular_vel)
            learning_rate: Alpha in Q-learning formula
            discount_factor: Gamma in Q-learning formula
            epsilon_start: Initial exploration probability
            epsilon_min: Minimum exploration probability
            epsilon_decay: Multiplicative decay per episode
        """
        self.n_bins = n_bins
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table: Shape (bins_pos, bins_vel, bins_angle, bins_ang_vel, 2 actions)
        self.q_table = np.zeros(n_bins + (2,))  # Initialize to zeros
        
        # Define bounds for state discretization
        # These bounds are based on CartPole's observation space and practical limits
        self.state_bounds = [
            (-4.8, 4.8),           # Cart position
            (-4.0, 4.0),           # Cart velocity (clipped)
            (-0.418, 0.418),       # Pole angle (radians, ~24 degrees)
            (-4.0, 4.0)            # Pole angular velocity (clipped)
        ]
        
        # Pre-compute bin edges for faster discretization
        self.bin_edges = [
            np.linspace(low, high, n_bins[i] + 1)
            for i, (low, high) in enumerate(self.state_bounds)
        ]
        
    def discretize_state(self, state):
        """
        Convert continuous state to discrete state indices.
        
        This is crucial for Q-learning with continuous states. We divide
        each state dimension into bins and return the bin index.
        
        Args:
            state: Continuous state from environment [pos, vel, angle, ang_vel]
            
        Returns:
            Tuple of bin indices for each state dimension
        """
        discrete_state = []
        for i, value in enumerate(state):
            # Clip value to bounds
            clipped = np.clip(value, self.state_bounds[i][0], self.state_bounds[i][1])
            # Find which bin it falls into
            bin_idx = np.digitize(clipped, self.bin_edges[i]) - 1
            # Ensure within valid range
            bin_idx = np.clip(bin_idx, 0, self.n_bins[i] - 1)
            discrete_state.append(bin_idx)
        return tuple(discrete_state)
    
    def get_action(self, state, explore=True):
        """
        Choose action using epsilon-greedy policy.
        
        Epsilon-greedy balances:
        - Exploration: Try random actions to discover new strategies
        - Exploitation: Use best known action from Q-table
        
        Args:
            state: Discrete state tuple
            explore: If False, always choose best action (for evaluation)
            
        Returns:
            Action (0 or 1)
        """
        if explore and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(0, 2)
        else:
            # Exploit: best action from Q-table
            return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state, done):
        """
        Update Q-table using Q-learning update rule.
        
        Q-Learning Formula:
        Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
        
        Where:
        - α (learning_rate): How much to update (0 = no learning, 1 = replace old value)
        - r (reward): Immediate reward received
        - γ (discount_factor): How much we value future rewards
        - max_a' Q(s',a'): Best possible future value from next state
        - Q(s,a): Current Q-value estimate
        
        Args:
            state: Current discrete state
            action: Action taken
            reward: Reward received
            next_state: Next discrete state
            done: Whether episode ended
        """
        # Current Q-value
        current_q = self.q_table[state + (action,)]
        
        # Best possible future Q-value (0 if episode ended)
        if done:
            max_future_q = 0
        else:
            max_future_q = np.max(self.q_table[next_state])
        
        # TD Target: r + γ·max_a' Q(s',a')
        td_target = reward + self.discount_factor * max_future_q
        
        # TD Error: difference between target and current estimate
        td_error = td_target - current_q
        
        # Update Q-value: Q(s,a) ← Q(s,a) + α·TD_error
        self.q_table[state + (action,)] = current_q + self.learning_rate * td_error
    
    def decay_epsilon(self):
        """
        Decay exploration rate after each episode.
        
        As training progresses, we explore less and exploit more.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_stats(self):
        """
        Get current agent statistics for monitoring.
        
        Returns:
            Dictionary with epsilon, Q-table stats, etc.
        """
        return {
            'epsilon': self.epsilon,
            'q_mean': np.mean(self.q_table),
            'q_max': np.max(self.q_table),
            'q_min': np.min(self.q_table),
            'q_std': np.std(self.q_table)
        }
