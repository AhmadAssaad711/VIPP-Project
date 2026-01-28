"""
Evaluate a trained Q-learning agent without visualization.

This script demonstrates how to:
1. Load or train a Q-learning agent
2. Evaluate its performance without exploration
3. Calculate statistics over multiple evaluation episodes

Useful for benchmarking and testing.
"""

import gymnasium as gym
import numpy as np
from q_agent import QLearningAgent
import pickle


def train_agent(episodes=500, verbose=True):
    """
    Train a Q-learning agent from scratch.
    
    Args:
        episodes: Number of training episodes
        verbose: Whether to print training progress
        
    Returns:
        Trained QLearningAgent
    """
    env = gym.make('CartPole-v1')
    agent = QLearningAgent(
        n_bins=(6, 6, 12, 12),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    if verbose:
        print(f"Training agent for {episodes} episodes...")
    
    episode_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        discrete_state = agent.discretize_state(state)
        total_reward = 0
        
        for step in range(500):
            action = agent.get_action(discrete_state, explore=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            next_discrete_state = agent.discretize_state(next_state)
            agent.update(discrete_state, action, reward, next_discrete_state, done)
            
            total_reward += reward
            discrete_state = next_discrete_state
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        agent.decay_epsilon()
        
        if verbose and (episode + 1) % 50 == 0:
            avg = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Avg Reward: {avg:.1f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    env.close()
    
    if verbose:
        final_avg = np.mean(episode_rewards[-100:])
        print(f"\nTraining complete! Final average: {final_avg:.1f}")
    
    return agent


def evaluate_agent(agent, num_episodes=100, render=False, verbose=True):
    """
    Evaluate a trained agent without exploration.
    
    Args:
        agent: Trained QLearningAgent
        num_episodes: Number of evaluation episodes
        render: Whether to render the environment (slow)
        verbose: Whether to print results
        
    Returns:
        Dictionary with evaluation statistics
    """
    render_mode = 'human' if render else None
    env = gym.make('CartPole-v1', render_mode=render_mode)
    
    if verbose:
        print(f"\nEvaluating agent for {num_episodes} episodes (no exploration)...")
    
    rewards = []
    steps_list = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        discrete_state = agent.discretize_state(state)
        total_reward = 0
        steps = 0
        
        for step in range(500):
            # No exploration during evaluation
            action = agent.get_action(discrete_state, explore=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            discrete_state = agent.discretize_state(next_state)
            
            if done:
                break
        
        rewards.append(total_reward)
        steps_list.append(steps)
    
    env.close()
    
    # Calculate statistics
    stats = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'median_reward': np.median(rewards),
        'mean_steps': np.mean(steps_list),
        'success_rate': np.mean([r >= 195 for r in rewards]) * 100  # 195 = "solved"
    }
    
    if verbose:
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Mean Reward:    {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"Median Reward:  {stats['median_reward']:.2f}")
        print(f"Min Reward:     {stats['min_reward']:.2f}")
        print(f"Max Reward:     {stats['max_reward']:.2f}")
        print(f"Mean Steps:     {stats['mean_steps']:.2f}")
        print(f"Success Rate:   {stats['success_rate']:.1f}% (reward >= 195)")
        print("="*60)
    
    return stats


def save_agent(agent, filename='trained_agent.pkl'):
    """
    Save a trained agent to disk.
    
    Args:
        agent: QLearningAgent to save
        filename: Path to save file
    """
    with open(filename, 'wb') as f:
        pickle.dump(agent, f)
    print(f"Agent saved to {filename}")


def load_agent(filename='trained_agent.pkl'):
    """
    Load a trained agent from disk.
    
    Args:
        filename: Path to saved agent file
        
    Returns:
        Loaded QLearningAgent
    """
    with open(filename, 'rb') as f:
        agent = pickle.load(f)
    print(f"Agent loaded from {filename}")
    return agent


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Q-Learning Agent Training & Evaluation")
    print("="*60 + "\n")
    
    # Train agent
    agent = train_agent(episodes=500, verbose=True)
    
    # Evaluate agent
    stats = evaluate_agent(agent, num_episodes=100, render=False, verbose=True)
    
    # Optional: Save agent for later use
    save_agent(agent, 'trained_cartpole_agent.pkl')
    
    # Optional: Load and re-evaluate
    # loaded_agent = load_agent('trained_cartpole_agent.pkl')
    # evaluate_agent(loaded_agent, num_episodes=10, render=True)
    
    print("\nDone!")
