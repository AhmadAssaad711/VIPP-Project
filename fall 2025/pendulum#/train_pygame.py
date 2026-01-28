"""
CartPole Q-Learning with Pygame Visualization

This script trains a Q-learning agent on CartPole and visualizes
the learning process in real-time using Pygame.

You'll see:
- The cart and pole animating as the agent learns
- Episode number, step count, and total reward
- Epsilon (exploration rate) decreasing over time
- Average reward over recent episodes
- The pole gradually stabilizing as learning progresses

Run this script to see the agent learn from scratch!
"""

import gymnasium as gym
import pygame
import numpy as np
import sys
from q_agent import QLearningAgent


class CartPoleVisualizer:
    """
    Real-time Pygame visualization of CartPole training.
    """
    
    def __init__(self, width=800, height=600):
        """Initialize Pygame and create display."""
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("CartPole Q-Learning - Live Training")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 50, 50)
        self.GREEN = (50, 255, 50)
        self.BLUE = (50, 150, 255)
        self.GRAY = (150, 150, 150)
        
        # Cart and pole dimensions (scaled for display)
        self.cart_width = 80
        self.cart_height = 40
        self.pole_length = 150
        self.pole_width = 8
        
        # Track position on screen
        self.track_y = height * 0.6
        self.track_left = 100
        self.track_right = width - 100
        
    def render(self, state, episode, step, total_reward, epsilon, avg_reward, q_stats):
        """
        Render the current state of the environment and training statistics.
        
        Args:
            state: Current environment state [pos, vel, angle, ang_vel]
            episode: Current episode number
            step: Current step in episode
            total_reward: Total reward accumulated this episode
            epsilon: Current exploration rate
            avg_reward: Average reward over recent episodes
            q_stats: Q-table statistics from agent
        """
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        
        # Clear screen
        self.screen.fill(self.WHITE)
        
        # Extract state variables
        cart_pos, cart_vel, pole_angle, pole_ang_vel = state
        
        # Draw track (ground)
        pygame.draw.line(
            self.screen, self.BLACK,
            (self.track_left, self.track_y),
            (self.track_right, self.track_y),
            3
        )
        
        # Map cart position (-4.8 to 4.8) to screen coordinates
        cart_x = self.track_left + (cart_pos + 4.8) / 9.6 * (self.track_right - self.track_left)
        cart_y = self.track_y
        
        # Draw cart
        cart_rect = pygame.Rect(
            int(cart_x - self.cart_width // 2),
            int(cart_y - self.cart_height),
            int(self.cart_width),
            int(self.cart_height)
        )
        pygame.draw.rect(self.screen, self.BLUE, cart_rect)
        pygame.draw.rect(self.screen, self.BLACK, cart_rect, 2)
        
        # Draw wheels
        wheel_radius = 8
        pygame.draw.circle(
            self.screen, self.BLACK,
            (int(cart_x - 20), int(cart_y + 2)),
            wheel_radius
        )
        pygame.draw.circle(
            self.screen, self.BLACK,
            (int(cart_x + 20), int(cart_y + 2)),
            wheel_radius
        )
        
        # Draw pole
        # Pole angle is relative to vertical (0 = upright)
        pole_end_x = cart_x + self.pole_length * np.sin(pole_angle)
        pole_end_y = cart_y - self.cart_height - self.pole_length * np.cos(pole_angle)
        
        # Color pole based on angle (green = good, red = bad)
        angle_severity = abs(pole_angle) / 0.418  # Normalize to [0, 1]
        pole_color = (
            int(self.RED[0] * angle_severity + self.GREEN[0] * (1 - angle_severity)),
            int(self.RED[1] * angle_severity + self.GREEN[1] * (1 - angle_severity)),
            int(self.RED[2] * angle_severity + self.GREEN[2] * (1 - angle_severity))
        )
        
        pygame.draw.line(
            self.screen, pole_color,
            (int(cart_x), int(cart_y - self.cart_height)),
            (int(pole_end_x), int(pole_end_y)),
            self.pole_width
        )
        
        # Draw joint
        pygame.draw.circle(
            self.screen, self.BLACK,
            (int(cart_x), int(cart_y - self.cart_height)),
            6
        )
        
        # Draw pole tip
        pygame.draw.circle(
            self.screen, pole_color,
            (int(pole_end_x), int(pole_end_y)),
            10
        )
        
        # Draw statistics
        y_offset = 20
        
        # Episode and step info
        text = self.font.render(f"Episode: {episode}", True, self.BLACK)
        self.screen.blit(text, (10, y_offset))
        y_offset += 40
        
        text = self.small_font.render(f"Step: {step}", True, self.BLACK)
        self.screen.blit(text, (10, y_offset))
        y_offset += 30
        
        text = self.small_font.render(f"Reward: {total_reward:.0f}", True, self.BLACK)
        self.screen.blit(text, (10, y_offset))
        y_offset += 30
        
        # Average reward
        color = self.GREEN if avg_reward > 150 else self.BLACK
        text = self.small_font.render(f"Avg Reward (100): {avg_reward:.1f}", True, color)
        self.screen.blit(text, (10, y_offset))
        y_offset += 40
        
        # Epsilon (exploration rate)
        text = self.small_font.render(f"Epsilon: {epsilon:.3f}", True, self.BLACK)
        self.screen.blit(text, (10, y_offset))
        y_offset += 30
        
        # State information
        y_offset += 10
        text = self.small_font.render("State:", True, self.GRAY)
        self.screen.blit(text, (10, y_offset))
        y_offset += 25
        
        text = self.small_font.render(f"  Pos: {cart_pos:+.3f}", True, self.BLACK)
        self.screen.blit(text, (10, y_offset))
        y_offset += 25
        
        text = self.small_font.render(f"  Vel: {cart_vel:+.3f}", True, self.BLACK)
        self.screen.blit(text, (10, y_offset))
        y_offset += 25
        
        text = self.small_font.render(f"  Angle: {pole_angle:+.3f}", True, self.BLACK)
        self.screen.blit(text, (10, y_offset))
        y_offset += 25
        
        text = self.small_font.render(f"  AngVel: {pole_ang_vel:+.3f}", True, self.BLACK)
        self.screen.blit(text, (10, y_offset))
        
        # Q-table stats (top right)
        y_offset = 20
        text = self.small_font.render("Q-Table Stats:", True, self.GRAY)
        self.screen.blit(text, (self.width - 220, y_offset))
        y_offset += 25
        
        text = self.small_font.render(f"Mean: {q_stats['q_mean']:.3f}", True, self.BLACK)
        self.screen.blit(text, (self.width - 220, y_offset))
        y_offset += 25
        
        text = self.small_font.render(f"Max: {q_stats['q_max']:.3f}", True, self.BLACK)
        self.screen.blit(text, (self.width - 220, y_offset))
        y_offset += 25
        
        text = self.small_font.render(f"Min: {q_stats['q_min']:.3f}", True, self.BLACK)
        self.screen.blit(text, (self.width - 220, y_offset))
        
        # Instructions (bottom)
        text = self.small_font.render("Press ESC to quit", True, self.GRAY)
        self.screen.blit(text, (self.width // 2 - 100, self.height - 30))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)  # 60 FPS
        
        return True
    
    def close(self):
        """Clean up Pygame."""
        pygame.quit()


def train_with_visualization(
    num_episodes=1000,
    max_steps_per_episode=500,
    render_delay=0  # Additional delay in ms per step (0 = fast as possible)
):
    """
    Train Q-learning agent with real-time Pygame visualization.
    
    Args:
        num_episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
        render_delay: Extra delay per step for slower visualization (ms)
    """
    # Create environment
    env = gym.make('CartPole-v1')
    
    # Create agent with discretized state space
    agent = QLearningAgent(
        n_bins=(6, 6, 12, 12),      # Bins for [pos, vel, angle, ang_vel]
        learning_rate=0.1,           # How fast to learn
        discount_factor=0.99,        # How much to value future rewards
        epsilon_start=1.0,           # Start with full exploration
        epsilon_min=0.01,            # Minimum exploration
        epsilon_decay=0.995          # Decay exploration each episode
    )
    
    # Create visualizer
    visualizer = CartPoleVisualizer()
    
    # Track rewards for averaging
    episode_rewards = []
    
    print("Starting Q-Learning training with visualization...")
    print("Watch the pole gradually stabilize as the agent learns!")
    print("\nControls:")
    print("  - Close window or press ESC to stop training")
    print()
    
    try:
        for episode in range(num_episodes):
            # Reset environment
            state, _ = env.reset()
            discrete_state = agent.discretize_state(state)
            
            total_reward = 0
            
            for step in range(max_steps_per_episode):
                # Choose action using epsilon-greedy
                action = agent.get_action(discrete_state, explore=True)
                
                # Take action in environment
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Discretize next state
                next_discrete_state = agent.discretize_state(next_state)
                
                # Update Q-table
                agent.update(discrete_state, action, reward, next_discrete_state, done)
                
                # Accumulate reward
                total_reward += reward
                
                # Calculate average reward over last 100 episodes
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                
                # Render visualization
                q_stats = agent.get_stats()
                should_continue = visualizer.render(
                    state=next_state,
                    episode=episode + 1,
                    step=step + 1,
                    total_reward=total_reward,
                    epsilon=agent.epsilon,
                    avg_reward=avg_reward,
                    q_stats=q_stats
                )
                
                if not should_continue:
                    print("\nTraining interrupted by user")
                    env.close()
                    visualizer.close()
                    return episode_rewards
                
                # Optional delay for slower visualization
                if render_delay > 0:
                    pygame.time.delay(render_delay)
                
                # Update state
                state = next_state
                discrete_state = next_discrete_state
                
                if done:
                    break
            
            # Episode finished
            episode_rewards.append(total_reward)
            
            # Decay exploration rate
            agent.decay_epsilon()
            
            # Print progress every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode + 1}/{num_episodes} | "
                      f"Reward: {total_reward:.0f} | "
                      f"Avg(100): {avg_reward:.1f} | "
                      f"Epsilon: {agent.epsilon:.3f}")
        
        print("\nTraining complete!")
        print(f"Final average reward (last 100 episodes): {np.mean(episode_rewards[-100:]):.1f}")
        
    finally:
        env.close()
        visualizer.close()
    
    return episode_rewards


if __name__ == "__main__":
    # Train the agent with visualization
    rewards = train_with_visualization(
        num_episodes=500,
        max_steps_per_episode=500,
        render_delay=0  # Set to 10-20 for slower, easier-to-watch visualization
    )
    
    # Keep window open briefly to show final state
    import time
    print("\nClosing in 2 seconds...")
    time.sleep(2)
