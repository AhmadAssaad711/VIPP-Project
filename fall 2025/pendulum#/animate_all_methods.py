"""Quick animations of all three methods"""
import pygame
import numpy as np
import gymnasium as gym
from q_learning import QLearningAgent
from policy_gradient_simple import PolicyGradientAgent
from model_based import LQRController

pygame.init()
WIDTH, HEIGHT = 900, 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.Font(None, 32)
small_font = pygame.font.Font(None, 24)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (66, 133, 244)
RED = (234, 67, 53)
GREEN = (52, 168, 83)
PURPLE = (156, 39, 176)
GRAY = (200, 200, 200)

def draw_cart_pole(screen, state, x_offset=450, y_offset=400):
    x_pos, x_vel, theta, theta_vel = state
    cart_x = int(x_offset + float(x_pos) * 100)
    cart_y = int(y_offset)
    cart_width, cart_height = 70, 35
    pole_length = 140
    
    pygame.draw.line(screen, GRAY, (50, y_offset + 18), (WIDTH - 50, y_offset + 18), 4)
    
    cart_color = BLUE
    pygame.draw.rect(screen, cart_color, 
                     (cart_x - cart_width//2, cart_y - cart_height//2, cart_width, cart_height))
    pygame.draw.rect(screen, BLACK, 
                     (cart_x - cart_width//2, cart_y - cart_height//2, cart_width, cart_height), 3)
    
    pole_end_x = int(cart_x + pole_length * np.sin(float(theta)))
    pole_end_y = int(cart_y - pole_length * np.cos(float(theta)))
    
    angle_deg = abs(float(theta) * 180 / np.pi)
    pole_color = GREEN if angle_deg < 8 else (RED if angle_deg > 15 else PURPLE)
    
    pygame.draw.line(screen, pole_color, (cart_x, cart_y), (pole_end_x, pole_end_y), 10)
    pygame.draw.circle(screen, BLACK, (cart_x, cart_y), 8)
    pygame.draw.circle(screen, pole_color, (pole_end_x, pole_end_y), 10)

def run_animation(method_name, agent, color, num_episodes=200):
    env = gym.make('CartPole-v1')
    pygame.display.set_caption(f"CartPole - {method_name}")
    
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        if hasattr(agent, 'reset_episode'):
            agent.reset_episode()
        
        episode_reward = 0
        step = 0
        
        for step in range(500):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return rewards
            
            # Get action based on agent type
            if isinstance(agent, QLearningAgent):
                action = agent.get_action(state, explore=(episode < num_episodes - 10))
            elif isinstance(agent, PolicyGradientAgent):
                action = agent.get_action(state)
            elif isinstance(agent, LQRController):
                action = agent.get_action(state)
            else:
                action = env.action_space.sample()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Update based on agent type
            if isinstance(agent, QLearningAgent):
                agent.update(state, action, reward, next_state, terminated or truncated)
            elif isinstance(agent, PolicyGradientAgent):
                agent.store_reward(reward)
            
            state = next_state
            episode_reward += reward
            
            # Render every few frames
            if step % 3 == 0:
                screen.fill(WHITE)
                
                # Title
                title = font.render(method_name, True, color)
                screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))
                
                # Stats
                avg = np.mean(rewards[-10:]) if rewards else 0
                stats = [
                    f"Episode: {episode + 1}/{num_episodes}",
                    f"Steps: {step}/500",
                    f"Current Reward: {episode_reward}",
                    f"Avg (last 10): {avg:.1f}"
                ]
                for i, stat in enumerate(stats):
                    text = small_font.render(stat, True, BLACK)
                    screen.blit(text, (20, 80 + i * 30))
                
                # Draw cart-pole
                draw_cart_pole(screen, state)
                
                # Episode progress bar
                progress = step / 500
                bar_width = 300
                pygame.draw.rect(screen, GRAY, (WIDTH - bar_width - 20, HEIGHT - 50, bar_width, 25))
                pygame.draw.rect(screen, color, (WIDTH - bar_width - 20, HEIGHT - 50, 
                                                 int(bar_width * progress), 25))
                pygame.draw.rect(screen, BLACK, (WIDTH - bar_width - 20, HEIGHT - 50, bar_width, 25), 2)
                
                pygame.display.flip()
                clock.tick(60)
            
            if terminated or truncated:
                break
        
        # Update policy for PG
        if isinstance(agent, PolicyGradientAgent):
            agent.update()
        
        rewards.append(episode_reward)
        
        # Show episode result briefly
        screen.fill(WHITE)
        title = font.render(method_name, True, color)
        screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))
        result = font.render(f"Episode {episode + 1}: {episode_reward} steps", True, color)
        screen.blit(result, (WIDTH//2 - result.get_width()//2, HEIGHT//2))
        pygame.display.flip()
        pygame.time.wait(200)
    
    env.close()
    
    # Final summary screen
    screen.fill(WHITE)
    title = font.render(f"{method_name} - Complete!", True, color)
    screen.blit(title, (WIDTH//2 - title.get_width()//2, HEIGHT//2 - 80))
    
    final_avg = np.mean(rewards[-10:])
    summary = small_font.render(f"Final Average (last 10): {final_avg:.1f}", True, BLACK)
    screen.blit(summary, (WIDTH//2 - summary.get_width()//2, HEIGHT//2 - 30))
    
    best = small_font.render(f"Best Episode: {max(rewards)}", True, BLACK)
    screen.blit(best, (WIDTH//2 - best.get_width()//2, HEIGHT//2 + 10))
    
    next_text = small_font.render("Loading next method...", True, GRAY)
    screen.blit(next_text, (WIDTH//2 - next_text.get_width()//2, HEIGHT//2 + 60))
    
    pygame.display.flip()
    pygame.time.wait(2000)
    
    return rewards

def main():
    print("\n" + "="*70)
    print("SEQUENTIAL ANIMATIONS: ALL THREE RL METHODS")
    print("="*70)
    
    # 1. Policy Gradient
    print("\n[1/3] Policy Gradient (REINFORCE) - 200 episodes")
    pg_agent = PolicyGradientAgent(learning_rate=0.01, discount_factor=0.99)
    pg_rewards = run_animation("Policy Gradient (REINFORCE)", pg_agent, PURPLE, 200)
    
    # 2. Q-Learning
    print("\n[2/3] Q-Learning (Value-Based) - 200 episodes")
    q_agent = QLearningAgent()
    q_rewards = run_animation("Q-Learning (Value-Based)", q_agent, BLUE, 200)
    
    # 3. Model-Based LQR
    print("\n[3/3] Model-Based LQR (Optimal Control) - 50 episodes")
    lqr_agent = LQRController()
    lqr_rewards = run_animation("Model-Based LQR (Optimal)", lqr_agent, RED, 50)
    
    pygame.quit()
    
    # Print summary
    print("\n" + "="*70)
    print("ANIMATION SUMMARY")
    print("="*70)
    print(f"\nPolicy Gradient:")
    print(f"  Final avg (last 10): {np.mean(pg_rewards[-10:]):.1f}")
    print(f"  Best episode: {max(pg_rewards)}")
    
    print(f"\nQ-Learning:")
    print(f"  Final avg (last 10): {np.mean(q_rewards[-10:]):.1f}")
    print(f"  Best episode: {max(q_rewards)}")
    
    print(f"\nModel-Based LQR:")
    print(f"  Final avg (last 10): {np.mean(lqr_rewards[-10:]):.1f}")
    print(f"  Best episode: {max(lqr_rewards)}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
