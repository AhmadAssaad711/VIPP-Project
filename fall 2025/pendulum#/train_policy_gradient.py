"""Pygame animation of Policy Gradient learning CartPole"""
import pygame
import numpy as np
import gymnasium as gym
from policy_gradient_simple import PolicyGradientAgent

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Policy Gradient Learning CartPole")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 28)
small_font = pygame.font.Font(None, 22)

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (66, 133, 244)
RED = (234, 67, 53)
GREEN = (52, 168, 83)
YELLOW = (251, 188, 5)
GRAY = (200, 200, 200)

def draw_cart_pole(screen, state, x_offset=400, y_offset=400):
    """Draw the cart and pole"""
    # Unpack state
    x_pos, x_vel, theta, theta_vel = state
    
    # Scale for visualization
    cart_x = int(x_offset + float(x_pos) * 100)
    cart_y = int(y_offset)
    cart_width, cart_height = 60, 30
    pole_length = 120
    
    # Draw track
    pygame.draw.line(screen, GRAY, (50, y_offset + 15), (WIDTH - 50, y_offset + 15), 3)
    
    # Draw cart
    cart_color = BLUE
    pygame.draw.rect(screen, cart_color, 
                     (cart_x - cart_width//2, cart_y - cart_height//2, 
                      cart_width, cart_height))
    pygame.draw.rect(screen, BLACK, 
                     (cart_x - cart_width//2, cart_y - cart_height//2, 
                      cart_width, cart_height), 2)
    
    # Draw pole
    pole_end_x = int(cart_x + pole_length * np.sin(float(theta)))
    pole_end_y = int(cart_y - pole_length * np.cos(float(theta)))
    
    # Color based on angle (green if balanced, red if falling)
    angle_deg = abs(float(theta) * 180 / np.pi)
    if angle_deg < 5:
        pole_color = GREEN
    elif angle_deg < 10:
        pole_color = YELLOW
    else:
        pole_color = RED
    
    pygame.draw.line(screen, pole_color, (cart_x, cart_y), 
                     (pole_end_x, pole_end_y), 8)
    pygame.draw.circle(screen, BLACK, (cart_x, cart_y), 6)
    pygame.draw.circle(screen, pole_color, (pole_end_x, pole_end_y), 8)

def draw_stats(screen, episode, reward, avg_reward, step, max_steps):
    """Draw statistics overlay"""
    # Episode info
    text = font.render(f"Episode: {episode}", True, BLACK)
    screen.blit(text, (10, 10))
    
    # Current reward
    text = font.render(f"Steps: {step}/{max_steps}", True, BLACK)
    screen.blit(text, (10, 45))
    
    # Average reward
    color = GREEN if avg_reward >= 195 else (BLUE if avg_reward >= 100 else BLACK)
    text = font.render(f"Avg (100): {avg_reward:.1f}", True, color)
    screen.blit(text, (10, 80))
    
    # Current episode reward
    text = font.render(f"Current: {reward}", True, BLACK)
    screen.blit(text, (10, 115))
    
    # Progress bar for episode
    bar_width = 200
    bar_height = 20
    progress = step / max_steps
    pygame.draw.rect(screen, GRAY, (10, 150, bar_width, bar_height))
    pygame.draw.rect(screen, GREEN, (10, 150, int(bar_width * progress), bar_height))
    pygame.draw.rect(screen, BLACK, (10, 150, bar_width, bar_height), 2)
    
    # Learning phase indicator
    if episode < 50:
        phase = "Early Learning (Random)"
        phase_color = RED
    elif episode < 150:
        phase = "Learning Phase"
        phase_color = YELLOW
    elif avg_reward >= 195:
        phase = "SOLVED!"
        phase_color = GREEN
    else:
        phase = "Still Learning..."
        phase_color = BLUE
    
    text = small_font.render(phase, True, phase_color)
    screen.blit(text, (10, 180))
    
    # Method name
    text = small_font.render("Policy Gradient (REINFORCE)", True, BLACK)
    screen.blit(text, (10, HEIGHT - 30))

def draw_reward_graph(screen, rewards, x=550, y=50, width=230, height=150):
    """Draw mini reward graph"""
    if len(rewards) < 2:
        return
    
    # Background
    pygame.draw.rect(screen, WHITE, (x, y, width, height))
    pygame.draw.rect(screen, BLACK, (x, y, width, height), 2)
    
    # Title
    text = small_font.render("Reward History", True, BLACK)
    screen.blit(text, (x + 5, y - 25))
    
    # Draw rewards
    max_reward = 500
    window = rewards[-100:] if len(rewards) > 100 else rewards
    
    if len(window) > 1:
        points = []
        for i, r in enumerate(window):
            px = x + (i / len(window)) * width
            py = y + height - (r / max_reward) * height
            points.append((px, py))
        
        # Draw line
        if len(points) > 1:
            pygame.draw.lines(screen, BLUE, False, points, 2)
        
        # Draw solved threshold
        threshold_y = y + height - (195 / max_reward) * height
        pygame.draw.line(screen, GREEN, (x, threshold_y), (x + width, threshold_y), 1)
        text = small_font.render("195", True, GREEN)
        screen.blit(text, (x + width + 5, threshold_y - 10))

def main():
    env = gym.make('CartPole-v1')
    agent = PolicyGradientAgent(learning_rate=0.005, discount_factor=0.99)
    
    episode = 0
    max_episodes = 300
    all_rewards = []
    running = True
    paused = False
    
    print("="*60)
    print("POLICY GRADIENT LEARNING VISUALIZATION")
    print("="*60)
    print("\nControls:")
    print("  SPACE - Pause/Resume")
    print("  Q - Quit")
    print("  + - Speed up")
    print("  - - Slow down")
    print("\nWatch the agent learn to balance the pole!")
    print("="*60)
    
    speed = 1  # Episodes per second multiplier
    
    while running and episode < max_episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    speed = min(speed + 1, 10)
                    print(f"Speed: {speed}x")
                elif event.key == pygame.K_MINUS:
                    speed = max(speed - 1, 1)
                    print(f"Speed: {speed}x")
        
        if paused:
            # Just redraw
            screen.fill(WHITE)
            text = font.render("PAUSED - Press SPACE to resume", True, RED)
            screen.blit(text, (WIDTH//2 - 200, HEIGHT//2))
            pygame.display.flip()
            clock.tick(30)
            continue
        
        # Run episode
        state, _ = env.reset()
        agent.reset_episode()
        episode_reward = 0
        step = 0
        
        while step < 500:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
            
            if not running:
                break
            
            # Agent acts
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.store_reward(reward)
            
            state = next_state
            episode_reward += reward
            step += 1
            
            # Render (every Nth frame based on speed)
            if step % max(1, 6 // speed) == 0:
                screen.fill(WHITE)
                draw_cart_pole(screen, state)
                avg_reward = np.mean(all_rewards[-100:]) if all_rewards else 0
                draw_stats(screen, episode + 1, episode_reward, avg_reward, step, 500)
                draw_reward_graph(screen, all_rewards)
                
                # Speed indicator
                text = small_font.render(f"Speed: {speed}x", True, BLACK)
                screen.blit(text, (WIDTH - 100, HEIGHT - 30))
                
                pygame.display.flip()
                clock.tick(60 * speed)
            
            if terminated or truncated:
                break
        
        # Update policy
        agent.update()
        all_rewards.append(episode_reward)
        episode += 1
        
        # Print progress
        if episode % 50 == 0:
            avg = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
            print(f"Episode {episode}/{max_episodes} | Avg reward: {avg:.1f}")
            if avg >= 195:
                print("🎉 SOLVED! CartPole is balanced!")
    
    # Final screen
    if running:
        screen.fill(WHITE)
        final_avg = np.mean(all_rewards[-100:]) if len(all_rewards) >= 100 else np.mean(all_rewards)
        
        if final_avg >= 195:
            text = font.render("TRAINING COMPLETE - SOLVED!", True, GREEN)
            status = f"Final Average: {final_avg:.1f} (Target: 195)"
        else:
            text = font.render("TRAINING COMPLETE", True, BLUE)
            status = f"Final Average: {final_avg:.1f} (Target: 195 - Need more training)"
        
        screen.blit(text, (WIDTH//2 - 200, HEIGHT//2 - 50))
        text = small_font.render(status, True, BLACK)
        screen.blit(text, (WIDTH//2 - 250, HEIGHT//2))
        text = small_font.render("Close window to continue...", True, BLACK)
        screen.blit(text, (WIDTH//2 - 150, HEIGHT//2 + 50))
        
        pygame.display.flip()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
            clock.tick(30)
    
    env.close()
    pygame.quit()
    
    print(f"\nTraining finished!")
    print(f"Final 100-episode average: {np.mean(all_rewards[-100:]):.1f}")
    print(f"Best episode: {max(all_rewards)}")

if __name__ == "__main__":
    main()
