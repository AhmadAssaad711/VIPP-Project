"""
Model-Based Control for CartPole using LQR

Model-Based Reinforcement Learning
==================================

Unlike model-free methods (Q-Learning, Policy Gradient), model-based approaches
use knowledge of the system dynamics to compute optimal control policies.

Linear Quadratic Regulator (LQR):
- Assumes linear dynamics: x_{t+1} = Ax_t + Bu_t
- Quadratic cost: J = Σ (x^T Q x + u^T R u)
- Computes optimal gain matrix K via Riccati equation
- Control law: u = -Kx (simple matrix multiplication)

Key Differences from Model-Free Methods:
========================================

1. **Requires Model Knowledge:**
   - Needs system matrices (A, B) describing dynamics
   - Model-free methods learn purely from experience
   
2. **No Learning Required:**
   - Analytical solution (no episodes, no gradients)
   - Instant policy computation
   
3. **Optimality Guarantees:**
   - Globally optimal for linear systems with quadratic cost
   - Model-free methods converge to local optima
   
4. **Limited Applicability:**
   - Only works for systems with known/linearized dynamics
   - Model-free methods work for any MDP
   
5. **Sample Efficiency:**
   - Zero samples needed (uses model)
   - Model-free requires thousands of episodes

Assumptions:
- System is linear or can be linearized around equilibrium
- Dynamics are known (A, B matrices)
- Cost is quadratic (Q, R matrices)
- Full state observability

For CartPole:
- Linearize around upright position (θ = 0)
- Use physics equations to derive A, B
- Design Q, R to penalize angle and position errors
"""

import numpy as np
import gymnasium as gym
from scipy import linalg


class LQRController:
    """
    Linear Quadratic Regulator for CartPole.
    
    System dynamics (linearized around θ=0):
        x_{t+1} = A x_t + B u_t
    
    where x = [position, velocity, angle, angular_velocity]
          u = force applied to cart
    """
    
    def __init__(self):
        """
        Initialize LQR controller with CartPole dynamics.
        """
        # CartPole parameters (from Gymnasium source)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # Half pole length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        
        # Linearized system matrices
        self.A, self.B = self._compute_linear_dynamics()
        
        # Cost matrices
        # Q: State cost (penalize position, angle)
        # R: Control cost (penalize large forces)
        self.Q = np.diag([1.0, 0.1, 10.0, 0.1])  # High cost on angle
        self.R = np.array([[0.1]])  # Small control cost
        
        # Compute optimal gain matrix K
        self.K = self._solve_lqr()
        
        print("\n" + "="*60)
        print("LQR Controller Initialized")
        print("="*60)
        print(f"Optimal Gain Matrix K:")
        print(f"  K = {self.K}")
        print(f"\nControl Law: u = -K @ x")
        print(f"  u = -{self.K[0,0]:.2f}*pos {self.K[0,1]:.2f}*vel " 
              f"{self.K[0,2]:.2f}*θ {self.K[0,3]:.2f}*θ_dot")
        print("="*60)
        
    def _compute_linear_dynamics(self):
        """
        Compute linearized dynamics matrices A, B.
        
        Linearization around equilibrium (θ = 0, upright position):
            ẍ = (u + m*l*θ̈*sin(θ) - m*l*θ̇²*sin(θ)) / M
            θ̈ = (g*sin(θ) - cos(θ)*(u + m*l*θ̇²*sin(θ))/M) / (l*(4/3 - m*cos²(θ)/M))
        
        At θ = 0:
            sin(θ) ≈ θ, cos(θ) ≈ 1, θ̇² ≈ 0
        
        Returns:
            A: 4x4 state transition matrix
            B: 4x1 input matrix
        """
        g = self.gravity
        m = self.masspole
        M = self.total_mass
        l = self.length
        
        # Continuous-time matrices (derivative form)
        # State: [x, ẋ, θ, θ̇]
        A_cont = np.array([
            [0, 1, 0, 0],
            [0, 0, -m*g/M, 0],
            [0, 0, 0, 1],
            [0, 0, g*(M+m)/(M*l), 0]
        ])
        
        B_cont = np.array([
            [0],
            [1/M],
            [0],
            [-1/(M*l)]
        ])
        
        # Discrete-time conversion (Euler method, dt=0.02)
        dt = 0.02
        A = np.eye(4) + A_cont * dt
        B = B_cont * dt
        
        return A, B
    
    def _solve_lqr(self):
        """
        Solve discrete-time algebraic Riccati equation to find optimal gain K.
        
        Riccati equation:
            P = A^T P A - A^T P B (R + B^T P B)^{-1} B^T P A + Q
        
        Optimal gain:
            K = (R + B^T P B)^{-1} B^T P A
        
        Returns:
            K: Optimal gain matrix (1x4)
        """
        try:
            # Solve discrete-time Riccati equation
            P = linalg.solve_discrete_are(self.A, self.B, self.Q, self.R)
            
            # Compute optimal gain
            K = linalg.inv(self.R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
            
            return K
        
        except Exception as e:
            print(f"LQR solution failed: {e}")
            print("Using default gains...")
            return np.array([[1.0, 1.5, 18.0, 3.0]])  # Hand-tuned fallback
    
    def get_action(self, state, explore=False):
        """
        Compute control action using LQR law: u = -K @ x
        
        Args:
            state: Current state [x, ẋ, θ, θ̇]
            explore: Not used (LQR is deterministic)
        
        Returns:
            action: 0 (left) or 1 (right)
        """
        # Compute continuous control: u = -K @ x
        u = -(self.K @ state.reshape(-1, 1))[0, 0]
        
        # Convert to discrete action (0 or 1)
        # u > 0 → push right (action 1)
        # u < 0 → push left (action 0)
        action = 1 if u > 0 else 0
        
        return action


def train_model_based(n_episodes=500, seed=42, verbose=True):
    """
    "Train" (evaluate) LQR controller on CartPole.
    
    Note: LQR doesn't require training - the optimal controller is
    computed analytically from the system model.
    
    Args:
        n_episodes: Number of evaluation episodes
        seed: Random seed
        verbose: Whether to print progress
    
    Returns:
        dict: Evaluation history
    """
    # Set random seed
    np.random.seed(seed)
    
    # Create environment
    env = gym.make('CartPole-v1')
    env.reset(seed=seed)
    
    # Initialize LQR controller
    controller = LQRController()
    
    # Evaluation history
    rewards = []
    steps = []
    
    if verbose:
        print("\n" + "="*60)
        print("MODEL-BASED: LQR Control")
        print("="*60)
        print("\nMethod Description:")
        print("- Uses known system dynamics (linearized)")
        print("- Analytically computes optimal controller")
        print("- No learning required (zero training samples)")
        print("- Control law: u = -K @ state")
        print("\nEvaluating for {} episodes...".format(n_episodes))
    
    # Evaluation loop
    for episode in range(n_episodes):
        state, _ = env.reset()
        
        episode_reward = 0
        episode_steps = 0
        
        # Episode loop
        for step in range(500):
            # Compute action using LQR
            action = controller.get_action(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_steps += 1
            
            if done:
                break
        
        # Record metrics
        rewards.append(episode_reward)
        steps.append(episode_steps)
        
        # Print progress
        if verbose and (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards[-100:])
            print(f"Episode {episode+1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.1f}")
    
    env.close()
    
    if verbose:
        final_avg = np.mean(rewards[-100:])
        success_rate = np.sum(np.array(rewards) > 200) / n_episodes * 100
        print(f"\nEvaluation Complete!")
        print(f"Final 100-episode average: {final_avg:.1f}")
        print(f"Success rate (>200): {success_rate:.1f}%")
        print("\nKey Insight:")
        print("LQR achieves near-optimal performance instantly,")
        print("but only works because we know the system dynamics.")
        print("Model-free methods learn from scratch without this knowledge.")
        print("="*60)
    
    return {
        'rewards': rewards,
        'steps': steps,
        'controller': controller,
        'method': 'Model-Based (LQR)'
    }


if __name__ == "__main__":
    # Test LQR independently
    results = train_model_based(n_episodes=500, seed=42)
    print(f"\nFinal average reward: {np.mean(results['rewards'][-100:]):.2f}")
