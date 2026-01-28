"""
Visualization Module for CartPole Comparison

Generates comprehensive comparison plots for three RL methods:
1. Q-Learning (Model-Free, Value-Based)
2. Policy Gradient (Model-Free, Policy-Based)
3. Model-Based LQR (Model-Based Control)

Plots Generated:
1. Average Return vs Episode (with moving average)
2. Success Rate (% episodes with reward > 200)
3. Epsilon Decay Schedule (Q-Learning only)
4. Reward Variance Across Seeds (shaded confidence bands)
5. Method Comparison Summary
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set style for professional-looking plots
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['axes.titlesize'] = 12
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9


def moving_average(data, window=50):
    """
    Compute moving average for smoothing.
    
    Args:
        data: Array of values
        window: Window size for averaging
    
    Returns:
        Smoothed array
    """
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_single_run(results, save_path='single_run_comparison.png'):
    """
    Plot results from a single training run (all three methods).
    
    Args:
        results: Dict with keys 'Q-Learning', 'Policy Gradient', 'Model-Based'
        save_path: Where to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CartPole: RL Methods Comparison (Single Run)', 
                 fontsize=14, fontweight='bold')
    
    colors = {
        'Q-Learning': '#3498db',          # Blue
        'Policy Gradient': '#e74c3c',     # Red
        'Model-Based (LQR)': '#2ecc71'    # Green
    }
    
    # Plot 1: Average Return vs Episode
    ax1 = axes[0, 0]
    for method_name, result in results.items():
        rewards = result['rewards']
        smoothed = moving_average(rewards, window=50)
        episodes = np.arange(len(smoothed))
        
        ax1.plot(episodes, smoothed, label=method_name, 
                color=colors.get(method_name, 'gray'), linewidth=2, alpha=0.8)
        
        # Also plot raw data (lighter)
        ax1.plot(np.arange(len(rewards)), rewards, 
                color=colors.get(method_name, 'gray'), 
                linewidth=0.5, alpha=0.2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return (Reward)')
    ax1.set_title('Average Return vs Episode (50-ep Moving Avg)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=200, color='black', linestyle='--', linewidth=1, 
                alpha=0.5, label='Success Threshold')
    
    # Plot 2: Success Rate Over Time
    ax2 = axes[0, 1]
    window = 100
    for method_name, result in results.items():
        rewards = np.array(result['rewards'])
        success_rate = []
        
        for i in range(len(rewards)):
            start_idx = max(0, i - window + 1)
            window_rewards = rewards[start_idx:i+1]
            rate = np.sum(window_rewards > 200) / len(window_rewards) * 100
            success_rate.append(rate)
        
        ax2.plot(np.arange(len(success_rate)), success_rate, 
                label=method_name, color=colors.get(method_name, 'gray'), 
                linewidth=2)
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title(f'Success Rate (% Episodes > 200, {window}-ep Window)')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])
    
    # Plot 3: Epsilon Decay (Q-Learning only)
    ax3 = axes[1, 0]
    if 'Q-Learning' in results and 'epsilons' in results['Q-Learning']:
        epsilons = results['Q-Learning']['epsilons']
        ax3.plot(np.arange(len(epsilons)), epsilons, 
                color=colors['Q-Learning'], linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon (ε)')
        ax3.set_title('Q-Learning: Epsilon Decay Schedule')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim([0, 1.05])
    else:
        ax3.text(0.5, 0.5, 'Epsilon decay\n(Q-Learning only)', 
                ha='center', va='center', fontsize=12)
        ax3.set_xticks([])
        ax3.set_yticks([])
    
    # Plot 4: Final Performance Comparison
    ax4 = axes[1, 1]
    method_names = []
    final_avgs = []
    success_rates = []
    
    for method_name, result in results.items():
        rewards = np.array(result['rewards'])
        method_names.append(method_name.replace(' (LQR)', '\n(LQR)'))
        final_avgs.append(np.mean(rewards[-100:]))
        success_rates.append(np.sum(rewards > 200) / len(rewards) * 100)
    
    x = np.arange(len(method_names))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, final_avgs, width, label='Avg Return (last 100)', 
                   color=[colors[list(results.keys())[i]] for i in range(len(method_names))],
                   alpha=0.7)
    bars2 = ax4.bar(x + width/2, success_rates, width, label='Success Rate (%)', 
                   color=[colors[list(results.keys())[i]] for i in range(len(method_names))],
                   alpha=0.4)
    
    ax4.set_ylabel('Score')
    ax4.set_title('Final Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(method_names, fontsize=9)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_multi_seed_comparison(all_results, save_path='multi_seed_comparison.png'):
    """
    Plot results from multiple seeds with confidence bands.
    
    Args:
        all_results: Dict with structure:
            {method_name: [result_seed1, result_seed2, ...]}
        save_path: Where to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Multi-Seed Analysis: Reward Variance Across Methods', 
                 fontsize=14, fontweight='bold')
    
    colors = {
        'Q-Learning': '#3498db',
        'Policy Gradient': '#e74c3c',
        'Model-Based (LQR)': '#2ecc71'
    }
    
    # Plot 1: Mean ± Std across seeds
    ax1 = axes[0]
    for method_name, seed_results in all_results.items():
        # Stack rewards from all seeds
        all_rewards = []
        max_length = 0
        for result in seed_results:
            rewards = result['rewards']
            all_rewards.append(rewards)
            max_length = max(max_length, len(rewards))
        
        # Pad shorter runs with last value
        padded_rewards = []
        for rewards in all_rewards:
            if len(rewards) < max_length:
                padded = np.concatenate([rewards, [rewards[-1]]*(max_length - len(rewards))])
            else:
                padded = rewards
            padded_rewards.append(padded)
        
        rewards_array = np.array(padded_rewards)
        
        # Compute mean and std
        mean_rewards = np.mean(rewards_array, axis=0)
        std_rewards = np.std(rewards_array, axis=0)
        
        # Smooth
        mean_smooth = moving_average(mean_rewards, window=50)
        episodes = np.arange(len(mean_smooth))
        
        # Plot mean
        ax1.plot(episodes, mean_smooth, label=method_name, 
                color=colors.get(method_name, 'gray'), linewidth=2)
        
        # Plot confidence band (mean ± std)
        if len(episodes) == len(mean_smooth):
            std_smooth = moving_average(std_rewards, window=50)
            ax1.fill_between(episodes, 
                            mean_smooth - std_smooth, 
                            mean_smooth + std_smooth, 
                            color=colors.get(method_name, 'gray'), 
                            alpha=0.2)
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return (Reward)')
    ax1.set_title(f'Mean Reward ± Std ({len(seed_results)} seeds)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=200, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Plot 2: Final performance distribution
    ax2 = axes[1]
    method_names = []
    final_means = []
    final_stds = []
    
    for method_name, seed_results in all_results.items():
        final_rewards = [np.mean(result['rewards'][-100:]) for result in seed_results]
        method_names.append(method_name.replace(' (LQR)', '\n(LQR)'))
        final_means.append(np.mean(final_rewards))
        final_stds.append(np.std(final_rewards))
    
    x = np.arange(len(method_names))
    bars = ax2.bar(x, final_means, 
                   color=[colors[list(all_results.keys())[i]] for i in range(len(method_names))],
                   alpha=0.7, yerr=final_stds, capsize=5)
    
    ax2.set_ylabel('Final Avg Return (last 100 eps)')
    ax2.set_title(f'Final Performance ({len(seed_results)} seeds)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(method_names, fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, final_means, final_stds)):
        ax2.text(bar.get_x() + bar.get_width()/2., mean + std,
                f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def print_summary_statistics(results):
    """
    Print summary statistics for all methods.
    
    Args:
        results: Dict with keys being method names
    """
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    for method_name, result in results.items():
        rewards = np.array(result['rewards'])
        
        print(f"\n{method_name}:")
        print(f"  Final 100-episode average: {np.mean(rewards[-100:]):.2f} ± {np.std(rewards[-100:]):.2f}")
        print(f"  Overall average: {np.mean(rewards):.2f}")
        print(f"  Maximum episode reward: {np.max(rewards):.0f}")
        print(f"  Success rate (>200): {np.sum(rewards > 200) / len(rewards) * 100:.1f}%")
        print(f"  Episodes to reach 200 avg: ", end="")
        
        # Find first episode where moving average reaches 200
        window = 100
        reached = False
        for i in range(window, len(rewards)):
            if np.mean(rewards[i-window:i]) >= 200:
                print(f"{i} episodes")
                reached = True
                break
        if not reached:
            print("Not reached")
    
    print("="*70)


if __name__ == "__main__":
    # Test plotting with dummy data
    print("Plotting module ready for use with pendulum_compare.py")
