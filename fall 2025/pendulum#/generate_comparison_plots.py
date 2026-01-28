"""Generate comprehensive comparison plots for RL methods"""
import numpy as np
import matplotlib.pyplot as plt
from q_learning import train_q_learning
from model_based import train_model_based

def smooth_curve(data, window=10):
    """Smooth data using moving average"""
    return np.convolve(data, np.ones(window)/window, mode='valid')

def generate_all_plots():
    print("=" * 70)
    print("COMPREHENSIVE RL METHODS COMPARISON")
    print("=" * 70)
    
    # Train both methods with 500 episodes
    print("\n[1/2] Training Q-Learning (500 episodes)...")
    q_results = train_q_learning(n_episodes=500, seed=42, verbose=True)
    
    print("\n[2/2] Evaluating Model-Based LQR (500 episodes)...")
    lqr_results = train_model_based(n_episodes=500, seed=42, verbose=True)
    
    # Create comprehensive comparison plots
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Learning Curves
    ax1 = plt.subplot(2, 3, 1)
    episodes = np.arange(1, 501)
    ax1.plot(episodes, q_results['rewards'], alpha=0.3, color='blue', linewidth=0.5)
    ax1.plot(episodes, smooth_curve(q_results['rewards'], 20), color='blue', linewidth=2, label='Q-Learning')
    ax1.plot(episodes, lqr_results['rewards'], color='red', linewidth=2, label='Model-Based LQR', alpha=0.8)
    ax1.axhline(y=195, color='green', linestyle='--', linewidth=1.5, label='Solved Threshold')
    ax1.set_xlabel('Episode', fontsize=11)
    ax1.set_ylabel('Episode Reward', fontsize=11)
    ax1.set_title('Learning Curves Comparison', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative Average Reward
    ax2 = plt.subplot(2, 3, 2)
    q_cumavg = np.cumsum(q_results['rewards']) / episodes
    lqr_cumavg = np.cumsum(lqr_results['rewards']) / episodes
    ax2.plot(episodes, q_cumavg, color='blue', linewidth=2, label='Q-Learning')
    ax2.plot(episodes, lqr_cumavg, color='red', linewidth=2, label='Model-Based LQR')
    ax2.set_xlabel('Episode', fontsize=11)
    ax2.set_ylabel('Cumulative Average Reward', fontsize=11)
    ax2.set_title('Cumulative Performance', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rolling Average (100 episodes)
    ax3 = plt.subplot(2, 3, 3)
    q_rolling = [np.mean(q_results['rewards'][max(0, i-100):i+1]) for i in range(500)]
    lqr_rolling = [np.mean(lqr_results['rewards'][max(0, i-100):i+1]) for i in range(500)]
    ax3.plot(episodes, q_rolling, color='blue', linewidth=2, label='Q-Learning')
    ax3.plot(episodes, lqr_rolling, color='red', linewidth=2, label='Model-Based LQR')
    ax3.axhline(y=195, color='green', linestyle='--', linewidth=1.5, label='Solved')
    ax3.set_xlabel('Episode', fontsize=11)
    ax3.set_ylabel('100-Episode Average', fontsize=11)
    ax3.set_title('Rolling Average Performance', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Success Rate Over Time
    ax4 = plt.subplot(2, 3, 4)
    window_size = 50
    q_success = [np.mean(np.array(q_results['rewards'][max(0, i-window_size):i+1]) >= 195) * 100 
                 for i in range(500)]
    lqr_success = [np.mean(np.array(lqr_results['rewards'][max(0, i-window_size):i+1]) >= 195) * 100 
                   for i in range(500)]
    ax4.plot(episodes, q_success, color='blue', linewidth=2, label='Q-Learning')
    ax4.plot(episodes, lqr_success, color='red', linewidth=2, label='Model-Based LQR')
    ax4.set_xlabel('Episode', fontsize=11)
    ax4.set_ylabel('Success Rate (%)', fontsize=11)
    ax4.set_title(f'Success Rate (Rolling {window_size} Episodes)', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-5, 105)
    
    # Plot 5: Reward Distribution
    ax5 = plt.subplot(2, 3, 5)
    bins = np.linspace(0, 500, 30)
    ax5.hist(q_results['rewards'], bins=bins, alpha=0.6, color='blue', label='Q-Learning', edgecolor='black')
    ax5.hist(lqr_results['rewards'], bins=bins, alpha=0.6, color='red', label='Model-Based LQR', edgecolor='black')
    ax5.axvline(x=195, color='green', linestyle='--', linewidth=2, label='Solved Threshold')
    ax5.set_xlabel('Episode Reward', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.set_title('Reward Distribution (500 Episodes)', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Performance Summary Table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate statistics
    q_final = np.mean(q_results['rewards'][-100:])
    lqr_final = np.mean(lqr_results['rewards'][-100:])
    q_total = np.sum(q_results['rewards'])
    lqr_total = np.sum(lqr_results['rewards'])
    q_success_rate = np.sum(np.array(q_results['rewards']) >= 195) / 500 * 100
    lqr_success_rate = np.sum(np.array(lqr_results['rewards']) >= 195) / 500 * 100
    q_std = np.std(q_results['rewards'][-100:])
    lqr_std = np.std(lqr_results['rewards'][-100:])
    
    table_data = [
        ['Metric', 'Q-Learning', 'LQR'],
        ['', '', ''],
        ['Final Avg (100 eps)', f'{q_final:.1f}', f'{lqr_final:.1f}'],
        ['Std Dev (100 eps)', f'{q_std:.1f}', f'{lqr_std:.1f}'],
        ['Total Reward', f'{q_total:.0f}', f'{lqr_total:.0f}'],
        ['Success Rate', f'{q_success_rate:.1f}%', f'{lqr_success_rate:.1f}%'],
        ['Max Reward', f'{np.max(q_results["rewards"]):.0f}', f'{np.max(lqr_results["rewards"]):.0f}'],
        ['Min Reward', f'{np.min(q_results["rewards"]):.0f}', f'{np.min(lqr_results["rewards"]):.0f}'],
        ['Median Reward', f'{np.median(q_results["rewards"]):.0f}', f'{np.median(lqr_results["rewards"]):.0f}'],
    ]
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Header formatting
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternating row colors
    for i in range(2, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax6.set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('rl_comparison_comprehensive.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: rl_comparison_comprehensive.png")
    
    # Create individual detailed plots
    create_individual_plots(q_results, lqr_results)
    
    plt.show()

def create_individual_plots(q_results, lqr_results):
    """Create additional individual plots"""
    
    # Q-Learning detailed analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Q-Learning Detailed Analysis', fontsize=16, fontweight='bold')
    
    episodes = np.arange(1, 501)
    
    # Raw rewards
    axes[0, 0].plot(episodes, q_results['rewards'], alpha=0.5, color='blue', linewidth=1)
    axes[0, 0].plot(episodes, smooth_curve(q_results['rewards'], 30), color='darkblue', linewidth=2)
    axes[0, 0].axhline(y=195, color='green', linestyle='--', label='Solved')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Episode Rewards with Smoothing (window=30)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Exploration decay
    axes[0, 1].plot(episodes, q_results['epsilons'], color='orange', linewidth=2)
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Epsilon (Exploration Rate)')
    axes[0, 1].set_title('Exploration vs Exploitation Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Steps per episode
    axes[1, 0].plot(episodes, q_results['steps'], alpha=0.5, color='purple', linewidth=1)
    axes[1, 0].plot(episodes, smooth_curve(q_results['steps'], 30), color='darkviolet', linewidth=2)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Steps')
    axes[1, 0].set_title('Episode Length (Steps)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reward vs Epsilon scatter
    axes[1, 1].scatter(q_results['epsilons'], q_results['rewards'], alpha=0.4, s=10)
    axes[1, 1].set_xlabel('Epsilon')
    axes[1, 1].set_ylabel('Reward')
    axes[1, 1].set_title('Reward vs Exploration Rate')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('q_learning_detailed.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: q_learning_detailed.png")
    
    # Comparison box plots
    fig, ax = plt.subplots(figsize=(10, 6))
    box_data = [q_results['rewards'], lqr_results['rewards']]
    bp = ax.boxplot(box_data, labels=['Q-Learning', 'Model-Based LQR'],
                    patch_artist=True, showmeans=True)
    
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.axhline(y=195, color='green', linestyle='--', linewidth=2, label='Solved Threshold')
    ax.set_ylabel('Episode Reward', fontsize=12)
    ax.set_title('Reward Distribution Comparison (500 Episodes)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('comparison_boxplot.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: comparison_boxplot.png")

if __name__ == "__main__":
    generate_all_plots()
    print("\n" + "=" * 70)
    print("All comparison plots generated successfully!")
    print("=" * 70)
