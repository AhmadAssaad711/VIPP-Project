"""Quick comparison with 300 episodes"""
import numpy as np
import matplotlib.pyplot as plt
from q_learning import train_q_learning
from model_based import train_model_based

# Train both methods
print("Training Q-Learning (300 episodes)...")
q_res = train_q_learning(n_episodes=300, seed=42, verbose=True)

print("\nEvaluating LQR (300 episodes)...")  
lqr_res = train_model_based(n_episodes=300, seed=42, verbose=True)

# Create main comparison plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('CartPole RL Methods Comparison', fontsize=16, fontweight='bold')

eps = np.arange(1, 301)

# Plot 1: Learning curves
axes[0, 0].plot(eps, q_res['rewards'], alpha=0.3, color='blue', linewidth=0.8)
smoothed = np.convolve(q_res['rewards'], np.ones(20)/20, mode='valid')
axes[0, 0].plot(eps[:len(smoothed)], smoothed, 
                color='blue', linewidth=2.5, label='Q-Learning')
axes[0, 0].plot(eps, lqr_res['rewards'], color='red', linewidth=2.5, label='Model-Based LQR', alpha=0.9)
axes[0, 0].axhline(y=195, color='green', linestyle='--', linewidth=2, label='Solved Threshold')
axes[0, 0].set_xlabel('Episode', fontsize=12)
axes[0, 0].set_ylabel('Reward', fontsize=12)
axes[0, 0].set_title('Learning Curves', fontsize=13, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Rolling average (100 eps)
q_roll = [np.mean(q_res['rewards'][max(0, i-100):i+1]) for i in range(300)]
lqr_roll = [np.mean(lqr_res['rewards'][max(0, i-100):i+1]) for i in range(300)]
axes[0, 1].plot(eps, q_roll, color='blue', linewidth=2.5, label='Q-Learning')
axes[0, 1].plot(eps, lqr_roll, color='red', linewidth=2.5, label='Model-Based LQR')
axes[0, 1].axhline(y=195, color='green', linestyle='--', linewidth=2)
axes[0, 1].set_xlabel('Episode', fontsize=12)
axes[0, 1].set_ylabel('100-Episode Average', fontsize=12)
axes[0, 1].set_title('Rolling Average Performance', fontsize=13, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: Reward distribution
bins = np.linspace(0, 500, 25)
axes[1, 0].hist(q_res['rewards'], bins=bins, alpha=0.7, color='blue', 
                label='Q-Learning', edgecolor='black', linewidth=1.2)
axes[1, 0].hist(lqr_res['rewards'], bins=bins, alpha=0.7, color='red', 
                label='LQR', edgecolor='black', linewidth=1.2)
axes[1, 0].axvline(x=195, color='green', linestyle='--', linewidth=2.5, label='Solved')
axes[1, 0].set_xlabel('Reward', fontsize=12)
axes[1, 0].set_ylabel('Frequency', fontsize=12)
axes[1, 0].set_title('Reward Distribution', fontsize=13, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Plot 4: Statistics table
axes[1, 1].axis('off')
q_final = np.mean(q_res['rewards'][-100:])
lqr_final = np.mean(lqr_res['rewards'][-100:])
q_success = np.sum(np.array(q_res['rewards']) >= 195) / 300 * 100
lqr_success = np.sum(np.array(lqr_res['rewards']) >= 195) / 300 * 100

table_data = [
    ['Metric', 'Q-Learning', 'LQR'],
    ['Final Avg (100)', f'{q_final:.1f}', f'{lqr_final:.1f}'],
    ['Std Dev', f'{np.std(q_res["rewards"][-100:]):.1f}', f'{np.std(lqr_res["rewards"][-100:]):.1f}'],
    ['Success Rate', f'{q_success:.1f}%', f'{lqr_success:.1f}%'],
    ['Max Reward', f'{np.max(q_res["rewards"]):.0f}', f'{np.max(lqr_res["rewards"]):.0f}'],
    ['Total Reward', f'{np.sum(q_res["rewards"]):.0f}', f'{np.sum(lqr_res["rewards"]):.0f}'],
]

table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.4, 0.3, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 3)

for i in range(3):
    table[(0, i)].set_facecolor('#2196F3')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(table_data)):
    for j in range(3):
        if i % 2 == 1:
            table[(i, j)].set_facecolor('#f5f5f5')

axes[1, 1].set_title('Performance Summary', fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('rl_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: rl_comparison.png")
plt.close()

# Create Q-Learning details plot
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle('Q-Learning Detailed Analysis (300 Episodes)', fontsize=15, fontweight='bold')

axes[0, 0].plot(eps, q_res['rewards'], alpha=0.4, color='blue')
smoothed = np.convolve(q_res['rewards'], np.ones(30)/30, mode='valid')
axes[0, 0].plot(eps[:len(smoothed)], smoothed, color='darkblue', linewidth=2)
axes[0, 0].axhline(y=195, color='green', linestyle='--')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].set_title('Rewards (30-episode smoothing)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(eps, q_res['epsilons'], color='orange', linewidth=2)
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Epsilon')
axes[0, 1].set_title('Exploration Rate Decay')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(eps, q_res['steps'], alpha=0.4, color='purple')
smoothed = np.convolve(q_res['steps'], np.ones(30)/30, mode='valid')
axes[1, 0].plot(eps[:len(smoothed)], smoothed, color='darkviolet', linewidth=2)
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Steps')
axes[1, 0].set_title('Episode Length')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(q_res['epsilons'], q_res['rewards'], alpha=0.5, s=15, c=eps, cmap='viridis')
axes[1, 1].set_xlabel('Epsilon')
axes[1, 1].set_ylabel('Reward')
axes[1, 1].set_title('Reward vs Exploration')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('q_learning_details.png', dpi=300, bbox_inches='tight')
print("✓ Saved: q_learning_details.png")

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"\nQ-Learning (Model-Free, Value-Based):")
print(f"  • Final 100-episode average: {q_final:.1f}")
print(f"  • Success rate (≥195): {q_success:.1f}%")
print(f"  • Total cumulative reward: {np.sum(q_res['rewards']):.0f}")
print(f"\nModel-Based LQR (Optimal Control):")
print(f"  • Final 100-episode average: {lqr_final:.1f}")
print(f"  • Success rate (≥195): {lqr_success:.1f}%")
print(f"  • Total cumulative reward: {np.sum(lqr_res['rewards']):.0f}")
print(f"\n{'='*70}")
print("✓ All plots generated successfully!")
print("="*70)
