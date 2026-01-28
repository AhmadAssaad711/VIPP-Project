"""Three-way comparison: Q-Learning vs Policy Gradient vs LQR"""
import numpy as np
import matplotlib.pyplot as plt
from q_learning import train_q_learning
from model_based import train_model_based
from policy_gradient_simple import train_policy_gradient

print("="*70)
print("THREE-WAY RL METHODS COMPARISON")
print("="*70)
print("\n1. Q-Learning (Model-Free, Value-Based)")
print("2. Policy Gradient (Model-Free, Policy-Based)")
print("3. Model-Based LQR (Optimal Control)")
print("\n" + "="*70)

# Train all three methods
print("\n[1/3] Training Q-Learning (300 episodes)...")
q_res = train_q_learning(n_episodes=300, seed=42, verbose=True)

print("\n[2/3] Training Policy Gradient (300 episodes)...")
pg_res = train_policy_gradient(n_episodes=300, seed=42, verbose=True)

print("\n[3/3] Evaluating LQR (300 episodes)...")
lqr_res = train_model_based(n_episodes=300, seed=42, verbose=True)

# Create comprehensive comparison
fig = plt.figure(figsize=(18, 11))

eps = np.arange(1, 301)

# Plot 1: Learning Curves
ax1 = plt.subplot(3, 3, 1)
ax1.plot(eps, q_res['rewards'], alpha=0.25, color='blue', linewidth=0.7)
smoothed_q = np.convolve(q_res['rewards'], np.ones(20)/20, mode='valid')
ax1.plot(eps[:len(smoothed_q)], smoothed_q, color='blue', linewidth=2.5, label='Q-Learning')

ax1.plot(eps, pg_res['rewards'], alpha=0.25, color='purple', linewidth=0.7)
smoothed_pg = np.convolve(pg_res['rewards'], np.ones(20)/20, mode='valid')
ax1.plot(eps[:len(smoothed_pg)], smoothed_pg, color='purple', linewidth=2.5, label='Policy Gradient')

ax1.plot(eps, lqr_res['rewards'], color='red', linewidth=2.5, label='LQR', alpha=0.9)
ax1.axhline(y=195, color='green', linestyle='--', linewidth=2, label='Solved')
ax1.set_xlabel('Episode', fontsize=11)
ax1.set_ylabel('Reward', fontsize=11)
ax1.set_title('Learning Curves (20-ep smoothing)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9, loc='lower right')
ax1.grid(True, alpha=0.3)

# Plot 2: Rolling Average
ax2 = plt.subplot(3, 3, 2)
q_roll = [np.mean(q_res['rewards'][max(0, i-100):i+1]) for i in range(300)]
pg_roll = [np.mean(pg_res['rewards'][max(0, i-100):i+1]) for i in range(300)]
lqr_roll = [np.mean(lqr_res['rewards'][max(0, i-100):i+1]) for i in range(300)]
ax2.plot(eps, q_roll, color='blue', linewidth=2.5, label='Q-Learning')
ax2.plot(eps, pg_roll, color='purple', linewidth=2.5, label='Policy Gradient')
ax2.plot(eps, lqr_roll, color='red', linewidth=2.5, label='LQR')
ax2.axhline(y=195, color='green', linestyle='--', linewidth=2)
ax2.set_xlabel('Episode', fontsize=11)
ax2.set_ylabel('100-Episode Average', fontsize=11)
ax2.set_title('Rolling Average Performance', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Cumulative Reward
ax3 = plt.subplot(3, 3, 3)
ax3.plot(eps, np.cumsum(q_res['rewards']), color='blue', linewidth=2.5, label='Q-Learning')
ax3.plot(eps, np.cumsum(pg_res['rewards']), color='purple', linewidth=2.5, label='Policy Gradient')
ax3.plot(eps, np.cumsum(lqr_res['rewards']), color='red', linewidth=2.5, label='LQR')
ax3.set_xlabel('Episode', fontsize=11)
ax3.set_ylabel('Cumulative Reward', fontsize=11)
ax3.set_title('Total Cumulative Reward', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Reward Distribution
ax4 = plt.subplot(3, 3, 4)
bins = np.linspace(0, 500, 25)
ax4.hist(q_res['rewards'], bins=bins, alpha=0.6, color='blue', label='Q-Learning', edgecolor='black')
ax4.hist(pg_res['rewards'], bins=bins, alpha=0.6, color='purple', label='Policy Gradient', edgecolor='black')
ax4.hist(lqr_res['rewards'], bins=bins, alpha=0.6, color='red', label='LQR', edgecolor='black')
ax4.axvline(x=195, color='green', linestyle='--', linewidth=2.5)
ax4.set_xlabel('Reward', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title('Reward Distribution', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Box Plot Comparison
ax5 = plt.subplot(3, 3, 5)
box_data = [q_res['rewards'], pg_res['rewards'], lqr_res['rewards']]
bp = ax5.boxplot(box_data, labels=['Q-Learning', 'Policy\nGradient', 'LQR'],
                patch_artist=True, showmeans=True)
colors = ['lightblue', 'plum', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax5.axhline(y=195, color='green', linestyle='--', linewidth=2)
ax5.set_ylabel('Reward', fontsize=11)
ax5.set_title('Statistical Comparison', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Plot 6: Success Rate Over Time
ax6 = plt.subplot(3, 3, 6)
window = 50
q_success = [np.mean(np.array(q_res['rewards'][max(0, i-window):i+1]) >= 195) * 100 for i in range(300)]
pg_success = [np.mean(np.array(pg_res['rewards'][max(0, i-window):i+1]) >= 195) * 100 for i in range(300)]
lqr_success = [np.mean(np.array(lqr_res['rewards'][max(0, i-window):i+1]) >= 195) * 100 for i in range(300)]
ax6.plot(eps, q_success, color='blue', linewidth=2.5, label='Q-Learning')
ax6.plot(eps, pg_success, color='purple', linewidth=2.5, label='Policy Gradient')
ax6.plot(eps, lqr_success, color='red', linewidth=2.5, label='LQR')
ax6.set_xlabel('Episode', fontsize=11)
ax6.set_ylabel('Success Rate (%)', fontsize=11)
ax6.set_title(f'Success Rate (Rolling {window} eps)', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)
ax6.set_ylim(-5, 105)

# Plot 7: Q-Learning Details
ax7 = plt.subplot(3, 3, 7)
ax7.plot(eps, q_res['epsilons'], color='orange', linewidth=2)
ax7.set_xlabel('Episode', fontsize=11)
ax7.set_ylabel('Epsilon', fontsize=11)
ax7.set_title('Q-Learning Exploration Decay', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

# Plot 8: Policy Gradient Loss
ax8 = plt.subplot(3, 3, 8)
ax8.plot(eps, pg_res['losses'], alpha=0.3, color='purple', linewidth=0.7)
smoothed_loss = np.convolve(pg_res['losses'], np.ones(20)/20, mode='valid')
ax8.plot(eps[:len(smoothed_loss)], smoothed_loss, color='purple', linewidth=2)
ax8.set_xlabel('Episode', fontsize=11)
ax8.set_ylabel('Loss', fontsize=11)
ax8.set_title('Policy Gradient Loss', fontsize=12, fontweight='bold')
ax8.grid(True, alpha=0.3)

# Plot 9: Performance Table
ax9 = plt.subplot(3, 3, 9)
ax9.axis('off')

q_final = np.mean(q_res['rewards'][-100:])
pg_final = np.mean(pg_res['rewards'][-100:])
lqr_final = np.mean(lqr_res['rewards'][-100:])

q_success_rate = np.sum(np.array(q_res['rewards']) >= 195) / 300 * 100
pg_success_rate = np.sum(np.array(pg_res['rewards']) >= 195) / 300 * 100
lqr_success_rate = np.sum(np.array(lqr_res['rewards']) >= 195) / 300 * 100

table_data = [
    ['Metric', 'Q-Learn', 'Policy Grad', 'LQR'],
    ['Final Avg', f'{q_final:.1f}', f'{pg_final:.1f}', f'{lqr_final:.1f}'],
    ['Std Dev', f'{np.std(q_res["rewards"][-100:]):.1f}', 
     f'{np.std(pg_res["rewards"][-100:]):.1f}', f'{np.std(lqr_res["rewards"][-100:]):.1f}'],
    ['Success %', f'{q_success_rate:.1f}', f'{pg_success_rate:.1f}', f'{lqr_success_rate:.1f}'],
    ['Max', f'{np.max(q_res["rewards"]):.0f}', f'{np.max(pg_res["rewards"]):.0f}', 
     f'{np.max(lqr_res["rewards"]):.0f}'],
    ['Total', f'{np.sum(q_res["rewards"]):.0f}', f'{np.sum(pg_res["rewards"]):.0f}', 
     f'{np.sum(lqr_res["rewards"]):.0f}'],
]

table = ax9.table(cellText=table_data, cellLoc='center', loc='center', colWidths=[0.3, 0.23, 0.23, 0.23])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.5)

for i in range(4):
    table[(0, i)].set_facecolor('#FF6B6B')
    table[(0, i)].set_text_props(weight='bold', color='white')

for i in range(1, len(table_data)):
    for j in range(4):
        if i % 2 == 1:
            table[(i, j)].set_facecolor('#f5f5f5')

ax9.set_title('Performance Summary (300 Episodes)', fontsize=12, fontweight='bold', pad=20)

plt.suptitle('CartPole RL Methods Comparison: Q-Learning vs Policy Gradient vs LQR', 
             fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('three_way_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: three_way_comparison.png")
plt.close()

# Print Summary
print("\n" + "="*70)
print("FINAL COMPARISON SUMMARY")
print("="*70)

print(f"\n1. Q-LEARNING (Model-Free, Value-Based)")
print(f"   • Final 100-episode average: {q_final:.1f}")
print(f"   • Success rate: {q_success_rate:.1f}%")
print(f"   • Total reward: {np.sum(q_res['rewards']):.0f}")
print(f"   • Learning: Discretized state space, epsilon-greedy")

print(f"\n2. POLICY GRADIENT (Model-Free, Policy-Based)")
print(f"   • Final 100-episode average: {pg_final:.1f}")
print(f"   • Success rate: {pg_success_rate:.1f}%")
print(f"   • Total reward: {np.sum(pg_res['rewards']):.0f}")
print(f"   • Learning: Direct policy optimization, REINFORCE")

print(f"\n3. MODEL-BASED LQR (Optimal Control)")
print(f"   • Final 100-episode average: {lqr_final:.1f}")
print(f"   • Success rate: {lqr_success_rate:.1f}%")
print(f"   • Total reward: {np.sum(lqr_res['rewards']):.0f}")
print(f"   • Learning: Analytical solution, known dynamics")

print("\n" + "="*70)
print("✓ Three-way comparison complete!")
print("="*70)
