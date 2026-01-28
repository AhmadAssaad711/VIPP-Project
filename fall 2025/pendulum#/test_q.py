"""Simple test of Q-Learning"""
from q_learning import train_q_learning

print("Testing Q-Learning...")
results = train_q_learning(n_episodes=100, seed=42, verbose=True)
print(f"\n✓ Completed {len(results['rewards'])} episodes")
print(f"✓ Final average: {sum(results['rewards'][-20:])/20:.1f}")
