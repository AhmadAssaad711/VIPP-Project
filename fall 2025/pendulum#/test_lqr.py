"""Simple test of Model-Based LQR"""
from model_based import train_model_based

print("Testing Model-Based LQR...")
results = train_model_based(n_episodes=100, seed=42, verbose=True)
print(f"\n✓ Completed {len(results['rewards'])} episodes")
print(f"✓ Final average: {sum(results['rewards'][-20:])/20:.1f}")
