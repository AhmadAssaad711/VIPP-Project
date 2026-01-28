"""
Quick launcher for CartPole animations
Simply run: python run_animation.py
"""
import sys
import os

print("\n" + "="*70)
print("CARTPOLE RL METHODS ANIMATION")
print("="*70)
print("\nThis will show sequential animations of:")
print("  1. Policy Gradient (REINFORCE) - Purple (200 episodes)")
print("  2. Q-Learning (Value-Based) - Blue (200 episodes)")  
print("  3. Model-Based LQR (Optimal) - Red (50 episodes)")
print("\nWatch the learning agents improve over time!")
print("Close the Pygame window to exit early.")
print("="*70 + "\n")

# Import and run the main animation
try:
    from animate_all_methods import main
    main()
except KeyboardInterrupt:
    print("\n\nAnimation interrupted by user.")
    sys.exit(0)
except Exception as e:
    print(f"\n\nError running animation: {e}")
    print("\nMake sure you have all dependencies installed:")
    print("  pip install gymnasium numpy pygame")
    sys.exit(1)

print("\nAnimation complete!")
