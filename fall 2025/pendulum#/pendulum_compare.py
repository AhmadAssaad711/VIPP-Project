"""
CartPole RL Methods Comparison

This script trains and compares three reinforcement learning approaches:
1. Q-Learning (Model-Free, Value-Based)
2. Policy Gradient (Model-Free, Policy-Based)  
3. Model-Based LQR (Model-Based Control)

Each method is run across multiple random seeds to estimate variance.
Comprehensive plots and statistics are generated for comparison.

Usage:
    python pendulum_compare.py [--n_episodes 500] [--n_seeds 5]
"""

import numpy as np
import argparse
from datetime import datetime

# Import our modules
from q_learning import train_q_learning
from policy_gradient import train_policy_gradient
from model_based import train_model_based
from plots import (plot_single_run, plot_multi_seed_comparison, 
                   print_summary_statistics)


def print_header():
    """Print welcome header."""
    print("\n" + "="*70)
    print(" " * 15 + "CARTPOLE RL METHODS COMPARISON")
    print("="*70)
    print("\nComparing Three Approaches:")
    print("  1. Q-Learning: Model-Free, Value-Based (Discretized Q-table)")
    print("  2. Policy Gradient: Model-Free, Policy-Based (Neural network)")
    print("  3. Model-Based LQR: Uses known dynamics (Analytical solution)")
    print("\nEach method represents a different paradigm in RL:")
    print("  • Model-Free learns from experience without system model")
    print("  • Model-Based exploits known dynamics for instant optimization")
    print("  • Value-Based learns Q(s,a), Policy-Based learns π(a|s) directly")
    print("="*70 + "\n")


def run_single_comparison(n_episodes=500, seed=42):
    """
    Run all three methods once with a single seed.
    
    Args:
        n_episodes: Number of episodes to train/evaluate
        seed: Random seed
    
    Returns:
        Dict of results for each method
    """
    print(f"\n{'='*70}")
    print(f"SINGLE RUN COMPARISON (Seed: {seed}, Episodes: {n_episodes})")
    print(f"{'='*70}\n")
    
    results = {}
    
    # Method 1: Q-Learning
    print("\n[1/3] Training Q-Learning...")
    results['Q-Learning'] = train_q_learning(
        n_episodes=n_episodes, 
        seed=seed, 
        verbose=True
    )
    
    # Method 2: Policy Gradient
    print("\n[2/3] Training Policy Gradient...")
    results['Policy Gradient'] = train_policy_gradient(
        n_episodes=n_episodes,
        seed=seed,
        verbose=True
    )
    
    # Method 3: Model-Based LQR
    print("\n[3/3] Evaluating Model-Based LQR...")
    results['Model-Based (LQR)'] = train_model_based(
        n_episodes=n_episodes,
        seed=seed,
        verbose=True
    )
    
    return results


def run_multi_seed_comparison(n_episodes=500, n_seeds=5, base_seed=42):
    """
    Run all methods multiple times with different seeds.
    
    Args:
        n_episodes: Number of episodes per run
        n_seeds: Number of different seeds to try
        base_seed: Base random seed
    
    Returns:
        Dict mapping method names to lists of results
    """
    print(f"\n{'='*70}")
    print(f"MULTI-SEED COMPARISON ({n_seeds} seeds, {n_episodes} episodes each)")
    print(f"{'='*70}\n")
    
    all_results = {
        'Q-Learning': [],
        'Policy Gradient': [],
        'Model-Based (LQR)': []
    }
    
    seeds = [base_seed + i * 100 for i in range(n_seeds)]
    
    for i, seed in enumerate(seeds):
        print(f"\n{'─'*70}")
        print(f"SEED {i+1}/{n_seeds} (seed={seed})")
        print(f"{'─'*70}")
        
        # Q-Learning
        print(f"\n[Seed {i+1}] Q-Learning...")
        result_q = train_q_learning(n_episodes=n_episodes, seed=seed, verbose=False)
        all_results['Q-Learning'].append(result_q)
        print(f"  Final avg: {np.mean(result_q['rewards'][-100:]):.1f}")
        
        # Policy Gradient
        print(f"[Seed {i+1}] Policy Gradient...")
        result_pg = train_policy_gradient(n_episodes=n_episodes, seed=seed, verbose=False)
        all_results['Policy Gradient'].append(result_pg)
        print(f"  Final avg: {np.mean(result_pg['rewards'][-100:]):.1f}")
        
        # Model-Based
        print(f"[Seed {i+1}] Model-Based LQR...")
        result_mb = train_model_based(n_episodes=n_episodes, seed=seed, verbose=False)
        all_results['Model-Based (LQR)'].append(result_mb)
        print(f"  Final avg: {np.mean(result_mb['rewards'][-100:]):.1f}")
    
    return all_results


def print_method_comparison():
    """Print detailed comparison of the three methods."""
    print("\n" + "="*70)
    print("METHOD COMPARISON: Key Differences")
    print("="*70)
    
    comparison = """
╔═══════════════════╦══════════════════╦══════════════════╦══════════════════╗
║                   ║   Q-LEARNING     ║ POLICY GRADIENT  ║  MODEL-BASED LQR ║
╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
║ Paradigm          ║ Model-Free       ║ Model-Free       ║ Model-Based      ║
║                   ║ Value-Based      ║ Policy-Based     ║ Optimal Control  ║
╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
║ What it learns    ║ Q(s,a) values    ║ π_θ(a|s) policy  ║ Nothing (instant)║
╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
║ Representation    ║ Discrete Q-table ║ Neural network   ║ Gain matrix K    ║
║                   ║ (6×6×12×12×2)    ║ (128-64-2)       ║ (1×4)            ║
╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
║ State space       ║ Discretized bins ║ Continuous       ║ Continuous       ║
╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
║ Update rule       ║ TD(0) Bellman    ║ REINFORCE        ║ Riccati equation ║
║                   ║ Q←Q+α[r+γQ'-Q]   ║ ∇J(θ)=∇log π·G   ║ Analytical       ║
╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
║ Exploration       ║ ε-greedy decay   ║ Stochastic policy║ Deterministic    ║
╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
║ Requires model    ║ No               ║ No               ║ Yes (A, B known) ║
╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
║ Sample efficiency ║ Medium (~200)    ║ Low (~300)       ║ Infinite (0)     ║
╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
║ Convergence       ║ Guaranteed*      ║ Local optimum    ║ Global optimum** ║
╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
║ Scalability       ║ Poor (curse of   ║ Good (function   ║ Limited (linear  ║
║                   ║ dimensionality)  ║ approximation)   ║ systems only)    ║
╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
║ Advantages        ║ • Simple         ║ • Continuous     ║ • Instant        ║
║                   ║ • Proven theory  ║ • Stochastic     ║ • Optimal        ║
║                   ║ • Off-policy     ║ • Scalable       ║ • Stable         ║
╠═══════════════════╬══════════════════╬══════════════════╬══════════════════╣
║ Disadvantages     ║ • Discretization ║ • High variance  ║ • Needs model    ║
║                   ║ • Memory grows   ║ • Sample hungry  ║ • Linear only    ║
║                   ║ • Slow for large ║ • Local optima   ║ • No adaptation  ║
╚═══════════════════╩══════════════════╩══════════════════╩══════════════════╝

* With infinite visits to all state-action pairs
** For systems matching linear-quadratic assumptions
"""
    print(comparison)
    
    print("\nKEY TAKEAWAYS:")
    print("─" * 70)
    print("1. Model-Based methods are optimal BUT require accurate system models")
    print("2. Model-Free methods learn from scratch BUT need many samples")
    print("3. Q-Learning is simple BUT limited by state discretization")
    print("4. Policy Gradient scales well BUT has high variance")
    print("5. Choice depends on: knowledge, samples, state/action spaces")
    print("="*70 + "\n")


def main():
    """Main comparison pipeline."""
    # Parse arguments
    parser = argparse.ArgumentParser(description='Compare RL methods on CartPole')
    parser.add_argument('--n_episodes', type=int, default=500,
                       help='Number of episodes per run')
    parser.add_argument('--n_seeds', type=int, default=5,
                       help='Number of random seeds for variance estimation')
    parser.add_argument('--single_only', action='store_true',
                       help='Only run single seed comparison')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed')
    args = parser.parse_args()
    
    # Start time
    start_time = datetime.now()
    
    # Print header
    print_header()
    
    # Single run comparison
    print("PHASE 1: Single Run Analysis")
    print("─" * 70)
    single_results = run_single_comparison(
        n_episodes=args.n_episodes,
        seed=args.seed
    )
    
    # Print statistics
    print_summary_statistics(single_results)
    
    # Generate single-run plots
    print("\nGenerating single-run comparison plots...")
    plot_single_run(single_results, save_path='single_run_comparison.png')
    
    # Multi-seed comparison (if requested)
    if not args.single_only and args.n_seeds > 1:
        print("\n\nPHASE 2: Multi-Seed Analysis")
        print("─" * 70)
        multi_results = run_multi_seed_comparison(
            n_episodes=args.n_episodes,
            n_seeds=args.n_seeds,
            base_seed=args.seed
        )
        
        # Aggregate statistics
        print("\n" + "="*70)
        print(f"AGGREGATED STATISTICS ({args.n_seeds} seeds)")
        print("="*70)
        
        for method_name, seed_results in multi_results.items():
            final_avgs = [np.mean(r['rewards'][-100:]) for r in seed_results]
            success_rates = [np.sum(np.array(r['rewards']) > 200) / len(r['rewards']) * 100 
                           for r in seed_results]
            
            print(f"\n{method_name}:")
            print(f"  Mean final reward: {np.mean(final_avgs):.2f} ± {np.std(final_avgs):.2f}")
            print(f"  Mean success rate: {np.mean(success_rates):.1f}% ± {np.std(success_rates):.1f}%")
            print(f"  Best run: {np.max(final_avgs):.2f}")
            print(f"  Worst run: {np.min(final_avgs):.2f}")
        
        print("="*70)
        
        # Generate multi-seed plots
        print("\nGenerating multi-seed comparison plots...")
        plot_multi_seed_comparison(multi_results, save_path='multi_seed_comparison.png')
    
    # Print method comparison table
    print_method_comparison()
    
    # End time
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✓ Total runtime: {elapsed:.1f} seconds")
    print(f"✓ Plots saved:")
    print(f"  • single_run_comparison.png")
    if not args.single_only and args.n_seeds > 1:
        print(f"  • multi_seed_comparison.png")
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
