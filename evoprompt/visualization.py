"""
Visualization utilities for the evolutionary prompt optimizer.
"""

import matplotlib.pyplot as plt


def plot_evolution_history(history):
    """
    Plot the evolution history.
    
    Args:
        history: List of dictionaries with statistics for each generation
    """
    generations = [entry["generation"] for entry in history]
    best_scores = [entry["best_score"] for entry in history]
    avg_scores = [entry["avg_score"] for entry in history]
    population_sizes = [entry["population_size"] for entry in history]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot scores
    ax1.plot(generations, best_scores, 'b-', label='Best Score')
    ax1.plot(generations, avg_scores, 'g-', label='Average Score')
    ax1.set_ylabel('Score')
    ax1.set_title('Evolution of Prompt Scores')
    ax1.legend()
    ax1.grid(True)
    
    # Plot population size
    ax2.plot(generations, population_sizes, 'r-', label='Population Size')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Population Size')
    ax2.set_title('Evolution of Population Size')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
