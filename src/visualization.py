"""
Visualization Functions for Movie Recommendation Analysis
CS179 Final Project
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from sklearn.metrics import confusion_matrix

def plot_rating_distribution(ratings_df):
    """Plot rating distribution and ratings per user"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    ratings_df['rating'].hist(bins=20, edgecolor='black', alpha=0.7)
    plt.title('Rating Distribution')
    plt.xlabel('Rating')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    ratings_per_user = ratings_df.groupby('userId').size()
    ratings_per_user.hist(bins=30, edgecolor='black', alpha=0.7)
    plt.title('Ratings per User')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Number of Users')
    
    plt.tight_layout()
    plt.show()

def plot_movie_correlation_matrix(X, movie_titles=None):
    """Plot correlation matrix between movies"""
    movie_corrs = np.corrcoef(X.T)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(movie_corrs, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Movie-Movie Correlation Matrix')
    plt.xlabel('Movie Index')
    plt.ylabel('Movie Index')
    
    if movie_titles and len(movie_titles) <= 20:  # Only show labels if not too many
        plt.xticks(range(len(movie_titles)), [title[:10] for title in movie_titles], rotation=45)
        plt.yticks(range(len(movie_titles)), [title[:10] for title in movie_titles])
    
    plt.tight_layout()
    plt.show()

def plot_dependency_graph(nbrs, th_ij, movie_titles, weight_threshold=0.1, C=None):
    """
    Plot graph of learned movie dependencies
    
    Parameters:
    -----------
    nbrs : list
        Neighbor lists for each movie
    th_ij : list
        Pairwise parameters
    movie_titles : list
        Movie title strings
    weight_threshold : float
        Minimum edge weight to display
    C : float
        Regularization parameter (for title)
    """
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for i, title in enumerate(movie_titles):
        G.add_node(i, title=title)
    
    # Add edges from learned structure
    for i in range(len(movie_titles)):
        for j_idx, j in enumerate(nbrs[i]):
            if j > i:  # Avoid duplicate edges
                weight = abs(th_ij[i][j_idx])
                if weight > weight_threshold:  # Only show strong connections
                    G.add_edge(i, j, weight=weight)
    
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Plot the graph
    plt.figure(figsize=(15, 12))
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw edges with thickness proportional to weight
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]
    nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], alpha=0.6)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                          node_size=500, alpha=0.8)
    
    # Add labels
    labels = {i: title[:10] for i, title in enumerate(movie_titles)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    title_str = f"Movie Dependency Graph"
    if C is not None:
        title_str += f" (C={C})"
    title_str += f"\nEdge thickness ∝ |correlation strength|"
    
    plt.title(title_str)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_prediction_comparison(methods, y_true):
    """
    Plot comprehensive prediction evaluation results
    
    Parameters:
    -----------
    methods : dict
        Results from compare_methods()
    y_true : numpy.ndarray
        True labels
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Accuracy comparison
    method_names = list(methods.keys())
    accuracies = [methods[name]['accuracy'] for name in method_names]
    
    axes[0, 0].bar(method_names, accuracies, color=['lightcoral', 'lightblue', 'lightgreen'])
    axes[0, 0].set_title('Prediction Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    
    for i, acc in enumerate(accuracies):
        axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center')
    
    # Confusion matrix for best method (usually Ising)
    best_method = max(methods.keys(), key=lambda k: methods[k]['accuracy'])
    cm = methods[best_method]['confusion_matrix']
    
    im = axes[0, 1].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[0, 1].set_title(f'{best_method} Confusion Matrix')
    axes[0, 1].set_xlabel('Predicted')
    axes[0, 1].set_ylabel('True')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(['Dislike', 'Like'])
    axes[0, 1].set_yticks([0, 1])
    axes[0, 1].set_yticklabels(['Dislike', 'Like'])
    
    # Add text annotations to confusion matrix
    for i in range(2):
        for j in range(2):
            axes[0, 1].text(j, i, str(cm[i, j]), ha='center', va='center')
    
    # Prediction confidence analysis (if available)
    if 'probabilities' in methods.get('Ising Model', {}):
        probs = methods['Ising Model']['probabilities']
        axes[1, 0].hist(probs, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Ising Model Prediction Confidence')
        axes[1, 0].set_xlabel('P(Like)')
        axes[1, 0].set_ylabel('Frequency')
        
        # Accuracy vs confidence
        pred_ising = methods['Ising Model']['predictions']
        confidence = np.abs(probs - 0.5)  # Distance from 0.5
        correct = (pred_ising == y_true)
        
        bins = np.linspace(0, 0.5, 6)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_accuracies = []
        
        for i in range(len(bins)-1):
            mask = (confidence >= bins[i]) & (confidence < bins[i+1])
            if mask.sum() > 0:
                bin_accuracies.append(correct[mask].mean())
            else:
                bin_accuracies.append(0)
        
        axes[1, 1].plot(bin_centers, bin_accuracies, 'o-', color='green')
        axes[1, 1].set_title('Accuracy vs Prediction Confidence')
        axes[1, 1].set_xlabel('Confidence (|P(Like) - 0.5|)')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Alternative plot if no probabilities available
        method_f1s = [methods[name]['f1'] for name in method_names]
        axes[1, 0].bar(method_names, method_f1s, color=['lightcoral', 'lightblue', 'lightgreen'])
        axes[1, 0].set_title('F1 Score Comparison')
        axes[1, 0].set_ylabel('F1 Score')
        
        # Precision vs Recall
        precisions = [methods[name]['precision'] for name in method_names]
        recalls = [methods[name]['recall'] for name in method_names]
        
        for i, name in enumerate(method_names):
            axes[1, 1].scatter(recalls[i], precisions[i], s=100, label=name)
        
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_ylabel('Precision')
        axes[1, 1].set_title('Precision vs Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_connectivity_analysis(results):
    """Plot connectivity vs regularization parameter"""
    C_values = list(results.keys())
    connectivities = [results[C]['avg_connectivity'] for C in C_values]
    
    plt.figure(figsize=(8, 5))
    plt.plot(C_values, connectivities, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Regularization Parameter (C)')
    plt.ylabel('Average Connectivity')
    plt.title('Graph Connectivity vs Regularization')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    for i, (C, conn) in enumerate(zip(C_values, connectivities)):
        plt.annotate(f'{conn:.1f}', (C, conn), textcoords="offset points", 
                    xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.show()

def plot_top_dependencies(connections, movie_titles, top_k=15):
    """
    Plot strongest learned dependencies as horizontal bar chart
    
    Parameters:
    -----------
    connections : list
        List of (weight, i, j, orig_weight) tuples
    movie_titles : list
        Movie title strings
    top_k : int
        Number of dependencies to show
    """
    plt.figure(figsize=(12, 8))
    
    weights = [conn[0] for conn in connections[:top_k]]
    labels = []
    colors = []
    
    for weight, i, j, orig_weight in connections[:top_k]:
        sign = "+" if orig_weight > 0 else "−"
        label = f"{movie_titles[i][:15]} {sign} {movie_titles[j][:15]}"
        labels.append(label)
        colors.append('green' if orig_weight > 0 else 'red')
    
    y_pos = np.arange(len(labels))
    
    plt.barh(y_pos, weights, color=colors, alpha=0.7)
    plt.yticks(y_pos, labels)
    plt.xlabel('Dependency Strength')
    plt.title(f'Top {top_k} Learned Movie Dependencies')
    plt.gca().invert_yaxis()  # Highest weights at top
    
    # Add value labels
    for i, weight in enumerate(weights):
        plt.text(weight + 0.02, i, f'{weight:.2f}', va='center')
    
    plt.tight_layout()
    plt.show()

def create_summary_plots(X, movie_titles, results, best_C, methods, y_true, connections):
    """Create all summary visualizations"""
    print("Creating summary visualizations...")
    
    # 1. Movie correlation matrix
    plot_movie_correlation_matrix(X, movie_titles if len(movie_titles) <= 20 else None)
    
    # 2. Dependency graph
    best_results = results[best_C]
    plot_dependency_graph(best_results['nbrs'], best_results['th_ij'], 
                         movie_titles, weight_threshold=0.1, C=best_C)
    
    # 3. Prediction comparison
    plot_prediction_comparison(methods, y_true)
    
    # 4. Connectivity analysis
    plot_connectivity_analysis(results)
    
    # 5. Top dependencies
    plot_top_dependencies(connections, movie_titles, top_k=15)
    
    print("All visualizations complete!")