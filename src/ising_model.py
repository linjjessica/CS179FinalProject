"""
Ising Model Implementation for Movie Recommendations
CS179 Final Project
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

def array_to_dict(x_array):
    """Convert numpy array to dictionary format for pyGMs"""
    return {i: int(x_array[i]) for i in range(len(x_array))}

def independent_log_likelihood(X, pXi):
    """Compute log-likelihood for independent model"""
    ll = 0
    for x in X:
        for i, rating in enumerate(x):
            if rating == 1:
                ll += np.log(pXi[i] + 1e-10)  # Add small epsilon
            else:
                ll += np.log(1 - pXi[i] + 1e-10)
    return ll / len(X)

def learn_ising_structure(X, C_values=[0.1, 0.5, 1.0, 2.0, 5.0]):
    """
    Learn Ising model structure using L1-regularized logistic regression
    
    Parameters:
    -----------
    X : numpy.ndarray
        Binary rating matrix (users x movies)
    C_values : list
        Regularization strengths to try
        
    Returns:
    --------
    dict : Results for each C value containing neighbors, parameters, connectivity
    """
    n_users, n_movies = X.shape
    results = {}
    
    for C in C_values:
        print(f"Learning structure with C={C}...")
        
        # For each movie Xi, learn its neighborhood using L1-regularized logistic regression
        nbrs = [None] * n_movies
        th_ij = [None] * n_movies  
        th_i = np.zeros(n_movies)
        
        X_tmp = X.copy()
        
        for i in range(n_movies):
            # Remove movie i from features (can't predict itself)
            X_features = np.delete(X_tmp, i, axis=1)
            y_target = X_tmp[:, i]  # Predict movie i
            
            # Fit L1-regularized logistic regression
            lr = LogisticRegression(penalty='l1', C=C, solver='liblinear', random_state=42)
            lr.fit(X_features, y_target)
            
            # Find non-zero coefficients (neighbors)
            coef_with_i = np.insert(lr.coef_[0], i, 0)  # Insert 0 for self-connection
            nbrs[i] = np.where(np.abs(coef_with_i) > 1e-6)[0]
            nbrs[i] = nbrs[i][nbrs[i] != i]  # Remove self-connection
            
            # Store parameters (divide by 2 for Ising parameterization)
            th_ij[i] = coef_with_i[nbrs[i]] / 2.0
            th_i[i] = lr.intercept_[0] / 2.0
        
        # Calculate average connectivity
        avg_connectivity = np.mean([len(nn) for nn in nbrs])
        
        results[C] = {
            'nbrs': nbrs,
            'th_ij': th_ij, 
            'th_i': th_i,
            'avg_connectivity': avg_connectivity
        }
        
        print(f"Average connectivity: {avg_connectivity:.2f} ± {np.std([len(nn) for nn in nbrs]):.2f}")
    
    return results

def build_ising_model_pygms(nbrs, th_ij, th_i, n_movies):
    """
    Build Ising model from learned parameters using pyGMs
    
    Parameters:
    -----------
    nbrs : list
        Neighbor lists for each movie
    th_ij : list  
        Pairwise parameters
    th_i : numpy.ndarray
        Bias parameters
    n_movies : int
        Number of movies
        
    Returns:
    --------
    pyGMs.GraphModel or None if pyGMs not available
    """
    try:
        import pyGMs as gm
        
        factors = []
        
        # Single-variable factors (bias terms)
        for i in range(n_movies):
            t = th_i[i]
            factor = gm.Factor([gm.Var(i, 2)], [-t, t]).exp()
            factors.append(factor)
        
        # Pairwise factors
        added_edges = set()
        for i in range(n_movies):
            for j_idx, j in enumerate(nbrs[i]):
                if j < i and (j, i) not in added_edges:  # Avoid duplicate edges
                    t = th_ij[i][j_idx]
                    scope = [gm.Var(i, 2), gm.Var(int(j), 2)]
                    factor = gm.Factor(scope, [[t, -t], [-t, t]]).exp()
                    factors.append(factor)
                    added_edges.add((i, j))
        
        model = gm.GraphModel(factors)
        model.makeMinimal()
        return model
        
    except ImportError:
        print("pyGMs not available")
        return None

def compute_pseudolikelihood(X, nbrs, th_ij, th_i):
    """
    Compute pseudo-likelihood of the data
    
    Parameters:
    -----------
    X : numpy.ndarray
        Binary rating matrix
    nbrs : list
        Neighbor lists  
    th_ij : list
        Pairwise parameters
    th_i : numpy.ndarray
        Bias parameters
        
    Returns:
    --------
    float : Pseudo-likelihood
    """
    total_ll = 0
    n_samples, n_vars = X.shape
    
    for i in range(n_vars):  # For each variable
        for j in range(n_samples):  # For each sample
            # Compute P(Xi | X_neighbors)
            linear_term = th_i[i]  # Bias term
            
            # Add contributions from neighbors
            for k, neighbor in enumerate(nbrs[i]):
                linear_term += th_ij[i][k] * X[j, neighbor]
            
            # Logistic function: P(Xi=1 | neighbors) = sigmoid(linear_term)
            prob_1 = 1 / (1 + np.exp(-2 * linear_term))
            
            # Add log probability
            if X[j, i] == 1:
                total_ll += np.log(prob_1 + 1e-10)
            else:
                total_ll += np.log(1 - prob_1 + 1e-10)
    
    return total_ll / (n_samples * n_vars)

def select_best_model(X, results):
    """
    Select best model based on pseudo-likelihood
    
    Parameters:
    -----------
    X : numpy.ndarray
        Training data
    results : dict
        Results from learn_ising_structure
        
    Returns:
    --------
    tuple : (best_C, best_pseudo_ll)
    """
    best_C = None
    best_pseudo_ll = -np.inf
    
    for C in results.keys():
        nbrs = results[C]['nbrs']
        th_ij = results[C]['th_ij']
        th_i = results[C]['th_i']
        
        pseudo_ll = compute_pseudolikelihood(X, nbrs, th_ij, th_i)
        print(f"C={C}: Pseudo-likelihood = {pseudo_ll:.4f}")
        
        if pseudo_ll > best_pseudo_ll:
            best_pseudo_ll = pseudo_ll
            best_C = C
    
    return best_C, best_pseudo_ll

def analyze_dependencies(results, C, movie_titles, top_k=15):
    """
    Analyze strongest learned dependencies
    
    Parameters:
    -----------
    results : dict
        Structure learning results
    C : float
        Regularization parameter to analyze
    movie_titles : list
        Movie title strings
    top_k : int
        Number of top dependencies to return
        
    Returns:
    --------
    list : List of (weight, i, j, orig_weight) tuples
    """
    nbrs = results[C]['nbrs']
    th_ij = results[C]['th_ij']
    
    connections = []
    for i in range(len(movie_titles)):
        for j_idx, j in enumerate(nbrs[i]):
            if j > i:  # Avoid duplicates
                weight = abs(th_ij[i][j_idx])
                connections.append((weight, i, j, th_ij[i][j_idx]))
    
    connections.sort(reverse=True)
    return connections[:top_k]