"""
Evaluation Functions for Movie Recommendation Models
CS179 Final Project
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def create_missing_data_test(X, test_ratio=0.2, random_state=42):
    """
    Create test set by hiding some ratings
    
    Parameters:
    -----------
    X : numpy.ndarray
        Original rating matrix
    test_ratio : float
        Fraction of ratings to hide for testing
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    tuple : (X_train, X_test, test_positions)
    """
    np.random.seed(random_state)
    
    X_train = X.copy()
    X_test = X.copy()
    
    n_users, n_movies = X.shape
    n_test = int(n_users * n_movies * test_ratio)
    
    # Randomly select positions to hide
    test_positions = []
    for _ in range(n_test):
        user_idx = np.random.randint(n_users)
        movie_idx = np.random.randint(n_movies)
        test_positions.append((user_idx, movie_idx))
    
    # Hide ratings in training set
    for user_idx, movie_idx in test_positions:
        X_train[user_idx, movie_idx] = -1  # Mark as missing
    
    return X_train, X_test, test_positions

def predict_independent(X_train, test_positions, pXi):
    """
    Predict using independent model (baseline)
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training data with missing values
    test_positions : list
        List of (user_idx, movie_idx) tuples to predict
    pXi : numpy.ndarray
        Movie popularity probabilities
        
    Returns:
    --------
    numpy.ndarray : Predictions (0 or 1)
    """
    predictions = []
    
    for user_idx, movie_idx in test_positions:
        # Use movie's overall popularity
        prob_like = pXi[movie_idx]
        pred = 1 if prob_like > 0.5 else 0
        predictions.append(pred)
    
    return np.array(predictions)

def predict_user_average(X_train, test_positions):
    """
    Predict using user's average rating (collaborative filtering baseline)
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training data with missing values
    test_positions : list
        List of (user_idx, movie_idx) tuples to predict
        
    Returns:
    --------
    numpy.ndarray : Predictions (0 or 1)
    """
    predictions = []
    
    for user_idx, movie_idx in test_positions:
        user_ratings = X_train[user_idx]
        rated_mask = (user_ratings != -1)
        
        if rated_mask.sum() > 0:
            user_avg = user_ratings[rated_mask].mean()
            pred = 1 if user_avg > 0.5 else 0
        else:
            pred = 1  # Default to "like" if no other ratings
        
        predictions.append(pred)
    
    return np.array(predictions)

def predict_ising_simple(X_train, test_positions, nbrs, th_ij, th_i):
    """
    Predict using learned Ising model structure
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training data with missing values
    test_positions : list
        List of (user_idx, movie_idx) tuples to predict
    nbrs : list
        Neighbor lists for each movie
    th_ij : list
        Pairwise parameters
    th_i : numpy.ndarray
        Bias parameters
        
    Returns:
    --------
    numpy.ndarray : Predictions (0 or 1)
    """
    predictions = []
    
    for user_idx, movie_idx in test_positions:
        # Use the conditional probability P(movie_idx | neighbors)
        user_ratings = X_train[user_idx]
        
        # Compute linear term for this movie
        linear_term = th_i[movie_idx]  # Bias
        
        # Add neighbor contributions
        neighbor_indices = nbrs[movie_idx]
        for k, neighbor_idx in enumerate(neighbor_indices):
            if user_ratings[neighbor_idx] != -1:  # If neighbor is rated
                linear_term += th_ij[movie_idx][k] * user_ratings[neighbor_idx]
        
        # Convert to probability using logistic function
        prob_like = 1 / (1 + np.exp(-2 * linear_term))
        pred = 1 if prob_like > 0.5 else 0
        predictions.append(pred)
    
    return np.array(predictions)

def compute_ising_probabilities(X_train, test_positions, nbrs, th_ij, th_i):
    """
    Compute prediction probabilities for Ising model
    
    Returns probabilities instead of binary predictions for confidence analysis
    """
    probabilities = []
    
    for user_idx, movie_idx in test_positions:
        user_ratings = X_train[user_idx]
        linear_term = th_i[movie_idx]
        
        neighbor_indices = nbrs[movie_idx]
        for k, neighbor_idx in enumerate(neighbor_indices):
            if user_ratings[neighbor_idx] != -1:
                linear_term += th_ij[movie_idx][k] * user_ratings[neighbor_idx]
        
        prob = 1 / (1 + np.exp(-2 * linear_term))
        probabilities.append(prob)
    
    return np.array(probabilities)

def evaluate_predictions(y_true, y_pred, method_name):
    """
    Compute comprehensive evaluation metrics
    
    Parameters:
    -----------
    y_true : numpy.ndarray
        True labels
    y_pred : numpy.ndarray  
        Predicted labels
    method_name : str
        Name of the prediction method
        
    Returns:
    --------
    dict : Dictionary of evaluation metrics
    """
    metrics = {
        'method': method_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    
    return metrics

def compare_methods(X_train, test_positions, y_true, pXi, nbrs, th_ij, th_i):
    """
    Compare all prediction methods and return comprehensive results
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training data with missing values
    test_positions : list
        Test positions
    y_true : numpy.ndarray
        True labels
    pXi : numpy.ndarray
        Movie popularity probabilities
    nbrs, th_ij, th_i : Ising model parameters
        
    Returns:
    --------
    dict : Results for all methods
    """
    methods = {}
    
    # Independent model
    pred_independent = predict_independent(X_train, test_positions, pXi)
    methods['Independent'] = evaluate_predictions(y_true, pred_independent, 'Independent')
    methods['Independent']['predictions'] = pred_independent
    
    # User average  
    pred_user_avg = predict_user_average(X_train, test_positions)
    methods['User Average'] = evaluate_predictions(y_true, pred_user_avg, 'User Average')
    methods['User Average']['predictions'] = pred_user_avg
    
    # Ising model
    pred_ising = predict_ising_simple(X_train, test_positions, nbrs, th_ij, th_i)
    methods['Ising Model'] = evaluate_predictions(y_true, pred_ising, 'Ising Model')
    methods['Ising Model']['predictions'] = pred_ising
    methods['Ising Model']['probabilities'] = compute_ising_probabilities(X_train, test_positions, nbrs, th_ij, th_i)
    
    return methods

def statistical_comparison(pred1, pred2, y_true, name1, name2):
    """
    Compare two prediction methods statistically
    
    Returns contingency table and improvement statistics
    """
    both_correct = np.sum((pred1 == y_true) & (pred2 == y_true))
    method1_better = np.sum((pred1 == y_true) & (pred2 != y_true))
    method2_better = np.sum((pred1 != y_true) & (pred2 == y_true))
    both_wrong = np.sum((pred1 != y_true) & (pred2 != y_true))
    
    net_improvement = method1_better - method2_better
    improvement_rate = net_improvement / len(y_true) * 100
    
    results = {
        'both_correct': both_correct,
        'method1_better': method1_better,
        'method2_better': method2_better, 
        'both_wrong': both_wrong,
        'net_improvement': net_improvement,
        'improvement_rate': improvement_rate
    }
    
    return results

def analyze_prediction_errors(test_positions, y_true, y_pred, movie_titles):
    """
    Analyze which movies are hardest to predict
    
    Returns dictionary of error rates by movie
    """
    movie_errors = {}
    movie_totals = {}
    
    for i, (user_idx, movie_idx) in enumerate(test_positions):
        if movie_idx not in movie_totals:
            movie_totals[movie_idx] = 0
            movie_errors[movie_idx] = 0
        
        movie_totals[movie_idx] += 1
        if y_pred[i] != y_true[i]:
            movie_errors[movie_idx] += 1
    
    # Calculate error rates
    error_rates = {}
    for movie_idx in movie_errors:
        if movie_totals[movie_idx] > 0:
            error_rate = movie_errors[movie_idx] / movie_totals[movie_idx]
            error_rates[movie_idx] = {
                'title': movie_titles[movie_idx],
                'errors': movie_errors[movie_idx],
                'total': movie_totals[movie_idx],
                'error_rate': error_rate
            }
    
    return error_rates

def print_evaluation_summary(methods):
    """Print formatted evaluation results"""
    print("=" * 50)
    print("PREDICTION ACCURACY COMPARISON")
    print("=" * 50)
    
    for method_name, results in methods.items():
        print(f"\n{method_name:15s}:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1 Score:  {results['f1']:.4f}")