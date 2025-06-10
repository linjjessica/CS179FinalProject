"""
Data Preprocessing Functions for MovieLens Analysis
CS179 Final Project
"""

import pandas as pd
import numpy as np

def load_movielens_data(data_path='../data/ml-latest-small/'):
    """
    Load MovieLens dataset
    
    Parameters:
    -----------
    data_path : str
        Path to MovieLens data directory
        
    Returns:
    --------
    tuple : (ratings_df, movies_df)
    """
    ratings_df = pd.read_csv(data_path + 'ratings.csv')
    movies_df = pd.read_csv(data_path + 'movies.csv')
    
    print(f"Dataset Info:")
    print(f"- {len(ratings_df)} ratings")
    print(f"- {ratings_df['userId'].nunique()} users") 
    print(f"- {ratings_df['movieId'].nunique()} movies")
    print(f"- Rating scale: {ratings_df['rating'].min()} to {ratings_df['rating'].max()}")
    
    return ratings_df, movies_df

def filter_data(ratings_df, min_user_ratings=50, min_movie_ratings=20):
    """
    Filter data for computational efficiency
    
    Parameters:
    -----------
    ratings_df : pandas.DataFrame
        Raw ratings data
    min_user_ratings : int
        Minimum ratings per user
    min_movie_ratings : int
        Minimum ratings per movie
        
    Returns:
    --------
    pandas.DataFrame : Filtered ratings
    """
    user_counts = ratings_df.groupby('userId').size()
    movie_counts = ratings_df.groupby('movieId').size()
    
    active_users = user_counts[user_counts >= min_user_ratings].index
    popular_movies = movie_counts[movie_counts >= min_movie_ratings].index
    
    print(f"\nFiltering for computational efficiency:")
    print(f"- Users with ≥{min_user_ratings} ratings: {len(active_users)}")
    print(f"- Movies with ≥{min_movie_ratings} ratings: {len(popular_movies)}")
    
    # Filter the data
    filtered_ratings = ratings_df[
        (ratings_df['userId'].isin(active_users)) & 
        (ratings_df['movieId'].isin(popular_movies))
    ]
    
    print(f"- Filtered dataset: {len(filtered_ratings)} ratings")
    
    return filtered_ratings

def create_rating_matrix(ratings_df, fill_value=0):
    """
    Create user-movie rating matrix
    
    Parameters:
    -----------
    ratings_df : pandas.DataFrame
        Ratings data
    fill_value : float
        Value to fill for missing ratings
        
    Returns:
    --------
    pandas.DataFrame : User-movie matrix
    """
    user_movie_matrix = ratings_df.pivot_table(
        index='userId', 
        columns='movieId', 
        values='rating',
        fill_value=fill_value
    )
    
    print(f"\nRating matrix shape: {user_movie_matrix.shape}")
    sparsity = (user_movie_matrix == 0).sum().sum() / (user_movie_matrix.shape[0] * user_movie_matrix.shape[1])
    print(f"Sparsity: {sparsity:.3f}")
    
    return user_movie_matrix

def binarize_ratings(rating_matrix, threshold=4.0):
    """
    Convert ratings to binary (liked/not liked)
    
    Parameters:
    -----------
    rating_matrix : pandas.DataFrame
        Rating matrix
    threshold : float
        Threshold for "liked" (rating >= threshold)
        
    Returns:
    --------
    pandas.DataFrame : Binary matrix
    """
    binary_matrix = (rating_matrix >= threshold).astype(int)
    # Set unrated items to -1 so we can distinguish from "disliked"
    binary_matrix[rating_matrix == 0] = -1
    
    print(f"\nBinary matrix statistics:")
    print(f"- Liked (1): {(binary_matrix == 1).sum().sum()}")
    print(f"- Disliked (0): {(binary_matrix == 0).sum().sum()}")
    print(f"- Unrated (-1): {(binary_matrix == -1).sum().sum()}")
    
    return binary_matrix

def prepare_ising_data(binary_matrix, n_movies=50, n_users=100):
    """
    Prepare fixed-size matrix for Ising model training
    
    Parameters:
    -----------
    binary_matrix : pandas.DataFrame
        Binary rating matrix
    n_movies : int
        Number of movies to include
    n_users : int
        Number of users to include
        
    Returns:
    --------
    tuple : (X_ising, selected_users, top_movies_list, movie_titles)
    """
    # Get most popular movies by number of ratings
    movie_popularity = (binary_matrix != -1).sum(axis=0).sort_values(ascending=False)
    top_movies_list = movie_popularity.head(n_movies).index.tolist()
    
    # Get users who have rated enough of these top movies
    users_with_enough_ratings = []
    for user_id in binary_matrix.index:
        user_ratings = binary_matrix.loc[user_id, top_movies_list]
        num_rated = (user_ratings != -1).sum()
        if num_rated >= 10:  # User must have rated at least 10 of the top movies
            users_with_enough_ratings.append(user_id)
    
    # Take first n_users
    selected_users = users_with_enough_ratings[:n_users]
    print(f"Selected {len(selected_users)} users and {len(top_movies_list)} movies")
    
    # Create final training matrix
    X_ising = np.zeros((len(selected_users), n_movies))
    for i, user_id in enumerate(selected_users):
        user_data = binary_matrix.loc[user_id, top_movies_list]
        
        # Calculate user's overall preference for imputation
        rated_mask = (user_data != -1)
        if rated_mask.sum() > 0:
            user_pref = (user_data[rated_mask] == 1).mean()
        else:
            user_pref = 0.5
        
        for j, movie_id in enumerate(top_movies_list):
            if user_data[movie_id] != -1:
                # Use actual rating
                X_ising[i, j] = user_data[movie_id]
            else:
                # Impute missing values based on user preference
                X_ising[i, j] = 1 if np.random.random() < user_pref else 0
    
    print(f"\nFinal Ising training matrix: {X_ising.shape}")
    print(f"Fraction of 'likes': {X_ising.mean():.3f}")
    
    return X_ising, selected_users, top_movies_list

def create_movie_titles(movies_df, top_movies_list, max_length=20):
    """
    Create shortened movie titles for display
    
    Parameters:
    -----------
    movies_df : pandas.DataFrame
        Movies metadata
    top_movies_list : list
        List of movie IDs
    max_length : int
        Maximum title length
        
    Returns:
    --------
    list : Shortened movie titles
    """
    movie_titles = []
    for movie_id in top_movies_list:
        title = movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]
        # Shorten title for display
        short_title = title[:max_length] + "..." if len(title) > max_length else title
        movie_titles.append(short_title)
    
    return movie_titles

def get_movie_info(movies_df, top_movies_list):
    """
    Get movie information for analysis
    
    Returns DataFrame with movie metadata for selected movies
    """
    movie_info = movies_df[movies_df['movieId'].isin(top_movies_list)][['movieId', 'title', 'genres']]
    return movie_info.reset_index(drop=True)

def preprocessing_pipeline(data_path='../data/ml-latest-small/', 
                          min_user_ratings=50, 
                          min_movie_ratings=20,
                          binary_threshold=4.0,
                          n_movies=50, 
                          n_users=100):
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    data_path : str
        Path to data directory
    min_user_ratings : int
        Minimum ratings per user for filtering
    min_movie_ratings : int  
        Minimum ratings per movie for filtering
    binary_threshold : float
        Threshold for binary conversion
    n_movies : int
        Number of movies for final matrix
    n_users : int
        Number of users for final matrix
        
    Returns:
    --------
    tuple : All processed data components
    """
    print("="*60)
    print("MOVIELENS DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # 1. Load data
    ratings_df, movies_df = load_movielens_data(data_path)
    
    # 2. Filter data
    filtered_ratings = filter_data(ratings_df, min_user_ratings, min_movie_ratings)
    
    # 3. Create rating matrix
    rating_matrix = create_rating_matrix(filtered_ratings)
    
    # 4. Binarize ratings
    binary_matrix = binarize_ratings(rating_matrix, binary_threshold)
    
    # 5. Prepare Ising data
    X_ising, selected_users, top_movies_list = prepare_ising_data(
        binary_matrix, n_movies, n_users)
    
    # 6. Create movie titles
    movie_titles = create_movie_titles(movies_df, top_movies_list)
    
    # 7. Get movie info
    movie_info = get_movie_info(movies_df, top_movies_list)
    
    print(f"\nPreprocessing complete!")
    print(f"Ready for Ising model analysis with {X_ising.shape[0]} users and {X_ising.shape[1]} movies")
    
    return {
        'X_ising': X_ising,
        'binary_matrix': binary_matrix,
        'rating_matrix': rating_matrix,
        'selected_users': selected_users,
        'top_movies_list': top_movies_list,
        'movie_titles': movie_titles,
        'movie_info': movie_info,
        'movies_df': movies_df,
        'ratings_df': ratings_df,
        'filtered_ratings': filtered_ratings
    }