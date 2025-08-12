import numpy as np

def train_test_split(X, y, test_size=0.25, random_state=None):
    """
    Split arrays or matrices into random train and test subsets.
    
    Parameters:
    -----------
    X : array-like
        Training data
    y : array-like
        Target values
    test_size : float, default=0.25
        Proportion of the dataset to include in the test split
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : arrays
        Train and test splits
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    X = np.array(X)
    y = np.array(y)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    # Create indices and shuffle them
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    # Split indices
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # Split the data
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test
