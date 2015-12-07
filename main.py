import numpy as np

small_graph = np.array([[0,0,1,0,0,0,0,0,0,0,0,0],
                        [1,0,0,0,0,0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0.5,0.5,0,0,0,0,0,0],
                        [0,0,0,0.5,0,0.5,0,0,0,0,0,0],
                        [0,0,0,0,1,0,0,0,0,0,0,0],
                        [0,0,0,0,0.5,0.5,0,0,0,0,0,0],
                        [0,0,0,0,0,0.5,0.5,0,0,0,0,0],
                        [0,0,1.0/3,0,0,0,0,1.0/3,0,0,0,1.0/3],
                        [0,0,0,0,0,0,0,0.5,0.5,0,0,0],
                        [0,0,0,0,0,0,0,0.5,0,0.5,0,0],
                        [0,0,0,0,0,0,0,0.5,0.5,0,0,0]])

def eigenvector (self, M):
    """
    The function gets a contingency matrix of a graph and computes the left eigenvector associated
    with eigenvalue 1. It implements an iterative approach and also returns the arrays of deviations
    from the optimum on every step to evaluate the rate of convergence
    args:
        M - contingency matrix of a graph, scaled to be a probabilistic matrix (numpy nxn array)

    return (weights, conv_func, conv_norm)
        weights - the eigenvector of interest (numpy n*1 vector)
        conv_func - numpy vector with deviations from the optimum, described in terms of target function
        conv_norm - numpy vector with deviations from the optimum, described in terms of norm (euklid?)
    """
    weights = np.zeros((M.shape[0], 1))
    conv_func = np.array([]) #if it is more convenient to you, you can use simple python list here
    conv_norm = np.array([]) #if it is more convenient to you, you can use simple python list here
    return (weights, conv_func, conv_norm)

def pagerank (self, M):
    """
    The function gets a contingency matrix of a graph and computes the pagerank score with alpha = 0.85
    It implements an iterative approach and also returns the arrays of deviations
    from the optimum on every step to evaluate the rate of convergence
    args:
        M - contingency matrix of a graph, scaled to be a probabilistic matrix (numpy nxn array)

    return (weights, conv_func, conv_norm)
        weights - pagerank scores (numpy n*1 vector)
        conv_func - numpy vector with deviations from the optimum, described in terms of target function
        conv_norm - numpy vector with deviations from the optimum, described in terms of norm (euklid?)
    """
    alpha = 0.85
    weights = np.zeros((M.shape[0], 1))
    conv_func = np.array([]) #if it is more convenient to you, you can use simple python list here
    conv_norm = np.array([]) #if it is more convenient to you, you can use simple python list here
    return (weights, conv_func, conv_norm)

# note: if it is more convenient to you, you can use one function with additional parameter instead of two functions for robust pagerank

def robust_pagerank_euklid_test (self, M): #TODO specify the implementation of the algorithm in the docs
    """
    This is a wrapper function to test peformance of our algorithm for robust pagerank against the existing one.
    Here we use euklid norms both for a and b.
    It implements an iterative approach and also returns the arrays of deviations
    from the optimum on every step to evaluate the rate of convergence

    The original algorithm is that of ...

    args:
        M - contingency matrix of a graph, scaled to be a probabilistic matrix (numpy nxn array)

    return (weights, conv_func, conv_norm)
        weights - the pagerank_score (numpy n*1 vector)
        conv_func - numpy vector with deviations from the optimum, described in terms of target function
        conv_norm - numpy vector with deviations from the optimum, described in terms of norm (euklid?)
    """
    weights = np.zeros((M.shape[0], 1))
    conv_func = np.array([]) #if it is more convenient to you, you can use simple python list here
    conv_norm = np.array([]) #if it is more convenient to you, you can use simple python list here
    return (weights, conv_func, conv_norm)

def robust_pagerank_max_test (self, M): #TODO specify the implementation of the algorithm in the docs
    """
    This is a wrapper function to test peformance of our algorithm for robust pagerank against the existing one.
    Here we use max norms both for a and b.
    It implements an iterative approach and also returns the arrays of deviations
    from the optimum on every step to evaluate the rate of convergence

    The original algorithm is that of ...

    args:
        M - contingency matrix of a graph, scaled to be a probabilistic matrix (numpy nxn array)

    return (weights, conv_func, conv_norm)
        weights - the pagerank_score (numpy n*1 vector)
        conv_func - numpy vector with deviations from the optimum, described in terms of target function
        conv_norm - numpy vector with deviations from the optimum, described in terms of norm (euklid?)
    """
    weights = np.zeros((M.shape[0], 1))
    conv_func = np.array([]) #if it is more convenient to you, you can use simple python list here
    conv_norm = np.array([]) #if it is more convenient to you, you can use simple python list here
    return (weights, conv_func, conv_norm)


def robust_alternative_test (self, M): #TODO Please, specify the algorithm and it's implementation in the docs
    """
    This is a wrapper function to test peformance of our algorithm for ... against the existing one.
    Here we use max norms both for a and b.
    It implements an iterative approach and also returns the arrays of deviations
    from the optimum on every step to evaluate the rate of convergence

    The original algorithm is that of ...

    args:
        M - contingency matrix of a graph, scaled to be a probabilistic matrix (numpy nxn array)

    return (weights, conv_func, conv_norm)
        weights - the eigenvector of interest (numpy n*1 vector)
        conv_func - numpy vector with deviations from the optimum, described in terms of target function
        conv_norm - numpy vector with deviations from the optimum, described in terms of norm (euklid?)
    """
    weights = np.zeros((M.shape[0], 1))
    conv_func = np.array([]) #if it is more convenient to you, you can use simple python list here
    conv_norm = np.array([]) #if it is more convenient to you, you can use simple python list here
    return (weights, conv_func, conv_norm)

print small_graph
