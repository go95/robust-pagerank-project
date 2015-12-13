import numpy as np
import pandas as pd
import tarfile
from cvxpy import *

# small graph for algo testing

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



alpha = 0.85 #damping factor
eps = 0.7

# open compressed p2p_gnut archive

comp_gnut = tarfile.open('p2p_gnut.tar.gz')
uncomp_gnut = comp_gnut.extractall()
comp_gnut.close()

# load p2p_gnut graph into panda dataframe then numpy array

df_gnut = pd.read_csv('p2p_gnut.csv', header=None)

p2p_gnut_graph = df_gnut.iloc[:,:].values

def maxnorm(x1, x2):
    norm = 0
    for i in range(len(x1)):
        if (abs(x1[i]-x2[i]) > norm):
            norm = abs(x1[i]-x2[i])
    return norm

def maxnormdiffer(x1, x2):
    norm = 0
    for i in range(len(x1)):
        if (abs(x1[i]-x2[i]) > norm):
            norm = abs(x1[i]-x2[i])
    return norm
    
def euclnorm(x):
    norm = 0
    for i in range(len(x)):
        norm += abs(x[i])**(2)
    return math.sqrt(norm)

accuracy = 10**(-6)

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
    #M - hyperlink matrix

    n = len(M)

    v = [] #personalization vector
    for i in range(n):
        v.append(1/n)

    d = []#dangling nodes column
    for i in range(n):
        iszero = 1
        for j in range(n):
            if (M[i, j] != 0): iszero = 0
        d.append(iszero)
    d = np.matrix(d).T

    S = M + d*v #matrix with fixed dangling nodes

    e = [] # column of all 1's
    for i in range(n):
        e.append(1)
    e = np.matrix(e).T

    G = alpha*S+(1-alpha)*e*v #Google matrix

    x = []#iterations to optimum
    conv_norm = []#deviations from optimum
    x.append(v)
    k = 0

    x_new = alpha*(np.matrix(x[k])*M)+alpha*(np.matrix(x[k])*d)*v+(1-alpha)*np.matrix(v)
    x.append(x_new.getA1())
    k+=1

    while(maxnorm(x[k], x[k-1]) > accuracy):
        x_new = alpha*(np.matrix(x[k])*M)+alpha*(np.matrix(x[k])*d)*v+(1-alpha)*np.matrix(v)
        x.append(x_new.getA1())
        k+=1

    for i in range(0, k-1):
        conv_norm.append(maxnorm(x[k], x[i]))

    weights = x[k]

    return (weights, conv_norm)
    
def DW(x, beta): # x - column, beta - constant
    
    return np.matrix([0]*len(x)).T

def DU(y, delta): # y - column, delta - constant
    
    return np.matrix([0]*len(y)).T
    
def q_x(x, y, M): #M is column-stochastic, x,y - columns
    v = M.T*(y)-y.T+eps*(x)/(euclnorm(x))
    return v
    
def q_y(x, y, M): #M is column-stochastic, x,y - columns
    return (M*(x)-x)    

def pagerank (M, beta0, delta0): #M is column-stochastic
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
    weigths = []
    conv_norm = [] 
    conv_iter = []#iterations to optimum
    
    N = len(M)
    
    dzeta0 = np.matrix([0]*N).T
    eta0 = np.matrix([0]*N).T
    
    x0 = -DW(dzeta0, beta0)
    y0 = -DU(eta0, delta0)
    
    x = []
    y = []
    dzeta = []
    eta = []
    
    k = 1
    x.append(x0)
    y.append(y0)
    dzeta.append(dzeta0)
    eta.append(eta0)
    
    xaver = np.matrix([0]*N).T #x average        
    
    beta = beta0*math.sqrt(k)
    delta = delta0*math.sqrt(k)
    dzeta.append((dzeta[k-1] + q_x(x[k-1], y[k-1], M)))
    eta.append((eta[k-1] - q_y(x[k-1], y[k-1], M)))
    x.append(-DW(dzeta[k-1], beta))
    y.append(-DU(eta[k-1], delta))
    
    xaver = xaver*(1-1/k)+(x[k-1])*(1/k)
    conv_iter.append([0]*N)    
    conv_iter.append(xaver.getA1())
    k+=1    
    
    while(maxnormdiffer(conv_iter[k-1], conv_iter[k-2]) > 10**(-6)):
        beta = beta0*math.sqrt(k)
        delta = delta0*math.sqrt(k)
        dzeta.append(dzeta[k-1] + q_x(x[k-1], y[k-1], M))
        eta.append(eta[k-1] - q_y(x[k-1], y[k-1], M))
        x.append(-DW(dzeta[k-1], beta))
        y.append(-DU(eta[k-1], delta))
        
        xaver = xaver*(1-1/k)+x[:-1]*(1/n)
        conv_iter.append(xaver.getA1())
        k+=1
        
    weigths = xaver.getA1()
    
    for i in range(k):
        conv_norm.append(maxnormdiffer(conv_iter[k-1], xaver))
    
    return weigths, conv_norm

# note: if it is more convenient to you, you can use one function with additional parameter instead of two functions for robust pagerank

def robust_pagerank_euklid_test (self, M, alpha):
    """
    This is a wrapper function to test peformance of our algorithm for robust pagerank against the existing one.
    Here we use euklid norms both for a and b.

    args:

        M - contingency matrix of a graph, scaled to be a probabilistic matrix (numpy nxn array)

        alpha - regularization parameter

    return weights

        weights - the pagerank_score (numpy n*1 vector)

    """

    dim = M.shape[0]

    # define the problem

    x = Variable(shape)
    objective = Minimize(sum_squares(M*x) + alpha * sum_squares(x))
    constraints = [sum_entries(x) == 1, 0 <= x]
    prob = Problem(objective, constraints)

    # solve the problem

    result = prob.solve()

    weights = x.value

    return weights

def robust_pagerank_max_test (self, alpha): #TODO specify the implementation of the algorithm in the docs
    """
    This is a wrapper function to test peformance of our algorithm for robust pagerank against the existing one.
    Here we use max norms both for a and b.

    args:

        M - contingency matrix of a graph, scaled to be a probabilistic matrix (numpy nxn array)

        alpha - regularization parameter

    return weights

        weights - the pagerank_score (numpy n*1 vector)

    """

    dim = M.shape[0]

    # define the problem

    x = Variable(shape)
    objective = Minimize(norm(M*x, 'inf') + alpha * norm(x, inf))
    constraints = [sum_entries(x) == 1, 0 <= x]
    prob = Problem(objective, constraints)

    # solve the problem

    result = prob.solve()

    weights = x.value

    return weights


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

print(small_graph)
