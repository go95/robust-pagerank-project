import numpy as np
import math
import tarfile
import pandas as pd

small_graph = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
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

def maxnorm(x):
    norm = 0
    for i in range(len(x)):
        if (abs(x[i]) > norm):
            norm = abs(x[i])
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
    
def phi(x, M):
    x1 = []
    for i in range(len(x)):
        x1.append((M*(np.matrix(x).T)).getA1()[i]-x[i])
    return (euclnorm(x1)+eps*euclnorm(x))

def pagerank (M):
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
    
    M = M.T    
    
    weights = np.zeros((M.shape[0], 1))
    conv_func = np.array([]) #if it is more convenient to you, you can use simple python list here
    
    n = len(M)
     
    e = [1/n for i in range(n)]
    
    x = []#iterations to optimum
    conv_norm = []#deviations from optimum
    x.append(e)
    k = 1
    
        
    x_new = (1-(1.0/(k+1)))*M*(np.matrix(x[k-1]).T) + (1/(k+1))*(np.matrix(e).T)
    x.append(x_new.getA1())
    k+=1  
    
    while(phi(x[k-1], M) < phi(x[k-2], M)):
        x_new = (1-1/(k+1))*M*(np.matrix(x[k-1]).T) + (1/(k+1))*(np.matrix(e).T)
        x.append(x_new.getA1())
        k+=1
    
    for i in range(0, k-1):
        conv_norm.append(maxnormdiffer(x[k-1], x[i]))    
    
    weights = x[k-1]
    
    return (weights, conv_norm)


print(pagerank(p2p_gnut_graph))