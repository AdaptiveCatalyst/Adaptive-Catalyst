import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from proxTV import proxTv
from scipy.sparse.linalg import norm

# supported_penalties = ['l1', 'tv']

# This file is for basic functions that will be used in code
# s.t.
# Function $f$
# Regularizer $g$
# Proximal operator of regularizer g
# Gradient of $f$
# Objective function

# We assume all functions' inputs:
# x is a sparse vector of shape (n,1),
# A is a sparse matrices of shape (m,n),
# and b is a sparse vector of shape (m,1)

def f(x, A, b, lam2):
    #print(-A.dot(x).multiply(b).A)
    l = np.log(1 + np.exp(-A.dot(x).multiply(b).A))
    # l = np.log(1 + np.exp(-A.dot(x).T * b))
    m = b.shape[0]
    return np.sum(l) / m + lam2/2 * np.linalg.norm(x.toarray()) ** 2

def f_grad(x, A, b, lam2):
    assert ((b.shape[0] == A.shape[0]) & (x.shape[0] == A.shape[1]))
    assert lam2 >= 0
    denom = csr_matrix(1/(1 + np.exp(A.dot(x).multiply(b).A)))
    g = -(A.multiply(b).multiply(denom).sum(axis=0).T)
    m = b.shape[0]
    return csr_matrix(g) / m + lam2 * x

def lasso(x, A, b):
    C = A.dot(x) - b
    d = C.T.toarray()[0]
    return 0.5*np.linalg.norm(d,2)**2


def lasso_grad(x, A, b):
    return A.T.dot(A.dot(x) - b)
 
 
def l12(x, lam1, block_str):
    '''
    :param block_str: list of blocks
    '''
    xx = x.T.toarray()[0]
    res = 0.
    for j in range(len(block_str)):
        cur_res = 0.
        for i in block_str[j]:
            cur_res += xx[i]**2
        res = res + np.sqrt(cur_res)
    return res
    
   
def l1(x, lam1):
    return lam1 * np.linalg.norm(x.toarray(), ord = 1)

def tv(x, lam1):
    xx = x.T.toarray()[0]
    res = 0
    for i in range(len(xx) - 1):
        res += abs(xx[i] - xx[i+1])
    return lam1*res
    

def F_l1(x, A, b, lam2, lam1):
    assert ((b.shape[0] == A.shape[0]) & (x.shape[0] == A.shape[1]))
    assert ((lam2 >= 0) & (lam1 >= 0))
    return f(x, A, b, lam2) + l1(x, lam1)

def F_tv(x, A, b, lam2, lam1):
    assert ((b.shape[0] == A.shape[0]) & (x.shape[0] == A.shape[1]))
    assert ((lam2 >= 0) & (lam1 >= 0))
    return f(x, A, b, lam2) + tv(x, lam1)

def F_l12(x, A, b, lam2, lam1, block_str):
    assert ((b.shape[0] == A.shape[0]) & (x.shape[0] == A.shape[1]))
    assert ((lam2 >= 0) & (lam1 >= 0))
    return f(x, A, b, lam2) + l12(x, lam1, block_str)

def lasso_F_l1(x, A, b, lam1):
    assert ((b.shape[0] == A.shape[0]) & (x.shape[0] == A.shape[1]))
    assert (lam1 >= 0)
    return lasso(x, A, b) + l1(x, lam1)

def lasso_F_tv(x, A, b, lam1):
    assert ((b.shape[0] == A.shape[0]) & (x.shape[0] == A.shape[1]))
    assert (lam1 >= 0)
    return lasso(x, A, b) + tv(x, lam1)

def lasso_F_l12(x, A, b, lam1, block_str):
    assert ((b.shape[0] == A.shape[0]) & (x.shape[0] == A.shape[1]))
    assert (lam1 >= 0)
    return lasso(x, A, b) + l12(x, lam1, block_str)

def prox_l12(x, gamma, coef, block_str):
    '''
    :param block_str: list of blocks
    '''
    xx = x.T.toarray()[0]
    for j in range(len(block_str)):
        cur_res = 0.
        for i in block_str[j]:
            cur_res += xx[i]**2
        cur_res = np.sqrt(cur_res)
        for i in block_str[j]:
            if cur_res > gamma*coef:
                xx[i] = xx[i] - gamma*coef*xx[i]/cur_res
            else:
                xx[i] = 0
    return csr_matrix(xx).T

def prox_l1(x, gamma, coef):
    assert(gamma > 0 and coef >= 0)
    lam1 = coef
    return x - abs(x).minimum(lam1 * gamma).multiply(x.sign())

def prox_tv(x, gamma, coef):
    assert(gamma > 0 and coef >= 0)
    return proxTv(x, gamma*coef)
    