import autograd.numpy as np
from autograd import grad

import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve


def f(x, A, num_users, factors, lambda_):
    u = x[:num_users*factors].reshape((-1, factors))
    v = x[num_users*factors:].reshape((-1, factors))
    
    p = A.copy()
    p[np.where(p != 0)] = 1.0
    return np.sum(np.multiply(A + 1, (p - np.dot(u, v.T)) ** 2)) + lambda_ * (np.linalg.norm(u)**2 + np.linalg.norm(v)**2)


def alt_least_squares(x_start, A, factors, maxiter=100, lambda_=5):
    num_users = A.shape[0]
    num_items = A.shape[1]
        
    f_loss = lambda x: f(x, A, num_users, factors, lambda_)
        
    user_vectors = x_start[:num_users*factors].reshape((-1, factors))
    item_vectors = x_start[num_users*factors:].reshape((-1, factors))
    
    history = []
    history_f = []
    
    for i in range(maxiter):
        user_vectors = alt_least_squares_iter(A, True, num_users, item_vectors, factors, lambda_)
        item_vectors = alt_least_squares_iter(A, False, num_items, user_vectors, factors, lambda_)
        
        point = np.concatenate([user_vectors.reshape(-1), item_vectors.reshape(-1)])
        history.append(point)
        history_f.append(f_loss(point))
        
        if i % 100 == 0:
            x = user_vectors.dot(item_vectors.T)
            print("   ", history_f[-1])
            
    return history, history_f


def alt_least_squares_iter(A, first, size, fixed_vecs, factors, lambda_):
    num_fixed = fixed_vecs.shape[0]
    YTY = fixed_vecs.T.dot(fixed_vecs)
    eye = np.eye(num_fixed)
    lambda_eye = lambda_ * np.eye(factors)
    solve_vecs = np.zeros((size, factors))

    for i in range(size):
        if first:
            counts_i = A[i]
        else:
            counts_i = A[:, i].T
        CuI = np.eye(counts_i.shape[0])
        np.fill_diagonal(CuI, counts_i)
        
        pu = counts_i.copy()
        pu[np.where(pu != 0)] = 1.0
        YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
        YTCupu = fixed_vecs.T.dot(CuI + eye).dot(pu.T)
        xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
        solve_vecs[i] = xu

    return solve_vecs


def alt_least_squares_prox(x_start, Lk, A, factors, maxiter=500, lambda_=5):
    num_users = A.shape[0]
    num_items = A.shape[1]
        
    f_loss = lambda x: f(x, A, num_users, factors, lambda_)
        
    user_vectors = x_start[:num_users*factors].reshape((-1, factors))
    item_vectors = x_start[num_users*factors:].reshape((-1, factors))
    
    history = [x_start]
    history_f = [f_loss(x_start)]
    
    for i in range(maxiter):
        grad_ = grad(f_loss)(history[-1]) + Lk * (history[-1] - x_start)
        if norm(grad_) <= 0.5 * Lk * norm(history[-1] - x_start):
            break
        
        user_vectors = alt_least_squares_prox_iter(A, True, num_users, item_vectors, factors, lambda_, Lk, x_start[:num_users*factors])
        item_vectors = alt_least_squares_prox_iter(A, False, num_items, user_vectors, factors, lambda_, Lk, x_start[num_users*factors:])
        
        point = np.concatenate([user_vectors.reshape(-1), item_vectors.reshape(-1)])
        history.append(point)
        history_f.append(f_loss(point))
        
        if i % 20 == 0:
            x = user_vectors.dot(item_vectors.T)
            print("   ", history_f[-1])
            
    return history, history_f, i, 0


def alt_least_squares_prox_iter(A, first, size, fixed_vecs, factors, lambda_, Lk, x_start):
    num_fixed = fixed_vecs.shape[0]
    YTY = fixed_vecs.T.dot(fixed_vecs)
    eye = np.eye(num_fixed)
    lambda_eye = lambda_ * np.eye(factors)
    half_L_eye = 0.5 * Lk * np.eye(factors)
    solve_vecs = np.zeros((size, factors))

    x0 = x_start.reshape((-1, factors))
    
    for i in range(size):
        if first:
            counts_i = A[i]
        else:
            counts_i = A[:, i].T
        CuI = np.eye(counts_i.shape[0])
        np.fill_diagonal(CuI, counts_i)
        
        pu = counts_i.copy()
        pu[np.where(pu != 0)] = 1.0
        YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
        YTCupu = fixed_vecs.T.dot(CuI + eye).dot(pu.T)
        xu = spsolve(YTY + YTCuIY + lambda_eye + half_L_eye, YTCupu + 0.5 * Lk * x0[i])
        solve_vecs[i] = xu

    return solve_vecs