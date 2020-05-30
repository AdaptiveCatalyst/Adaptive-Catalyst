import random
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm_notebook


def calculate_gap(set_):
    """
    Special method for calculating Local SGD constants.
    Calculates maximum difference betwen two consequent integers in set
    """
    
    gap = 0
    for i in range(1, len(set_)):
        if set_[i] - set_[i - 1] > gap:
            gap = set_[i] - set_[i - 1]
    return gap


def local_sgd(F, x_start, L, mu, m, seed=777, workers=20, syncs_interval=400, maxiter=20,
              batch_size=None, stoch_grad_batch=None, stoch_grad_batch_flat=None):
    """
    Local SGD
    paper: Sebastian U. Stich. 
           Local SGD Converges Fast and Communicates Little 
           (https://arxiv.org/abs/1805.09767)
    
    param: F                objective functional
    param: x_start          starting point for all the workers
    param: L, mu            estimated Lipschitz constant and strong convexity constant
    param: m                count of samples in learning set
    param: seed             random seed
    param: workers          count of compute nodes, concurrently optimizing functional
    param: sync_interval    number of iterations between two consequent synchronization steps
    param: maxiter          maximum number of iterations
    param: batch_size       count of samples, using in one stochastic gradient calculation, 
                            default None means that stoch grad calculates with one sample
    param: stoch_grad_batch      general stoch grad function, takes point and set of sample indices
    param: stoch_grad_batch_flat general stoch grad function, takes ndarray of points (one for every worker) 
                                 and set of sample indices. using at synchronization steps
    
    :return: history        list of all interim points
    :return: history_f      ndarray of functional value at these points
    :return: achieved_syncs number of achieved synchronization steps
    """
    
    random.seed(seed)
    np.random.seed(seed)
    
    # initial points for all the workers
    x0 = np.array([x_start.copy() for _ in range(workers)])
    
    kappa = L / mu
    
    sync_steps = []

    gap = syncs_interval
    a_coef = max(16 * kappa, gap)
    
    sum_points = np.zeros_like(x_start)
    sum_coef = 0.
    
    history = []
    history_f = []
    
    x_it = x0.copy()
    achieved_syncs = 0
    
    for it in tqdm_notebook(range(maxiter)):
        eta = 4 / (mu * (a_coef + it))
        
        if it % syncs_interval == 0:
            sync_steps.append(it)
            achieved_syncs += 1
        
        x_new = x_it.copy()
        for k in range(workers):
            # if batch is None generate one index, else generate a set of indices
            i_set = [random.randint(0, m - 1)] if batch_size is None else np.random.randint(m, size=batch_size)
            
            if it in sync_steps:
                x_new[k] = (x_new - eta * stoch_grad_batch_flat(x_new, i_set)).mean(axis=0)
            else:
                x_new[k] = x_new[k] - eta * stoch_grad_batch(x_new[k], i_set)
        
        x_it = x_new.copy()
        
        omega = (a_coef + it) ** 2
        sum_points += omega * x_it.mean(axis=0)
        sum_coef += omega
        
        history.append(sum_points.copy() / sum_coef)
        history_f.append(F(history[-1]))
        
    return history, np.array(history_f), achieved_syncs


def local_sgd_prox(F, F_grad, x_start, L, Lk, mu, m, seed=777, workers=20, syncs_interval=400, maxiter=5000, total_iter=0,
                   batch_size=None, stoch_grad_batch=None, stoch_grad_batch_flat=None):
    """
    Proximal Local SGD, optimizing functional with new regilarization term
    and stopping criterion that allows wrapping this algorithm with adaptive catalyst
    
    param: F                objective functional
    param: F_grad           functional's gradient
    param: x_start          starting point for all the workers
    param: L, mu            estimated Lipschitz constant and strong convexity constant
    param: Lk               regularization coefficient
    param: m                count of samples in learning set
    param: seed             random seed
    param: workers          count of compute nodes, concurrently optimizing functional
    param: sync_interval    number of iterations between two consequent synchronization steps
    param: maxiter          maximum number of iterations
    param: total_iter       number of iterations that have been already executed at this inner iteration.
                            using to decide, when the synchronization iteration should be
    param: batch_size       count of samples, using in one stochastic gradient calculation, 
                            default None means that stoch grad calculates with one sample
    param: stoch_grad_batch      general stoch grad function, takes point and set of sample indices
    param: stoch_grad_batch_flat general stoch grad function, takes ndarray of points (one for every worker) 
                                 and set of sample indices. using at synchronization steps
    
    :return: history        list of all interim points
    :return: history_f      ndarray of functional value at these points
    :return: it             number of executed iterations
    :return: achieved_syncs number of achieved synchronization steps
    """
    
    # initial points for all the workers
    x0 = np.array([x_start.copy() for _ in range(workers)])
    
    kappa = (L + Lk) / (mu + Lk)
    
    sync_steps = []

    gap = syncs_interval
    a_coef = max(16 * kappa, gap)
    
    sum_points = np.zeros_like(x_start)
    sum_coef = 0.
    
    history = [x_start.copy()]
    history_f = [F(x_start)]
    
    x_it = x0.copy()
    achieved_syncs = 0
    
    for it in range(maxiter):
        eta = 4 / (mu * (a_coef + it))
        
        if (total_iter + it) % syncs_interval == 0:
            sync_steps.append(it)
            achieved_syncs += 1
        
        x_new = x_it.copy()
        for k in range(workers):
            # if batch is None generate one index, else generate a set of indices
            i_set = [random.randint(0, m - 1)] if batch_size is None else np.random.randint(m, size=batch_size)
            
            if it in sync_steps:
                x_new[k] = (x_new - eta * (stoch_grad_batch_flat(x_new, i_set) + Lk * (history[-1] - x_start))).mean(axis=0)
            else:
                x_new[k] = x_new[k] - eta * (stoch_grad_batch(x_new[k], i_set) + Lk * (history[-1] - x_start))
        
        x_it = x_new.copy()
        
        omega = (a_coef + it) ** 2
        sum_points += omega * x_it.mean(axis=0)
        sum_coef += omega
        
        history.append(sum_points.copy() / sum_coef)
        history_f.append(F(history[-1]))
        
        grad = F_grad(history[-1]) + Lk * (history[-1] - x_start)
        if norm(grad) <= 0.5 * Lk * norm(history[-1] - x_start):
            break
            
        if it % 20 == 0:
            print("   ", history_f[-1])
    
    return history, np.array(history_f), it, achieved_syncs
