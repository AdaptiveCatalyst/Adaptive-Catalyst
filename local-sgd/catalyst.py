import random
import numpy as np


def catalyst(F, grad, x_start, inner_method, L0, Lu, Ld, alpha, beta, gamma, 
             seed=777, maxiter=5, maxiter_inner=2000):
    """
    General Adaptive Catalyst algorithm wrapper
    
    :param F             being optimized objective functional
    :param grad          functional's gradient
    :param x_start       initial value for zk and yk
    :param inner_method  wrapper of the base algorithm, takes starting point, 
                         Lk coefficient, maximum iterations and already executed iterations count parameters
    :param L0, Lu, Ld    initial, minimal and maximal values for variable Lk, Lu >= L0 >= Ld > 0
    :param alpha, beta   multiplicators for Lk on outer and inner iterations
    :param gamma         ratio threshold between previous and new inner method's iteratons count, alpha > beta > gamma > 0
    :param seed          random seed
    :param maxiter       maximum number of outer iterations
    :param maxiter_inner maximum number of inner method's iterations, set to prevent infinite loop 
    
    :return: history_aggregator        list of all interim points of inner method
    :return: history_f_aggregator      ndarray of functional value at these points
    :return: achieved_syncs_aggregator number of achieved synchronization steps
    """
    
    random.seed(seed)
    np.random.seed(seed)
    
    assert alpha > beta > gamma > 0
    assert Lu >= L0 >= Ld > 0
    
    zk = x_start.copy()
    yk = x_start.copy()
    
    Lk = L0
    
    a = None
    A = 0.
    
    history_aggregator = []
    history_f_aggregator = np.array([])
    achieved_syncs_aggregator = 0
    
    for k in range(maxiter):
        print("outer iter", k + 1)
        
        Lk = beta * min(alpha * Lk, Lu)
        
        N_prev = None
        t = 0
        
        yk_new = yk.copy()
        A_new = None
        while True:
            t += 1            
            Lk = max(Lk / beta, Ld)
            print("––– inner iter", t, "L =", round(Lk, 4), 
                  f"(total iters {len(history_f_aggregator)})")
            
            a = (1/Lk + np.sqrt(1/Lk**2 + 4*A/Lk)) / 2
            A_new = A + a
            xk = (A/A_new) * yk + (a/A_new) * zk
            
            history, history_f, N, achieved_syncs = inner_method(xk, Lk, maxiter_inner, len(history_f_aggregator))
            history_aggregator += history
            history_f_aggregator = np.append(history_f_aggregator, history_f)
            achieved_syncs_aggregator += achieved_syncs
            yk_new = history[-1].copy()
            
            print("  >", history_f_aggregator[-1])
            
            if (t > 1 and N >= gamma * N_prev) or Lk == Ld or N == maxiter_inner - 1:
                break
            N_prev = N
        
        yk = yk_new.copy() if F(yk_new) < F(yk) else yk
        A = A_new
            
        zk = zk - a * grad(yk)
        
    print(f"(total iters {len(history_f_aggregator)})")
    return history_aggregator, history_f_aggregator, achieved_syncs_aggregator