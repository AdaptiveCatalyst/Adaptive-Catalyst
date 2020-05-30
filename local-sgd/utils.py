import numpy as np
import pickle
import enum


def save_object(obj, fname):
    with open("data/" + fname + ".data", "wb") as file:
        pickle.dump(obj, file)
    
    
def load_object(fname):
    obj = None
    with open("data/" + fname + ".data", "rb") as file:
        obj = pickle.load(file)
        
    return obj


def lipschitz_constant(func, n):
    interim_function = lambda x: abs(func(x[:n]) - func(x[n:])) / (np.linalg.norm(x[:n] - x[n:]) + 1e-7)
    
    return max([interim_function(np.random.random(2*n)) for i in range(10)])


def strong_convexity_constant(f_grad, n):
    interim_function = lambda x: (f_grad(x[:n]) - f_grad(x[n:])).T.dot(x[:n] - x[n:]) / \
                                 (np.linalg.norm(x[:n] - x[n:])**2)

    return min([interim_function(np.random.random(2*n)) for i in range(10)])


def estimate_constants(f, f_grad, n):
    return lipschitz_constant(f, n), strong_convexity_constant(f_grad, n)
