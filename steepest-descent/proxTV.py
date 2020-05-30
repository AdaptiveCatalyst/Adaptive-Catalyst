import numpy as np
from scipy.sparse import csr_matrix, csc_matrix


def proxTv(x, lam):
    """
    :param x: point to compute proximal operator it can be np.array or csr_marix
    :param lam: parameter for prox, gamma*lambda
    :return: if x is array then return value is array, otherwise it is csc_matrix of a form(n,1)
    """
    if (type(x) != type(np.zeros(2))):
        data_csr = True
        if x.shape[1] == 1:
            point = x.toarray()
        else:
            point = x.T.toarray()
    else:
        data_csr = False
        point = np.copy(x)
    k = k0=km=kp = 1
    vmin = point[0] - lam
    vmax = point[0] + lam
    umin = lam
    umax = -lam
    N = len(point)
    res = np.zeros(N)
    do = True
    while do:
        if k > N:
            if data_csr:
                res = csr_matrix(res).T
            return res
        if k == N:
            res[N-1] = vmin+umin
            if data_csr:
                res = csr_matrix(res).T
            return res
        while k < N:
            if (point[k] + umin < vmin - lam):
                for kk in range(k0, km+1):
                    res[kk-1] = vmin
                k = k0 = km = kp = km+1
                vmin = point[k-1]
                vmax = point[k-1] + 2*lam
                umin = lam
                umax = -lam
            elif point[k] + umax > vmax + lam:
                for kk in range(k0, kp+1):
                    res[kk-1] = vmax
                k = k0 = km = kp = kp+1
                vmin = point[k-1] - 2*lam
                vmax = point[k-1]
                umin = lam
                umax = -lam
            else:
                k = k+1
                umin = umin + point[k-1] - vmin
                umax = umax - vmax + point[k-1]
                if umin >= lam:
                    vmin = vmin + (umin - lam)/float(k - k0 + 1)
                    umin = lam
                    km = k
                if umax <= -lam:
                    vmax = vmax + (umax + lam)/float(k - k0 + 1)
                    umax = -lam
                    kp = k

        if umin < 0:
            for kk in range(k0, km+1):
                res[kk-1] = vmin
            k = k0 = km = km+1
            vmin = point[k-1]
            umin = lam
            umax = point[k-1] + lam - vmax
        elif umax > 0:
            for kk in range(k0, kp+1):
                res[kk-1] = vmax
            k = k0 = kp = kp + 1
            vmax = point[k-1]
            umax = -lam
            umin = point[k-1] - lam - vmin
        else:
            for kk in range(k0, N+1):
                res[kk-1] = vmin + umin/float(k-k0+1)
            if data_csr:
                res = csr_matrix(res).T
            return res
