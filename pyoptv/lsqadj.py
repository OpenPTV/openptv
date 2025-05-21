import numpy as np
import numba
from scipy.linalg import inv

@numba.jit(nopython=True)
def ata(a, m, n, n_large):
    ata = np.zeros((n, n_large))
    for i in range(n):
        for j in range(n):
            for k in range(m):
                ata[i, j] += a[k, i] * a[k, j]
    return ata

@numba.jit(nopython=True)
def atl(a, l, m, n, n_large):
    u = np.zeros(n)
    for i in range(n):
        for k in range(m):
            u[i] += a[k, i] * l[k]
    return u

@numba.jit(nopython=True)
def matinv(a, n, n_large):
    return inv(a[:n, :n])

@numba.jit(nopython=True)
def matmul(a, b, m, n, k, m_large, n_large):
    c = np.zeros((m, k))
    for i in range(m):
        for j in range(k):
            for l in range(n):
                c[i, j] += a[i, l] * b[l, j]
    return c

@numba.jit(nopython=True)
def norm_cross(a, b):
    n = np.zeros(3)
    n[0] = a[1] * b[2] - a[2] * b[1]
    n[1] = a[2] * b[0] - a[0] * b[2]
    n[2] = a[0] * b[1] - a[1] * b[0]
    return n
