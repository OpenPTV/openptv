import numpy as np
from typing import Tuple
from scipy.linalg import inv

def ata(a: np.ndarray, m: int, n: int, n_large: int) -> np.ndarray:
    """
    Computes A^T * A for matrix a of shape (m, n_large).
    Returns a matrix of shape (n, n_large).
    """
    ata = np.zeros((n, n_large))
    for i in range(n):
        for j in range(n):
            for k in range(m):
                ata[i, j] += a[k, i] * a[k, j]
    return ata

def atl(a: np.ndarray, l: np.ndarray, m: int, n: int, n_large: int) -> np.ndarray:
    """
    Computes A^T * l for matrix a of shape (m, n_large) and vector l of shape (m,).
    Returns a vector of shape (n,).
    """
    u = np.zeros(n)
    for i in range(n):
        for k in range(m):
            u[i] += a[k, i] * l[k]
    return u

def matinv(a: np.ndarray, n: int, n_large: int) -> np.ndarray:
    """
    Returns the inverse of the top-left (n, n) submatrix of a.
    """
    return inv(a[:n, :n])

def matmul(
    a: np.ndarray, b: np.ndarray, m: int, n: int, k: int, m_large: int, n_large: int
) -> np.ndarray:
    """
    Multiplies matrices a (m, n_large) and b (n_large, k).
    Returns a matrix of shape (m, k).
    """
    # Ensure dimensions passed to np.zeros are integers
    c = np.zeros((int(m), int(k)))
    for i in range(m):
        for j in range(k):
            for l in range(n):
                c[i, j] += a[i, l] * b[l, j]
    return c
