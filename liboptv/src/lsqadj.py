def ata(a, ata, m, n, n_large):
    """
    Multiply transpose of a matrix A by matrix A itself, creating symmetric matrix
    with the option of working with the sub-matrix only
    
    Arguments:
    a - matrix of doubles of the size (m x n_large).
    ata  - matrix of the result multiply(a.T,a) of size (n x n)
    m - number of rows in matrix a
    n - number of rows and columns in the output ata - the size of the sub-matrix
    n_large - number of columns in matrix a
    """
    for i in range(n):
        for j in range(n):
            sum_ = 0.0
            for k in range(m):
                sum_ += a[k * n_large + i] * a[k * n_large + j]
            ata[i * n_large + j] = sum_

import numpy as np

def atl(u, a, l, m, n, n_large):
    """
    Multiply transpose of a matrix A by vector l, creating vector u
    with the option of working with the sub-vector only, when n < n_large

    Arguments:
    u - vector of doubles of the size (n x 1)
    a - matrix of doubles of the size (m x n_large)
    l - vector of doubles (m x 1)
    m - number of rows in matrix a
    n - length of the output u - the size of the sub-matrix
    n_large - number of columns in matrix a
    """
    for i in range(n):
        u[i] = np.dot(a[:, i], l)
    return u


import numpy as np

def matinv(a, n, n_large):
    # Convert input to numpy array
    a = np.array(a)
    
    # Loop over pivots
    for ipiv in range(n):
        pivot = 1.0 / a[ipiv * n_large + ipiv]
        npivot = -pivot
        
        # Update elements of the matrix
        for irow in range(n):
            for icol in range(n):
                if irow != ipiv and icol != ipiv:
                    a[irow * n_large + icol] -= a[ipiv * n_large + icol] * a[irow * n_large + ipiv] * pivot
        
        # Update elements of the pivot column
        for irow in range(n):
            if irow != ipiv:
                a[irow * n_large + ipiv] *= pivot
        
        # Update elements of the pivot row
        for icol in range(n):
            if ipiv != icol:
                a[ipiv * n_large + icol] *= npivot
        
        # Update pivot element
        a[ipiv * n_large + ipiv] = pivot
        
    return a


import numpy as np

def matmul(a, b, c, m, n, k, m_large, n_large):
    for i in range(k):
        pb = b
        pa = a[i::k]
        for j in range(m):
            pc = c
            x = 0.0
            for l in range(n):
                x += pb[l * m_large] * pc[0]
                pc += k
            
            for l in range(n_large - n):
                pb += 1
                pc += k
            
            pa[0] = x
            pa += k
        
        for j in range(m_large - m):
            pa += k
        
        c += 1
