from __future__ import division, print_function

cimport cython
cimport numpy as np
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def average_precision(np.ndarray[float] target, np.ndarray[float] pred, N=None):
    cdef float n
    if N is None:
        n = target.sum()
    else:
        n = N
    
    ## copy the target and prediction into matrix
    cdef np.ndarray[float, ndim=2] matrix = np.zeros((target.shape[0],2), dtype=np.float32)
    matrix[:,0] = -pred # negate the prediction to sort in descending order
    matrix[:,1] = target
    matrix.view('f4,f4').sort(order='f0', axis=0) # sort the rows
    #print(matrix[:10])
    
    cdef float auprc, count, pr, relk, delta
    auprc = count = pr = relk = 0
    cdef int i = 0
    
    for i in range(matrix.shape[0]):
        count += 1
        relk += matrix[i,1] # target
        delta = matrix[i,1] - pr
        pr += delta/count
        if i >= matrix.shape[0] - 1 or matrix[i,0] != matrix[i+1,0]:
            auprc += pr*relk
            relk = 0
    auprc /= n
    
    return auprc

