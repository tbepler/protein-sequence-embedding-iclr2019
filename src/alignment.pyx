from __future__ import print_function,division

cimport numpy as np
import numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def half_global_alignment(np.ndarray[np.uint8_t] x, np.ndarray[np.uint8_t] y, np.ndarray[np.int32_t, ndim=2] S
                         ,  np.int32_t gap, np.int32_t extend):

    """ Matches all of x to a substring of y"""
    
    cdef int n,m
    n = len(x)
    m = len(y)

    cdef np.float32_t MI = -np.inf
    cdef np.ndarray[np.float32_t, ndim=3] A = np.zeros((n+1,m+1,3), dtype=np.float32)

    A[0,1:,0] = MI
    A[1:,0,0] = MI

    A[0,1:,1] = MI # gap in x
    A[1:,0,1] = gap + np.arange(n)*extend

    # starting from gap in y costs 0
    A[1:,0,2] = MI

    # initialize the traceback matrix
    cdef np.ndarray[np.int8_t, ndim=3] tb = np.zeros((n+1,m+1,3), dtype=np.int8) - 1

    cdef np.float32_t s
    cdef int i,j
    cdef np.int8_t k

    for i in range(n):
        for j in range(m):
            # match i,j
            k = 0
            s = A[i,j,0]
            if A[i,j,1] > s:
                k = 1
                s = A[i,j,1]
            if A[i,j,2] > s:
                k = 2
                s = A[i,j,2]
            A[i+1,j+1,0] = s + S[x[i],y[j]]
            tb[i+1,j+1,0] = k
            # insert in x
            k = 0
            s = A[i,j+1,0] + gap
            if A[i,j+1,1] + extend > s:
                k = 1
                s = A[i,j+1,1] + extend
            if A[i,j+1,2] + gap > s:
                k = 2
                s = A[i,j+1,2] +  gap
            A[i+1,j+1,1] = s
            tb[i+1,j+1,1] = k
            # insert in y 
            k = 0
            s = A[i+1,j,0] + gap
            if A[i+1,j,1] + gap > s:
                k = 1
                s = A[i+1,j,1] + gap
            if A[i+1,j,2] + extend > s:
                k = 2
                s = A[i+1,j,2] + extend
            A[i+1,j+1,2] = s
            tb[i+1,j+1,2] = k


    # find the end of the best alignment
    cdef int j_max, k_max
    cdef np.float32_t s_max

    j_max = 0
    k_max = 0
    s_max = A[n,j_max,k_max]
    for j in range(m+1):
        for k in range(3):
            if A[n,j,k] > s_max:
                s_max = A[n,j,k]
                j_max = j
                k_max = k

    # backtrack the alignment
    cdef int k_next
    i = n 
    j = j_max
    k = k_max
    while tb[i,j,k] >= 0:
        k_next = tb[i,j,k]
        if k == 0:
            i = i - 1
            j = j - 1
        elif k == 1:
            i = i - 1
        elif k == 2:
            j = j - 1
        k = k_next

    return s_max, j, j_max

cdef int argmax(np.float32_t a, np.float32_t b, np.float32_t c):
    if b > a:
        if c > b:
            return 2
        return 1
    if c > a:
        return 2
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
def nw_score_extra(np.ndarray[np.uint8_t] x, np.ndarray[np.uint8_t] y, np.ndarray[np.int32_t, ndim=2] S
                  ,  np.int32_t gap, np.int32_t extend):
    
    cdef int n,m
    n = len(x)
    m = len(y)

    cdef np.float32_t MI = -np.inf

    cdef np.ndarray[np.float32_t, ndim=2] A_prev = np.zeros((m+1,3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] A = np.zeros((m+1,3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] A_temp

    A[1:,0] = MI # match scores
    A[1:,1] = MI # gap in x
    A[1:,2] = gap + np.arange(m)*extend # gap in y	

    # also calculate the length of the alignment
    cdef np.ndarray[np.int32_t, ndim=2] L_prev = np.zeros((m+1,3), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=2] L = np.zeros((m+1,3), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=2] L_temp

    # and the number of exact matches within the alignment
    cdef np.ndarray[np.int32_t, ndim=2] M_prev = np.zeros((m+1,3), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=2] M = np.zeros((m+1,3), dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=2] M_temp


    cdef np.float32_t s
    cdef int i,j,k

    for i in range(n):
        # swap A and A_prev
        A_temp = A_prev
        A_prev = A
        A = A_temp

        L_temp = L_prev
        L_prev = L 
        L = L_temp

        M_temp = M_prev
        M_prev = M 
        M = M_temp

        # init A[0]
        A[0,0] = MI
        A[0,1] = gap + i*extend
        A[0,2] = MI
        for j in range(m):
            # match i,j
            s = S[x[i],y[j]]
            k = argmax(A_prev[j,0], A_prev[j,1], A_prev[j,2])
            A[j+1,0] = s + A_prev[j,k]
            L[j+1,0] = L_prev[j,k] + 1
            if x[i] == y[j]:
                M[j+1,0] = M_prev[j,k] + 1
            else:
                M[j+1,0] = M_prev[j,k]

            # insert in x 
            k = argmax(A_prev[j+1,0]+gap, A_prev[j+1,1]+extend, A_prev[j+1,2]+gap)
            A[j+1,1] = max(A_prev[j+1,0]+gap, A_prev[j+1,1]+extend, A_prev[j+1,2]+gap)
            L[j+1,1] = L_prev[j+1,k] + 1
            M[j+1,1] = M_prev[j+1,k]

            # insert in y 
            k = argmax(A[j,0]+gap, A[j,1]+gap, A[j,2]+extend)
            A[j+1,2] = max(A[j,0]+gap, A[j,1]+gap, A[j,2]+extend)
            L[j+1,2] = L[j,k] + 1
            M[j+1,2] = M[j,k]

    k = argmax(A[m,0], A[m,1], A[m,2])

    return A[m,k], M[m,k], L[m,k] 


@cython.boundscheck(False)
@cython.wraparound(False)
def sw_score_subst_no_affine(np.ndarray[np.float32_t,ndim=2] subst
                            ,  np.float32_t gap):
    
    cdef int n,m
    n = subst.shape[0]
    m = subst.shape[1]

    cdef np.float32_t MI = -np.inf

    cdef np.ndarray[np.float32_t, ndim=2] A = np.zeros((n+1,m+1), dtype=np.float32)

    cdef int i,j
    for i in range(n):
        for j in range(m):
            A[i+1,j+1] = max(subst[i,j] + A[i,j], gap+A[i+1,j], gap+A[i,j+1], 0)

    return np.max(A)

@cython.boundscheck(False)
@cython.wraparound(False)
def sw_score_subst(np.ndarray[np.float32_t,ndim=2] subst
                            ,  np.float32_t gap, np.float32_t extend):
    
    cdef int n,m
    n = subst.shape[0]
    m = subst.shape[1]

    cdef np.float32_t MI = -np.inf

    cdef np.ndarray[np.float32_t, ndim=3] A = np.zeros((n+1,m+1,3), dtype=np.float32)

    A[0,1:,0] = MI # match scores
    A[1:,0,0] = MI

    A[0,1:,1] = MI # gap in x
    A[1:,0,2] = MI # gap in y

    cdef np.float32_t s
    cdef int i,j

    for i in range(n):
        for j in range(m):
            # match i,j
            s = subst[i,j]
            A[i+1,j+1,0] = s + max(A[i,j,0], A[i,j,1], A[i,j,2], 0)
            # insert in x 
            A[i+1,j+1,1] = max(A[i,j+1,0]+gap, A[i,j+1,1]+extend, A[i,j+1,2]+gap, 0)
            # insert in y 
            A[i+1,j+1,2] = max(A[i+1,j,0]+gap, A[i+1,j,1]+gap, A[i+1,j,2]+extend, 0)

    s = np.max(A)
    #s = max(A[m,0], A[m,1], A[m,2])

    return s


@cython.boundscheck(False)
@cython.wraparound(False)
def sw_score(np.ndarray[np.uint8_t] x, np.ndarray[np.uint8_t] y, np.ndarray[np.int32_t, ndim=2] S
            ,  np.int32_t gap, np.int32_t extend):
    
    cdef int n,m
    n = len(x)
    m = len(y)

    cdef np.float32_t MI = -np.inf

    cdef np.ndarray[np.float32_t, ndim=3] A = np.zeros((n+1,m+1,3), dtype=np.float32)

    A[0,1:,0] = MI # match scores
    A[1:,0,0] = MI

    A[0,1:,1] = MI # gap in x
    A[1:,0,2] = MI # gap in y

    #A[1:,2] = gap + np.arange(m)*extend # gap in y	

    cdef np.float32_t s
    cdef int i,j

    for i in range(n):
        for j in range(m):
            # match i,j
            s = S[x[i],y[j]]
            A[i+1,j+1,0] = s + max(A[i,j,0], A[i,j,1], A[i,j,2], 0)
            # insert in x 
            A[i+1,j+1,1] = max(A[i,j+1,0]+gap, A[i,j+1,1]+extend, A[i,j+1,2]+gap, 0)
            # insert in y 
            A[i+1,j+1,2] = max(A[i+1,j,0]+gap, A[i+1,j,1]+gap, A[i+1,j,2]+extend, 0)

    s = np.max(A)
    #s = max(A[m,0], A[m,1], A[m,2])

    return s


@cython.boundscheck(False)
@cython.wraparound(False)
def nw_score(np.ndarray[np.uint8_t] x, np.ndarray[np.uint8_t] y, np.ndarray[np.int32_t, ndim=2] S
            ,  np.int32_t gap, np.int32_t extend):
    
    cdef int n,m
    n = len(x)
    m = len(y)

    cdef np.float32_t MI = -np.inf

    cdef np.ndarray[np.float32_t, ndim=2] A_prev = np.zeros((m+1,3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] A = np.zeros((m+1,3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] A_temp

    A[1:,0] = MI # match scores
    A[1:,1] = MI # gap in x
    A[1:,2] = gap + np.arange(m)*extend # gap in y	

    cdef np.float32_t s
    cdef int i,j

    for i in range(n):
        # swap A and A_prev
        A_temp = A_prev
        A_prev = A
        A = A_temp
        # init A[0]
        A[0,0] = MI
        A[0,1] = gap + i*extend
        A[0,2] = MI
        for j in range(m):
            # match i,j
            s = S[x[i],y[j]]
            A[j+1,0] = s + max(A_prev[j,0], A_prev[j,1], A_prev[j,2])
            # insert in x 
            A[j+1,1] = max(A_prev[j+1,0]+gap, A_prev[j+1,1]+extend, A_prev[j+1,2]+gap)
            #A[j+1,1] = max(A_prev[j+1,0]+gap, A_prev[j+1,1]+extend)
            # insert in y 
            A[j+1,2] = max(A[j,0]+gap, A[j,1]+gap, A[j,2]+extend)
            #A[j+1,2] = max(A[j,0]+gap, A[j,2]+extend)

    s = max(A[m,0], A[m,1], A[m,2])

    return s

@cython.boundscheck(False)
@cython.wraparound(False)
def nw_score_subst(np.ndarray[np.float32_t, ndim=2] S
            ,  np.int32_t gap, np.int32_t extend):
    
    cdef int n,m
    n = S.shape[0]
    m = S.shape[1]

    cdef np.float32_t MI = -np.inf

    cdef np.ndarray[np.float32_t, ndim=2] A_prev = np.zeros((m+1,3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] A = np.zeros((m+1,3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] A_temp

    A[1:,0] = MI # match scores
    A[1:,1] = MI # gap in x
    A[1:,2] = gap + np.arange(m)*extend # gap in y	

    cdef np.float32_t s
    cdef int i,j

    for i in range(n):
        # swap A and A_prev
        A_temp = A_prev
        A_prev = A
        A = A_temp
        # init A[0]
        A[0,0] = MI
        A[0,1] = gap + i*extend
        A[0,2] = MI
        for j in range(m):
            # match i,j
            s = S[i,j]
            A[j+1,0] = s + max(A_prev[j,0], A_prev[j,1], A_prev[j,2])
            # insert in x 
            A[j+1,1] = max(A_prev[j+1,0]+gap, A_prev[j+1,1]+extend, A_prev[j+1,2]+gap)
            #A[j+1,1] = max(A_prev[j+1,0]+gap, A_prev[j+1,1]+extend)
            # insert in y 
            A[j+1,2] = max(A[j,0]+gap, A[j,1]+gap, A[j,2]+extend)
            #A[j+1,2] = max(A[j,0]+gap, A[j,2]+extend)

    s = max(A[m,0], A[m,1], A[m,2])

    return s

@cython.boundscheck(False)
@cython.wraparound(False)
def nw_align_subst(np.ndarray[np.float32_t, ndim=2] S
            ,  np.int32_t gap, np.int32_t extend):
    
    cdef int n,m
    n = S.shape[0]
    m = S.shape[1]

    cdef np.float32_t MI = -np.inf

    cdef np.ndarray[np.float32_t, ndim=2] A_prev = np.zeros((m+1,3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] A = np.zeros((m+1,3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] A_temp

    A[1:,0] = MI # match scores
    A[1:,1] = MI # gap in x
    A[1:,2] = gap + np.arange(m)*extend # gap in y	

    cdef np.ndarray[np.int8_t, ndim=3] tb = np.zeros((n+1,m+1,3), dtype=np.int8)
    tb[:] = -1
    tb[1:,:,1] = 1
    tb[:,1:,2] = 2

    cdef np.float32_t s
    cdef int i,j,k

    for i in range(n):
        # swap A and A_prev
        A_temp = A_prev
        A_prev = A
        A = A_temp
        # init A[0]
        A[0,0] = MI
        A[0,1] = gap + i*extend
        A[0,2] = MI
        for j in range(m):
            # match i,j
            s = S[i,j]
            k = 0
            if A_prev[j,k] < A_prev[j,1]:
                k = 1
            if A_prev[j,k] < A_prev[j,2]:
                k = 2
            A[j+1,0] = s + A_prev[j,k]
            tb[i+1,j+1,0] = k

            # insert in x 
            s = A_prev[j+1,0] + gap
            k = 0
            if s < A_prev[j+1,1] + extend:
                s = A_prev[j+1,1] + extend
                k = 1
            if s < A_prev[j+1,2] + gap:
                s = A_prev[j+1,2] + gap
                k = 2
            A[j+1,1] = s
            tb[i+1,j+1,1] = k 

            # insert in y 
            s = A[j,0] + gap
            k = 0
            if s < A[j,1] + gap:
                s = A[j,1] + gap
                k = 1
            if s < A[j,2] + extend:
                s = A[j,2] + extend
                k = 2
            A[j+1,2] = s
            tb[i+1,j+1,2] = k

    k = 0
    s = A[m,k]
    if s < A[m,1]:
        s = A[m,1]
        k = 1
    if s < A[m,2]:
        s = A[m,2]
        k = 2

    ## traceback
    align = []
    i = n
    j = m
    while k >= 0:
        k_ = tb[i,j,k]
        if k == 0:
            align.append((i-1,j-1))
            i -= 1
            j -= 1
        elif k == 1:
            i -= 1
        elif k == 2:
            j -= 1
        k = k_
    align = np.array(align[::-1], dtype=int)

    return s, align

@cython.boundscheck(False)
@cython.wraparound(False)
def nw_affine_subst_score(np.ndarray[np.float32_t, ndim=2] S, np.float32_t gap
                         , np.float32_t extend):
    
    cdef int n,m
    n = S.shape[0]
    m = S.shape[1]

    cdef np.float32_t MI = -np.inf

    cdef np.ndarray[np.float32_t, ndim=2] A_prev = np.zeros((m+1,3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] A = np.zeros((m+1,3), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=2] A_temp

    A[1:,0] = MI # match scores
    A[1:,1] = MI # gap in x
    A[1:,2] = gap + np.arange(m)*extend # gap in y	

    cdef np.float32_t s
    cdef int i,j

    for i in range(n):
        # swap A and A_prev
        A_temp = A_prev
        A_prev = A
        A = A_temp
        # init A[0]
        A[0,0] = MI
        A[0,1] = gap + i*extend
        A[0,2] = MI
        for j in range(m):
            # match i,j
            s = S[i,j]
            A[j+1,0] = s + max(A_prev[j,0], A_prev[j,1], A_prev[j,2])
            # insert in x 
            A[j+1,1] = max(A_prev[j+1,0]+gap, A_prev[j+1,1]+extend, A_prev[j+1,2]+gap)
            # insert in y 
            A[j+1,2] = max(A[j,0]+gap, A[j,1]+gap, A[j,2]+extend)

    s = max(A[m,0], A[m,1], A[m,2])

    return s

@cython.boundscheck(False)
@cython.wraparound(False)
def nw_subst_score(np.ndarray[np.float32_t, ndim=2] S, np.float32_t gap): #, np.float32_t extend):
    
    cdef int n,m
    n = S.shape[0]
    m = S.shape[1]

    cdef np.float32_t MI = -np.inf

    cdef np.ndarray[np.float32_t] A_prev = np.zeros(m+1, dtype=np.float32)
    cdef np.ndarray[np.float32_t] A = np.zeros(m+1, dtype=np.float32)
    cdef np.ndarray[np.float32_t] A_temp

    A[1:] = gap*np.arange(m)	

    cdef np.float32_t s
    cdef int i,j

    for i in range(n):
        # swap A and A_prev
        A_temp = A_prev
        A_prev = A
        A = A_temp
        # init A[0]
        A[0] = gap*(i+1)
        for j in range(m):
            # match i,j
            s = S[i,j]
            A[j+1] = max(A_prev[j]+s, A_prev[j+1]+gap, A[j]+gap)

    s = A[m]

    return s

@cython.boundscheck(False)
@cython.wraparound(False)
def edit_distance(np.ndarray[np.uint8_t] x, np.ndarray[np.uint8_t] y):
    
    cdef int n,m
    n = len(x)
    m = len(y)

    cdef np.ndarray[np.int32_t, ndim=2] A = np.zeros((n+1,m+1), dtype=np.int32)
    A[0,1:] = np.arange(m) + 1
    A[1:,0] = np.arange(n) + 1

    cdef int i,j
    cdef np.int32_t s

    for i in range(n):
        for j in range(m):
            s = int(x[i] != y[j])
            A[i+1,j+1] = min(A[i,j] + s, A[i+1,j] + 1, A[i,j+1] + 1)

    return A[n,m]

@cython.boundscheck(False)
@cython.wraparound(False)
def dtw_subst_score(np.ndarray[np.float32_t, ndim=2] S):
    
    cdef int n,m
    n = S.shape[0]
    m = S.shape[1]

    cdef np.float32_t MI = -np.inf

    cdef np.ndarray[np.float32_t] A_prev = np.zeros(m+1, dtype=np.float32)
    cdef np.ndarray[np.float32_t] A = np.zeros(m+1, dtype=np.float32)
    cdef np.ndarray[np.float32_t] A_temp

    cdef np.float32_t s
    cdef int i,j

    for i in range(n):
        # swap A and A_prev
        A_temp = A_prev
        A_prev = A
        A = A_temp
        for j in range(m):
            # match i,j
            s = S[i,j]
            A[j+1] = s + max(A_prev[j], A_prev[j+1], A[j])

    s = A[m]

    return s

@cython.boundscheck(False)
@cython.wraparound(False)
def nw_align( np.ndarray[np.float32_t, ndim=2] S, np.float32_t gap):
    
    cdef int n,m
    n = S.shape[0]
    m = S.shape[1]
    
    cdef int i,j
    cdef np.float32_t score
    cdef np.ndarray[np.float32_t, ndim=2] A = np.zeros((n+1,m+1), dtype=np.float32)
    A[:,0] = gap*np.arange(n+1)
    A[0,:] = gap*np.arange(m+1)
    A[1:,1:] = -np.inf
    
    cdef np.ndarray[np.int8_t, ndim=2] paths = np.zeros((n+1,m+1), dtype=np.int8)
    paths[:] = -1
    paths[0,1:] = 1
    paths[1:,0] = 2
    
    i = j = 0
    score = -np.inf
    
    for i in range(n):
        for j in range(m):
            s = A[i,j] + S[i,j]
            if s > A[i+1,j+1]:
                A[i+1,j+1] = s
                paths[i+1,j+1] = 0
                
            gi = A[i+1,j] + gap
            if gi > A[i+1,j+1]:
                A[i+1,j+1] = gi
                paths[i+1,j+1] = 1
                
            gj = A[i,j+1] + gap
            if gj > A[i+1,j+1]:
                A[i+1,j+1] = gj
                paths[i+1,j+1] = 2

    ## traceback
    tb = []
    i = n
    j = m
    while paths[i,j] >= 0:
        if paths[i,j] == 0:
            tb.append((i-1,j-1))
            i -= 1
            j -= 1
        elif paths[i,j] == 1:
            j -= 1
        elif paths[i,j] == 2:
            i -= 1
    tb = np.array(tb[::-1], dtype=int)

    
    return A[n,m], tb




