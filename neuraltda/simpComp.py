#########################################################################################
## Routines for manipulating simplicial complexes                                      ##
## and computing the boundary operators.                                               ##
## Follows algorithms in "Computational Homology" by Kaczynski, Mischaikow, and Mrozek ##
## Bradley Theilman 2 February 2017                                                    ##
#########################################################################################

import numpy as np 
import scipy.linalg as spla

def union(a, b):
    return list(set(a) | set(b))

def primaryFaces(Q):
    L = []
    d = len(Q)
    Q = list(Q)
    Q.extend(Q[:d-2])
    for ind in range(d):
        s = (Q[ind:ind+(d-1)])
        L.append(tuple(sorted(s)))
    return L

def simplicialChainGroups(maxsimps, maxdim):
    E=[[] for ind in range(maxdim+2)]
    K = list(maxsimps)
    while(len(K) > 0):
        Q = K.pop(0)
        L = primaryFaces(Q)
        k = len(Q)-1
        K = union(K, L)
        E[k] = union(E[k], L))
        E[k+1] = union(E[k+1], {Q}))
    for k in range(len(E)):
        E[k] = sorted(E[k])
    return E

def boundaryOperator(Q):
    sgn = 1
    c = dict()
    Ql = list(Q)
    for ind in range(len(Ql)):
        n = Ql.pop(ind)
        c[tuple(Ql)] = sgn
        Ql.insert(ind, n)
        sgn = -1*sgn
    return c

def canonicalCoordinates(c, K):
    v = np.zeros(len(K))
    for ind in range(len(K)):
        if c.has_key(K[ind]):
            v[ind] = c[K[ind]]
    return v

def boundaryOperatorMatrix(E):
    
    nmat = len(E)
    D = [[] for i in range(nmat)]
    for k in range(1, nmat):
        m = len(E[k-1])
        n = len(E[k])
        mat = np.zeros((m, n))
        for j in range(n):
            c = boundaryOperator(E[k][j])
            mat[:, j] = canonicalCoordinates(c, E[k-1])
            D[k-1] = mat
    return D

def expandBasis(mat, oldK, newK, oldKm1, newKm1):
    
    basSource = sorted(union(oldK, newK))
    basTarget = sorted(union(oldKm1, newKm1))
    for ind, b in enumerate(basSource):
        if b not in oldK:
            mat = np.insert(mat, ind, 0, axis=1)
    for ind, b in enumerate(basTarget):
        if b not in oldKm1:
            mat = np.insert(mat, ind, 0, axis=0)
    return mat

def laplacian(D, dim):
    
    Di = D[dim]
    Di1 = D[dim+1]
    return np.dot(Di.T, Di) + np.dot(Di1, Di1.T)

def densityMatrices(D, beta_list):

    rhos = []
    for ind in range(len(D)-3):
        L = laplacian(D, ind)
        M = spla.expm(beta_list[ind]*L)
        M = M / np.trace(M)
        rhos.append(M)
    return rhos

def spectralEntropies(rhos):

    ents = []
    for ind in range(len(rhos)):
        v, w = np.linalg.eig(rhos[ind])
        ve = np.log(v)
        ent = -np.dot(v.T, ve)
        ents.append(ent)
    return ents