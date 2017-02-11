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

def simplexUnion(E1, E2):
    ''' Calculate the union of two simplicial complexes represented as lists of generators
    '''
    sortedE = sorted([E1, E2], key=len)
    maxlen = len(sortedE[1])
    minlen = len(sortedE[0])
    for i in range(maxlen-minlen):
        sortedE[0].append([])
    newE = []
    for ind in range(maxlen):
        newE.append(sorted(union(sortedE[0][ind], sortedE[1][ind])))
    return newE

def primaryFaces(Q):
    L = []
    d = len(Q)
    Q = list(Q)
    Q.extend(Q[:d-2])
    for ind in range(d):
        s = (Q[ind:ind+(d-1)])
        L.append(tuple(sorted(s)))
    return L

def simplicialChainGroups(maxsimps):
    maxdim = max([len(s) for s in maxsimps])
    E=[[] for ind in range(maxdim+2)]
    K = list(maxsimps)
    while(len(K) > 0):
        Q = K.pop(0)
        L = primaryFaces(Q)
        k = len(Q)-1
        K = union(K, L)
        E[k] = union(E[k], L)
        E[k+1] = union(E[k+1], {Q})
    for k in range(len(E)):
        E[k] = sorted(E[k])
    return E

def boundaryOperator(Q):
    sgn = 1
    c = dict()
    Ql = list(Q)
    for ind in range(len(Ql)):
        n = Ql.pop(ind)
        if len(Ql) == 0:
            c[tuple(Ql)] = 0
        else:
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

def maskedBoundaryOperatorMatrix(E, Emask):
    ''' Emask is the simplicial Chain groups you want to mask in
        It is the chain groups of the subsimplex
    '''
    nmat = len(E)-1
    D = [[] for i in range(nmat)]
    difflen = len(E) - len(Emask)
    for ind in range(difflen):
        Emask.append([])
    for k in range(1, nmat):
        m = len(E[k-1])
        n = len(E[k])
        mat = np.zeros((m, n))
        for j in range(n):
            if E[k][j] in Emask[k]:
                c = boundaryOperator(E[k][j])
                mat[:, j] = canonicalCoordinates(c, E[k-1])
        D[k-1] = mat
    return D

def expandBasis(mat, oldK, newK, oldKm1, newKm1):
    ''' oldK: source basis 1
        newK: source basis 2
        oldKm1: target basis 1
        newKm1: target basis 2
    '''
    basSource = sorted(union(oldK, newK))
    basTarget = sorted(union(oldKm1, newKm1))
    if mat == []:
        mat = np.zeros((len(basTarget), len(basSource)))
    else:    
        for ind, b in enumerate(basSource):
            if b not in oldK:
                mat = np.insert(mat, ind, 0, axis=1)
        for ind, b in enumerate(basTarget):
            if b not in oldKm1:
                mat = np.insert(mat, ind, 0, axis=0)
    return mat

def expandBases(D1, D2, E1, E2):
    newD1 = []
    newD2 = []
    minlen = min([len(D1), len(D2)])
    for ind in range(minlen-1):
        print(ind)
        print(D1[ind], D2[ind])
        newMat1 = expandBasis(D1[ind], E1[ind+1], E2[ind+1], E1[ind], E2[ind])
        newMat2 = expandBasis(D2[ind], E2[ind+1], E1[ind+1], E2[ind], E1[ind])
        newD1.append(newMat1)
        newD2.append(newMat2)
    return (newD1, newD2)

def laplacian(D, dim):
    
    Di = np.array(D[dim])
    Di1 = np.array(D[dim+1])
    return np.dot(Di.T, Di) + np.dot(Di1, Di1.T)

def laplacians(D):

    l = len(D)
    laps = []
    for dim in range(1, len(D)-1):
        laps.append(laplacian(D, dim))
    return laps 

def densityMatrices(D, beta_list):

    rhos = []
    for ind in range(len(D)-3):
        L = laplacian(D, ind)
        try:
            M = spla.expm(beta_list[ind]*L)
        except ValueError:
            print('ValueError')
            print(L)
        M = M / np.trace(M)
        rhos.append(M)
    return rhos

def densityMatrix(L, beta):
    try: 
        M = spla.expm(beta*L)
        M = M /np.trace(M)
    except ValueError:
        print('ValueError')
        print(L)
        M = 0
    return M

def Entropy(rho, beta):

    r, w = np.linalg.eig(rho)
    np.real(r)
    ent = np.real(np.sum(np.multiply(r, np.log(r)/np.log(2.0))))
    return ent

def KLdivergence(rho, sigma):
    r, w = np.linalg.eig(rho)
    s, w = np.linalg.eig(sigma)
    r = np.real(r)
    s = np.real(s)
    div = np.sum(np.multiply(r, (np.log(r) - np.log(s))/np.log(2.0)))
    return div

def KLdivergence_matrixlog(rho, sigma):

    divMat = np.dot(rho, (spla.logm(rho) - spla.logm(sigma))/ 2.0)
    return np.trace(divMat)

def JSdivergence(rho, sigma):
    
    M = (rho+sigma)/2.0
    return (KLdivergence(rho, M) + KLdivergence(sigma, M))/2.0

def JSdivergences(rho, sigma):
    
    assert (len(rho) == len(sigma))
    div = []
    for r, s in zip(rho, sigma):
        div.append(JSdivergence(r, s))
    return div

def spectralEntropies(rhos):

    ents = []
    for ind in range(len(rhos)):
        v, w = np.linalg.eig(rhos[ind])
        ve = np.log(v)
        ent = -np.dot(v.T, ve)
        ents.append(ent)
    return ents

def stimSpaceGraph(E, D):
    ''' Takes a set of generators for the chain groups 
    and produces adjacency matrixfor the graph of the space
    '''
    E[0] = []
    Ec = [v for sl in E for v in sl]
    adj = np.zeros((len(Ec), len(Ec)))
    for k in range(1, len(E)-1):
        mat = np.array(D[k])
        lm1 = sum([len(E[i]) for i in range(k)])
        lm2 = lm1 + len(E[k])
        lm3 = sum([len(E[i]) for i in range(k+1)])
        lm4 = lm2 + len(E[k+1])
        adj[lm1:lm2, lm3:lm4] = np.abs(mat)
    adj = (adj + adj.T)
    return (adj, Ec)

def graphLaplacian(adj):
    
    D = np.diag(np.sum(adj, axis=0))
    L = D - adj
    return L

def binnedtobinary(popvec, thresh):
    '''
    Takes a popvec array from a binned data file and converts to a binary matrix according to thresh

    Parameters
    ----------
    popvec : array 
        An NCells by Nwindow array containing firing rates in that window.  
    Thresh : float 
        Multiple of average firing rate to use for thresholding
    '''

    popvec = np.array(popvec)
    Ncells, Nwin = np.shape(popvec)
    means = popvec.sum(1)/Nwin
    means = np.tile(means, (Nwin, 1)).T
    meanthr = thresh*means
    
    activeUnits = np.greater(popvec, meanthr).astype(int)
    return activeUnits

def binarytomaxsimplex(binMat, rDup=False):
    ''' 
    Takes a binary matrix and computes maximal simplices according to CI 2008 

    Parameters
    ----------
    binMat : numpy array 
        An Ncells x Nwindows array 
    '''
    if rDup:
        lexInd = np.lexsort(binMat)
        binMat = binMat[:, lexInd]
        diff = np.diff(binMat, axis=1)
        ui = np.ones(len(binMat.T), 'bool')
        ui[1:] = (diff != 0).any(axis=0)
        binMat = binMat[:, ui]
    Ncells, Nwin = np.shape(binMat)
    MaxSimps = []
    for win in range(Nwin):
        if binMat[:, win].any():
            verts = np.arange(Ncells)[binMat[:, win] == 1]
            verts = np.sort(verts)
            MaxSimps.append(tuple(verts))
    return MaxSimps

def adjacency2maxsimp(adjmat, basis):
    '''
    Converts an adjacency matrix to a list of maximum 1-simplices (edges),
    allowing SimplicialComplex to handle graphs
    '''
    maxsimps = []
    uptr = np.triu(adjmat)
    for b in basis:
        maxsimps.append((b,))
    for ind, row in enumerate(uptr):
        for targ, val in enumerate(row):
            if val >0:
                maxsimps.append(tuple(sorted((basis[ind], basis[targ]))))
    return maxsimps 