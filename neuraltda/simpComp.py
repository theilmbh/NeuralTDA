################################################################################
## Routines for manipulating simplicial complexes                             ##
## and computing the boundary operators.                                      ##
## Follows algorithms in "Computational Homology"                             ##
## by Kaczynski, Mischaikow, and Mrozek                                       ##
## Bradley Theilman 2 February 2017                                           ##
################################################################################

import numpy as np
import scipy.linalg as spla

def maxEnt(scg, dim):
    '''
    Return maximum possible entropy of a simplicial complex in dimension dim
    '''
    return np.log(len(scg[dim]))/np.log(2)

def union(a, b):
    return list(set(a) | set(b))

def simplexUnion(E1, E2):
    ''' Calculate the union of two simplicial complexes
        represented as lists of generators
    '''
    sorted_E = sorted([E1, E2], key=len)
    maxlen = len(sorted_E[1])
    minlen = len(sorted_E[0])
    for i in range(maxlen-minlen):
        sorted_E[0].append([])
    new_E = []
    for ind in range(maxlen):
        new_E.append(sorted(union(sorted_E[0][ind], sorted_E[1][ind])))
    return new_E

def primaryFaces(Q):
    '''
    Take a simplex and return its primary faces
    These are the faces of one dimension less.

    Parameters
    ----------
    Q : tuple
        tuple of vertices defining a simplex

    Returns
    -------
    L : list
        list of simplices for the primary faces of Q
    '''

    L = []
    dim = len(Q)
    Q = list(Q)
    Q.extend(Q[:dim-2])
    for ind in range(dim):
        face = (Q[ind:ind+(dim-1)])
        L.append(tuple(sorted(face)))
    return L

def simplicialChainGroups(maxsimps):
    '''
    Take a list of maximal simplices and
    successively add faces until all generators
    of the chain groups are present

    Parameters
    ----------
    maxsimps : list of tuples
        list of the maximal simplices in the complex

    Returns
    -------
    E : list of lists
        simplicial complex generators in each dimension
    '''

    maxdim = max([len(s) for s in maxsimps])
    Elen = maxdim+2
    E = [[] for ind in range(Elen)]
    K = list(maxsimps)
    while(len(K) > 0):
        Q = K.pop(0)
        k = len(Q)-1
        if k < 0:
            continue
        L = primaryFaces(Q)
        K = union(K, L)
        E[k] = union(E[k], L)
        E[k+1] = union(E[k+1], {Q})
    for k in range(Elen):
        E[k] = sorted(E[k])
    return E

def boundaryOperator(Q):
    '''
    Given a simplex, return its boundary operator
    in a dictionary
    '''

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
    '''
    given a boundary operator dictionary,
    convert it into canonicalCoordinates using the basis K

    '''

    v = np.zeros(len(K))
    for ind in range(len(K)):
        if c.has_key(K[ind]):
            v[ind] = c[K[ind]]
    return v

def boundaryOperatorMatrix(E):
    '''
    Given a list of simplicial complex generators,
    Return a list of boundary operator matrices.
    '''

    nmat = len(E)-1
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

    try:
        Di = np.array(D[dim])
    except:
        Di = []
    try:
        Di1 = np.array(D[dim+1])
    except:
        Di1 = []

    if len(Di1) >0:
        L2 = np.dot(Di1, Di1.T)
    else:
        L2 = np.array([0], ndmin=2)

    if len(Di) > 0:
        L1 = np.dot(Di.T, Di)
    else:
        L1 = np.array([0], ndmin=2)
    return L1 + L2

def reconcile_laplacians(L1, L2):
    laps = sorted([L1, L2], key=np.size)
    L1 = laps[0]
    L2 = laps[1]
    L_new = np.zeros(L2.shape)
    try:
        (a,b) = L1.shape
        L_new[0:a, 0:b] = L1
    except ValueError:
        print('Reconcile Laplacians: L1 Size Value Error')
    return (L_new, L2)

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

def Entropy(rho):

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
    Takes a popvec array from a binned data file and converts
    to a binary matrix according to thresh

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

def binarytomaxsimplex(binMat, rDup=False, clus=None):
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
    if not clus:
        clus = np.arange(Ncells)
    MaxSimps = []
    MaxSimps = [tuple(clus[list(np.nonzero(t)[0])]) for t in binMat.T]
    #for win in range(Nwin):
    #    if binMat[:, win].any():
    #        verts = np.arange(Ncells)[binMat[:, win] == 1]
    #        verts = np.sort(verts)
    #        MaxSimps.append(tuple(verts))
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
            if val > 0:
                maxsimps.append(tuple(sorted((basis[ind], basis[targ]))))
    return maxsimps
