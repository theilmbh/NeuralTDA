################################################################################
## Routines for manipulating simplicial complexes                             ##
## and computing the boundary operators.                                      ##
## Follows algorithms in "Computational Homology"                             ##
## by Kaczynski, Mischaikow, and Mrozek                                       ##
## Bradley Theilman 2 February 2017                                           ##
################################################################################


####  Basically all of this deprecated by PySLSA, and the useful functions
####  Have been moved to stimulus_space.py

import numpy as np
import scipy.linalg as spla
import scipy.sparse as sp
import networkx as nx


def maxEnt(scg, dim):
    """
    Return maximum possible entropy of a simplicial complex in dimension dim
    """
    return np.log(len(scg[dim])) / np.log(2)


def union(a, b):
    return list(set(a) | set(b))


#####################################################
#### Bitwise computation of Simplicial Complexes ####
#####################################################


def num_ones(N):
    """ Return the number of ones 
        in the binary representation of N

    """
    count = 0
    while N:
        N = N & (N - 1)
        count += 1
    return count


def check_bit(N, i):
    """ 
    Checks to see if bit i is 1 in number N 
    
    """

    if N & (1 << i):
        return True
    else:
        return False


def common_get_faces(N, num_verts):
    """
    Get the faces of a simplex represented in the binary 
    number N.  N is an integer.  In the binary representation of N,
    each 1 represents the presence of a vertex

    """

    faces = []
    verts = list(range(num_verts))

    # For each possible subface
    # of the num_verts-1 simplex
    for k in range(2 ** num_verts):
        face_list = []

        # Check to see if the subface is in the simplex
        if k & N == k:

            # Dimension is number of vertices - 1
            dim = num_ones(k) - 1

            # Check which bits are set in the subface
            # These correspond to the vertices in the subface
            for j in verts:
                if check_bit(k, j):
                    face_list.append(j)
            faces.append([dim, face_list])
    return faces


def max_simp_to_binary(max_simp):
    """
    Compute a binary representation of a simplex 
    
    """
    # Start with 0
    N = 0

    # len(max_simp) is number of vertices.
    nbits = len(max_simp) + 1
    for n in range(nbits - 1):
        N += 2 ** n
    return (N, nbits)


def get_faces(max_simp):
    """
    Get all of the faces of a simplex 
    Returns a list of faces 

    """
    (N, nbits) = max_simp_to_binary(max_simp)
    faces = common_get_faces(N, nbits)
    out_faces = [[] for x in range(nbits)]
    for face in faces:
        out_faces[face[0] + 1].append(tuple(sorted(np.array(max_simp)[face[1]])))
    return out_faces


def simplicialChainGroups(maxsimps):
    """
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

    """
    maxdim = max([len(s) for s in maxsimps])
    Elen = maxdim + 1
    E = [[] for ind in range(Elen)]
    K = list(maxsimps)
    for maxsimp in K:
        faces = get_faces(maxsimp)
        for j in range(len(faces)):
            E[j] = union(E[j], faces[j])
    for k in range(Elen):
        E[k] = sorted(E[k])
    return E


###############################################################
#### KMM Computation of Simplicial Complexes (Much slower) ####
###############################################################


def simplexUnion(E1, E2):
    """
    Calculate the union of two simplicial complexes
    represented as lists of generators

    """
    sorted_E = sorted([E1, E2], key=len)
    maxlen = len(sorted_E[1])
    minlen = len(sorted_E[0])
    for i in range(maxlen - minlen):
        sorted_E[0].append([])
    new_E = []
    for ind in range(maxlen):
        new_E.append(sorted(union(sorted_E[0][ind], sorted_E[1][ind])))
    return new_E


def primaryFaces(Q):
    """
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
    """

    L = []
    dim = len(Q)
    Q = list(Q)
    Q.extend(Q[: dim - 2])
    for ind in range(dim):
        face = Q[ind : ind + (dim - 1)]
        L.append(tuple(sorted(face)))
    return L


def old_simplicialChainGroups(maxsimps):
    """
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
    """

    maxdim = max([len(s) for s in maxsimps])
    Elen = maxdim + 2
    E = [[] for ind in range(Elen)]
    K = list(maxsimps)
    while len(K) > 0:
        Q = K.pop(0)
        k = len(Q) - 1
        if k < 0:
            continue
        L = primaryFaces(Q)
        K = union(K, L)
        E[k] = union(E[k], L)
        E[k + 1] = union(E[k + 1], {Q})
    for k in range(Elen):
        E[k] = sorted(E[k])
    return E


#####################################
#### Boundary Operator Functions ####
#####################################


def boundaryOperator(Q):
    """
    Given a simplex, return its boundary operator
    in a dictionary

    """
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
        sgn = -1 * sgn
    return c


def canonicalCoordinates(c, K):
    """
    given a boundary operator dictionary,
    convert it into canonicalCoordinates using the basis K

    """

    v = np.zeros(len(K))
    for ind in range(len(K)):
        if K[ind] in c:
            v[ind] = c[K[ind]]
    return v


def boundaryOperatorMatrices(E):
    """
    Given a list of simplicial complex generators,
    Return a list of boundary operator matrices.

    """

    nmat = len(E) - 1
    D = [[] for i in range(nmat)]
    for k in range(1, nmat):
        m = len(E[k - 1])
        n = len(E[k])
        mat = np.zeros((m, n))
        for j in range(n):
            c = boundaryOperator(E[k][j])
            mat[:, j] = canonicalCoordinates(c, E[k - 1])
        D[k - 1] = mat
    return D


def boundaryOperatorMatrix(E, dim):
    """
    Return the matrix of the boundary operator
    in dimension dim for simplicial complex E

    """
    m = len(E[dim])
    n = len(E[dim + 1])
    mat = np.zeros((m, n))
    for j in range(n):
        c = boundaryOperator(E[dim + 1][j])
        mat[:, j] = canonicalCoordinates(c, E[dim])
    return mat

def sparseBoundaryOperatorMatrix(E, dim):
    """
    Return the matrix of the boundary operator
    in dimension dim for simplicial complex E

    """
    m = len(E[dim])
    n = len(E[dim + 1])
    rowind = np.empty(m*n, dtype=np.int32)
    colind = np.empty(m*n, dtype=np.int32)
    data = np.empty(m*n)
    for j in range(n):
        c = boundaryOperator(E[dim + 1][j])
        col = canonicalCoordinates(c, E[dim])
        rowind[j*m:(j+1)*m] = range(m)
        colind[j*m:(j+1)*m] = m*[j]
        data[j*m:(j+1)*m] = col
    return sp.csc_matrix((data, (rowind, colind)))

def maskedBoundaryOperatorMatrix(E, Emask):
    """ Emask is the simplicial Chain groups you want to mask in
        It is the chain groups of the subsimplex

    """
    nmat = len(E) - 1
    D = [[] for i in range(nmat)]
    difflen = len(E) - len(Emask)
    for ind in range(difflen):
        Emask.append([])
    for k in range(1, nmat):
        m = len(E[k - 1])
        n = len(E[k])
        mat = np.zeros((m, n))
        for j in range(n):
            if E[k][j] in Emask[k]:
                c = boundaryOperator(E[k][j])
                mat[:, j] = canonicalCoordinates(c, E[k - 1])
        D[k - 1] = mat
    return D


#############################
#### Laplacian Functions ####
#############################


def sparse_laplacian(E, dim):
    d_low = sparseBoundaryOperatorMatrix(E, dim)
    d_high = sparseBoundaryOperatorMatrix(E, dim+1)

    return (d_low.T).dot(d_low) + d_high.dot(d_high.T)

def sparse_reconcile_laplacians(L1, L2):
    if L1.size >= L2.size:
        L2.resize(L1.shape)
    elif L1.size < L2.size:
        L1.resize(L2.shape)
    return (L1, L2)

def sparse_reconcile_spectrum(r, s):
    assert r.ndim == 1
    assert s.ndim == 1
    if len(r) >= len(s):
        t = np.zeros(r.shape)
        t[:s.size] = s
        q = r
    else:
        q = np.zeros(s.shape)
        q[:r.size] = r
        t = s
    return (q, t)

def sparse_density_matrix(L, beta):
    ''' Computes the sparse density matrix associated to laplacian L '''
    rho = sp.linalg.expm(-beta*L.tocsc())
    rho = rho / (np.sum(rho.diagonal()))
    return rho

def sparse_spectrum(rho):
    '''
    Returns the spectrum of the sparse matrix rho
    '''
    return sp.linalg.svds(rho, k=(min(rho.shape)-1), return_singular_vectors = False)

def sparse_reconciled_spectrum_KL(r, s):
    if len(r) >= len(s):
        t = np.zeros(r.shape)
        t[:s.length] = s
        q = r
    else:
        q = np.zeros(s.shape)
        q[:r.length] = r
        t = s
    # renormalize
    q /= np.sum(q)
    t /= np.sum(t)
    return sparse_spectrum_KL(sorted(q), sorted(t))
        
def sparse_spectrum_KL(r, s):
    div = 0.0
    for rval, sval in zip(r, s):
        if rval < 1e-14 or sval < 1e-14:
            div += 0
        else:
            div += rval * (np.log(rval) - np.log(sval)) / np.log(2.0)
    return div

def sparse_KL_divergence(L1, L2, beta):
    (L1, L2) = sparse_reconcile_laplacians(L1, L2)
    rho = sparse_density_matrix(L1, beta)
    sigma = sparse_density_matrix(L2, beta)
    r = sparse_spectrum(rho)
    s = sparse_spectrum(sigma)
    KL = sparse_spectrum_KL(sorted(r), sorted(s))
    return KL

def sparse_JS_divergence(L1, L2, beta):
    (L1, L2) = sparse_reconcile_laplacians(L1, L2)
    rho = sparse_density_matrix(L1, beta)
    sigma = sparse_density_matrix(L2, beta)
    M = (rho + sigma) / 2.0
    r = sorted(sparse_spectrum(rho))
    s = sorted(sparse_spectrum(sigma))
    m = sorted(sparse_spectrum(M))
    JS = (sparse_spectrum_KL(r, m) + sparse_spectrum_KL(s, m))/2.0

    return JS

def sparse_JS_divergence2(L1, L2, beta):
    '''
    This matches the cuda code
    '''
    (L1, L2) = sparse_reconcile_laplacians(L1, L2)
    M = (L1 + L2) / 2.0
    r = np.exp(-beta* sparse_spectrum(L1))
    s = np.exp(-beta* sparse_spectrum(L2))
    m = np.exp(-beta* sparse_spectrum(M))
    r = np.array(sorted(r)) / np.sum(r)
    s = np.array(sorted(s)) / np.sum(s)
    m = np.array(sorted(m)) / np.sum(m)
    JS = (sparse_spectrum_KL(r, m) + sparse_spectrum_KL(s, m))/2.0

    return JS

def sparse_JS_divergence2_fast(L1, L2, specL1, specL2, beta):
    '''
    Uses precomputed spectra for L1 and L2
    '''
    (specL1, specL2) = sparse_reconcile_spectrum(specL1, specL2)
    (L1, L2) = sparse_reconcile_laplacians(L1, L2)
    M = (L1 + L2) / 2.0
    r = np.exp(-beta * specL1)
    s = np.exp(-beta * specL2)
    m = np.exp(-beta * sparse_spectrum(M))
    r = np.array(sorted(r)) / np.sum(r)
    s = np.array(sorted(s)) / np.sum(s)
    m = np.array(sorted(m)) / np.sum(m)
    JS = (sparse_spectrum_KL(r, m) + sparse_spectrum_KL(s, m)) / 2.0
    return JS


def sparse_JS_SCG(E1, E2, dim, beta):
    L1 = sparse_laplacian(E1, dim)
    L2 = sparse_laplacian(E2, dim)
    return sparse_JS_divergence(L1, L2, beta)

def sparse_reconciled_spectrum_JS(r, s):
    if len(r) >= len(s):
        t = np.ones(r.shape)
        t[:len(s)] = s
        q = r
    else:
        q = np.ones(s.shape)
        q[:len(r)] = r
        t = s
    # renormalize
    q = np.array(sorted(q/ np.sum(q)))
    t = np.array(sorted(t/np.sum(t)))
    m = np.array(sorted((q+t)/2.0))
    m /= np.sum(m)
    js = (sparse_spectrum_KL(q, m) + sparse_spectrum_KL(t, m))/2.0
    return js

def laplacian(D, dim):

    try:
        Di = np.array(D[dim])
    except:
        Di = []
    try:
        Di1 = np.array(D[dim + 1])
    except:
        Di1 = []

    if len(Di1) > 0:
        L2 = np.dot(Di1, Di1.T)
    else:
        L2 = np.array([0], ndmin=2)

    if len(Di) > 0:
        L1 = np.dot(Di.T, Di)
    else:
        L1 = np.array([0], ndmin=2)
    return L1 + L2


def compute_laplacian(scg, dim):
    """
    Compute the Laplacian matrix in dimension dim 
    for the simplicial complex scg 

    """
    try:
        Di = np.array(boundaryOperatorMatrix(scg, dim))
    except:
        print("error1")
        Di = []
    try:
        Di1 = np.array(boundaryOperatorMatrix(scg, dim + 1))
    except:
        print("error2")
        Di1 = []

    if len(Di1) > 0:
        L2 = np.dot(Di1, Di1.T)
    else:
        L2 = np.array([0], ndmin=2)

    if len(Di) > 0:
        L1 = np.dot(Di.T, Di)
    else:
        L1 = np.array([0], ndmin=2)
    return L1 + L2


def reconcile_laplacians(L1, L2):
    """ 
    Expand the bases so that Laplacian matrices 
    L1 and L2 have the same shape 
    """
    laps = sorted([L1, L2], key=np.size)
    L1 = laps[0]
    L2 = laps[1]
    L_new = np.zeros(L2.shape)
    try:
        (a, b) = L1.shape
        L_new[0:a, 0:b] = L1
    except ValueError:
        print("Reconcile Laplacians: L1 Size Value Error")
    return (L_new, L2)


#######################################################
#### Density Matrix and KL/JS Divergence Functions ####
#######################################################


def densityMatrix(L, beta):
    try:
        M = spla.expm(beta * L)
        M = M / np.trace(M)
    except ValueError:
        print("ValueError")
        print(L)
        M = 0
    return M


def Entropy(rho):

    r, w = np.linalg.eig(rho)
    np.real(r)
    ent = -np.real(np.sum(np.multiply(r, np.log(r) / np.log(2.0))))
    return ent


def KLdivergence_lap(LA, LB, beta):
    r, w = np.linalg.eig(LA)
    s, w = np.linalg.eig(LB)
    r = np.real(sorted(r))
    s = np.real(sorted(s))

    r = np.exp(beta * r)
    s = np.exp(beta * s)

    r = r / np.sum(r)
    s = s / np.sum(s)
    div = np.sum(np.multiply(r, (np.log(r) - np.log(s)) / np.log(2.0)))
    return div


def KLdivergence(rho, sigma):
    r = np.linalg.eigvalsh(rho)
    s = np.linalg.eigvalsh(sigma)
    # r = spla.eigh(rho, eigvals_only=True)
    # s = spla.eigh(sigma, eigvals_only=True)
    r = np.real(sorted(r))
    s = np.real(sorted(s))
    # r = np.real(r)
    # s = np.real(s)
    div = 0.0
    for rval, sval in zip(r, s):
        if rval < 1e-14 or sval < 1e-14:
            div += 0
        else:
            div += rval * (np.log(rval) - np.log(sval)) / np.log(2.0)
    # div = np.sum(np.multiply(r, (np.log(r) - np.log(s))/np.log(2.0)))
    return div


def Likelihood(rho, sigma):

    return np.linalg.det(spla.expm(np.dot(rho, spla.logm(sigma))))


def KLdivergence_matrixlog(rho, sigma):
    divMat = np.dot(rho, (spla.logm(rho) - spla.logm(sigma)) / 2.0)
    return np.trace(divMat)


def JSdivergence_matrixlog_old(rho, sigma):

    M = (rho + sigma) / 2.0
    return (KLdivergence_matrixlog(rho, M) + KLdivergence_matrixlog(sigma, M)) / 2.0


def JSdivergence(rho, sigma):
    M = (rho + sigma) / 2.0
    return (KLdivergence(rho, M) + KLdivergence(sigma, M)) / 2.0


def JSdivergence_BDD(rho, sigma):
    M = (rho + sigma) / 2.0
    return Entropy(M) - (1 / 2.0) * (Entropy(rho) + Entropy(sigma))


def JSdivergences(rho, sigma):

    assert len(rho) == len(sigma)
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


###############################################
#### Graph and Population Tensor Functions ####
###############################################


def mu_k(k, Ncells):
    """
    Cell group distance parameter from Curto Itskov 2008

    Parameters
    ----------
    k : int
        number of cells in cell group 
    Ncells : int 
        Total number of cells in population 
    """
    return 1 - np.pi * np.sqrt(float(k - 1) / float(Ncells))


def add_cellgroups(graph, cg, Ncells, depth):
    # for each neighbor:
    cg_orig = tuple(cg)
    cg_list = list(cg)
    if len(cg_list) <= 1:
        return
    k = len(cg) - 1
    muk = mu_k(k, Ncells)
    for ind in range(len(cg)):
        a = cg_list.pop(ind)
        graph.add_edge(tuple(cg_list), cg_orig, weight=muk)
        add_cellgroups(graph, cg_list, Ncells, depth + 1)
        cg_list.insert(ind, a)
    return


def stimspacegraph_nx(maxsimps, Ncells):
    """ 
    Construct the weighted graph of cell groups as defined in Curto Itskov 2008 

    Parameters 
    ----------
    maxsimps : list of tuples 
        The max simplices for the simplicial complex 
    Ncells : int 
        The total number of cells in the population (for computing metric)
    """

    g = nx.Graph()
    depth = 0
    for maxsimp in maxsimps:
        add_cellgroups(g, maxsimp, Ncells, depth)
    return g


def stimSpaceGraph(E, D):
    """ Takes a set of generators for the chain groups
    and produces adjacency matrixfor the graph of the space
    """
    E[0] = []
    Ec = [v for sl in E for v in sl]
    adj = np.zeros((len(Ec), len(Ec)))
    for k in range(1, len(E) - 1):
        mat = np.array(D[k])
        lm1 = sum([len(E[i]) for i in range(k)])
        lm2 = lm1 + len(E[k])
        lm3 = sum([len(E[i]) for i in range(k + 1)])
        lm4 = lm2 + len(E[k + 1])
        adj[lm1:lm2, lm3:lm4] = np.abs(mat)
    adj = adj + adj.T
    return (adj, Ec)


def graphLaplacian(adj):

    D = np.diag(np.sum(adj, axis=0))
    L = D - adj
    return L


def binnedtobinary(popvec, thresh):
    """
    Takes a popvec array from a binned data file and converts
    to a binary matrix according to thresh

    Parameters
    ----------
    popvec : array
        An NCells by Nwindow array containing firing rates in that window.
    Thresh : float
        Multiple of average firing rate to use for thresholding
    """

    popvec = np.array(popvec)
    Ncells, Nwin = np.shape(popvec)
    means = popvec.sum(1) / Nwin
    means = np.tile(means, (Nwin, 1)).T
    meanthr = thresh * means

    activeUnits = np.greater(popvec, meanthr).astype(int)
    return activeUnits


def binarytomaxsimplex(binMat, rDup=False, clus=None):
    """
    Takes a binary matrix and computes maximal simplices according to CI 2008

    Parameters
    ----------
    binMat : numpy array
        An Ncells x Nwindows array
    """
    if rDup:
        lexInd = np.lexsort(binMat)
        binMat = binMat[:, lexInd]
        diff = np.diff(binMat, axis=1)
        ui = np.ones(len(binMat.T), "bool")
        ui[1:] = (diff != 0).any(axis=0)
        binMat = binMat[:, ui]

    Ncells, Nwin = np.shape(binMat)
    if not clus:
        clus = np.arange(Ncells)
    MaxSimps = []
    MaxSimps = [tuple(clus[list(np.nonzero(t)[0])]) for t in binMat.T]
    # for win in range(Nwin):
    #    if binMat[:, win].any():
    #        verts = np.arange(Ncells)[binMat[:, win] == 1]
    #        verts = np.sort(verts)
    #        MaxSimps.append(tuple(verts))
    return MaxSimps


def adjacency2maxsimp(adjmat, basis):
    """
    Converts an adjacency matrix to a list of maximum 1-simplices (edges),
    allowing SimplicialComplex to handle graphs
    """
    maxsimps = []
    uptr = np.triu(adjmat)
    for b in basis:
        maxsimps.append((b,))
    for ind, row in enumerate(uptr):
        for targ, val in enumerate(row):
            if val > 0:
                maxsimps.append(tuple(sorted((basis[ind], basis[targ]))))
    return maxsimps
