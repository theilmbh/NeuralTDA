################################################################################
## Routines for reconstructing stimulus space from simplicial complexes       ##
## Also routines for computing max simplices from binned spike data           ##
## Brad Theilman 30 March 2018												  ##
################################################################################

import numpy as np
import scipy.linalg as spla
import networkx as nx
from sklearn.manifold import MDS 

###############################################
#### Graph and Population Tensor Functions ####
###############################################

def mu_k(k, Ncells):
    '''
    Cell group distance parameter from Curto Itskov 2008

    Parameters
    ----------
    k : int
        number of cells in cell group 
    Ncells : int 
        Total number of cells in population 
    '''
    return 1 - np.pi*np.sqrt(float(k-1)/float(Ncells))

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
        add_cellgroups(graph, cg_list, Ncells, depth+1)
        cg_list.insert(ind, a)
    return

def stimspacegraph_nx(maxsimps, Ncells, stimuli=None):
    ''' 
    Construct the weighted graph of cell groups as defined in Curto Itskov 2008 

    Parameters 
    ----------
    maxsimps : list of tuples 
        The max simplices for the simplicial complex 
    Ncells : int 
        The total number of cells in the population (for computing metric)
    '''
    g = nx.Graph()

    
    depth = 0
    for maxsimp in maxsimps:
        add_cellgroups(g, maxsimp, Ncells, depth)

    if stimuli is not None:
        vals = dict()
        for ind, cg, in enumerate(maxsimps):
            vals[cg] = stimuli[ind, :]
        nx.set_node_attributes(g, 'stimulus', vals)
    return g

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


def binarytomaxsimplex(binMat, rDup=False, clus=None, ):
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
    MaxSimps = [tuple(clus[list(np.nonzero(t)[0])]) for t in binMat.T if list(np.nonzero(t)[0])]
    #for win in range(Nwin):
    #    if binMat[:, win].any():
    #        verts = np.arange(Ncells)[binMat[:, win] == 1]
    #        verts = np.sort(verts)
    #        MaxSimps.append(tuple(verts))


    return MaxSimps

# def binarytomaxsimplex_withstim(binMat, rDup=False, clus=None, stimulus=None):
#     '''
#     Takes a binary matrix and computes maximal simplices according to CI 2008

#     Parameters
#     ----------
#     binMat : numpy array
#         An Ncells x Nwindows array
#     '''
#     if rDup:
#         lexInd = np.lexsort(binMat)
#         binMat = binMat[:, lexInd]
#         diff = np.diff(binMat, axis=1)
#         ui = np.ones(len(binMat.T), 'bool')
#         ui[1:] = (diff != 0).any(axis=0)
#         binMat = binMat[:, ui]

#     Ncells, Nwin = np.shape(binMat)
#     stims = dict()
#     if not clus:
#         clus = np.arange(Ncells)
#     MaxSimps = []
#     MaxSimps = [tuple(clus[list(np.nonzero(t)[0])]) for t in binMat.T if list(np.nonzero(t)[0])]
#     #for win in range(Nwin):
#     #    if binMat[:, win].any():
#     #        verts = np.arange(Ncells)[binMat[:, win] == 1]
#     #        verts = np.sort(verts)
#     #        MaxSimps.append(tuple(verts))

#     if stimulus is not None:
        
#         inds = np.nonzero((np.sum(binMat, axis=0) > 0)[0]
#         for cg, ind in zip(MaxSimps, inds):
#             stims[cg] = stimulus[ind, :]
#     return (MaxSimps, stims)

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

def prepare_affine_data(binmat, stim, embed_pts, sorted_node_list):

    '''
    Prepare the data for training an affine transformation
    x -> y where x is MDS point of embedded stimulus space and 
    y is the actual stimulus
    '''

    stimdim, nwin = np.shape(stim)
    _, embeddim = np.shape(embed_pts)
    # make sure we have same number of windows in stim as neural data
    assert np.shape(binmat)[1] == nwin 


    maxsimps = binarytomaxsimplex(binmat)
    inds = np.nonzero((np.sum(binmat, axis=0) > 0))[0]
    y = np.zeros((stimdim, len(inds)))
    x = np.zeros((embeddim, len(inds)))
    
    cg_stims = {}

    for ptind, (cg, ind) in enumerate(zip(maxsimps, inds)):
        if cg not in cg_stims.keys():
            cg_stims[cg] = [stim[:, ind]]
        else:
            cg_stims[cg].append(stim[:, ind])

        y[:, ptind] = stim[:, ind]
        x[:, ptind] = get_mds_position_of_cg(cg, embed_pts, sorted_node_list)
    return (x, y)

def mds_embed(graph):

    sorted_node_list = sorted(list(graph.nodes()), key=len)
    dmat = nx.floyd_warshall_numpy(graph, nodelist=sorted_node_list)

    gmds = MDS(n_jobs=-2, dissimilarity='precomputed')
    embed_pts = gmds.fit_transform(dmat)

    return (embed_pts, dmat, sorted_node_list)

def get_mds_position_of_cg(cg, embed_pts, sorted_node_list):

    return embed_pts[sorted_node_list.index(cg), :]

def prepare_stimulus(stimfile):
    pass

def affine_loss(affine, x, y, stimdim, embeddim):

    A = np.array(affine[0:stimdim*embeddim])
    b = np.array(affine[stimdim*embeddim:])
    A = np.reshape(A, (stimdim, embeddim))
    yhat = np.dot(A, x)
    yhat += np.tile(b[:, np.newaxis], (1, np.shape(x)[1]))

    s = np.power(yhat - y, 2)
    s = np.sum(s, axis=0)
    return np.sum(s)

def decompose_matrix(m):
    ''' 
    decomposes a matrix into:
    - isotropic
    - symmetric traceless
    - antisymmetric 
    '''
    n, n1 = np.shape(m)
    assert n1 == n
    tr = np.trace(m)
    tr = tr*np.eye(n) / n

    symm = (1/2)*((m-tr) + (m-tr).T)
    asymm = (1/2)*(m - m.T)

    return (tr, symm, asymm)

def affine_transform(affine, x, sd, ed):
    A = np.reshape(affine[0:sd*ed], (sd, ed))
    b = affine[sd*ed:]
    yhat = np.dot(A, x) + np.tile(b[:, np.newaxis], (1, np.shape(x)[1]))
    return yhat