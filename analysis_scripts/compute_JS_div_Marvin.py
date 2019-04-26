###
# The goal of this notebook is to:
# - Take two neural populations
# - Compute the JS divergence between stimuli pairs for each population (the same stimuli pairs)
# - Compute the mutual information between the distributions of JS divergences

import glob
import os
from importlib import reload
import pickle
import datetime

import numpy as np
import scipy as sp

import h5py as h5
from tqdm import tqdm as tqdm



import neuraltda.topology2 as tp2

import neuraltda.simpComp as sc

from joblib import Parallel, delayed

homef = 'brad'

daystr = datetime.datetime.now().strftime('%Y%m%d')
datsavepth = '/home/{}/marvinTDAdat/'.format(homef)
print(datsavepth)

mdatf = '/home/{}/marvinTDA.pkl'.format(homef)
with open(mdatf, 'rb') as f:
    mdat = pickle.load(f)
# flatten the list of population tensors for each population
threshold = 2

def threshold_poptens(tens, thresh):
    ncell, nwins, ntrials = tens.shape
    frs = np.mean(tens, axis=1)
    tfr = thresh*frs
    tfrtens = np.tile(tfr[:, np.newaxis, :], (1, nwins, 1))
    bintens = 1*np.greater(tens, tfrtens)
    return bintens

def shuffle_binmat(binmat):
    ncells, nwin = binmat.shape
    for i in range(ncells):
        binmat[i, :] = np.random.permutation(binmat[i, :])
    return binmat

def get_JS(i, j, Li, Lj, speci, specj, beta):
    js = (i, j, sc.sparse_JS_divergence2_fast(Li, Lj, speci, specj, beta))
    print((i, j))
    return js

def get_JS_spec(i, j, speci, specj, specm, beta):
	js = (i, j, sc.sparse_JS_divergence2_spec(speci, specj, specm, beta))
	return js

def get_Lap(trial_matrix, sh, trial):
    print(trial)
    if sh == 'shuffled':
        mat = shuffle_binmat(trial_matrix)
    else:
        mat = trial_matrix
    ms = sc.binarytomaxsimplex(trial_matrix, rDup=True)
    scg1 = sc.simplicialChainGroups(ms)
    L = sc.sparse_laplacian(scg1, dim)
    return L

def get_M(i, j, L1, L2):
	mspec = sc.compute_M_spec(L1, L2)
	print((i, j))
	return (i, j, mspec)


# mirroring cuda code
#Left vs right
reload(sc)
dim = 1

betas = [1, 0.25, 0.5, 2, 3]

poptensors = mdat['pop_tensors']
blocks = list(poptensors.keys())

for sh in ['original', 'shuffled']:

    for block in blocks:

                block_tensors = poptensors[block]
                SCG = []
                spectra = []
                laplacians_save = []
                print('Computing Laplacians for {} {}...'.format(block, sh))
                for morphdim in block_tensors.keys():
                    block_tensor = block_tensors[morphdim]
                    binmatlist = []
                    print(morphdim)
                    ncells, nwin, ntrials = block_tensor.shape
                    print(ncells, nwin, ntrials)
                    bin_tensor = threshold_poptens(block_tensor, threshold)
                    laps = Parallel(n_jobs=24)(delayed(get_Lap)(bin_tensor[:, :, trial], sh, trial) for trial in range(ntrials))
                    laplacians_save.append((block, morphdim, laps))
                    laplacians = laps
                    N = len(laplacians)
                    # compute spectra
                    print('Computing Spectra...')
                    spectra = Parallel(n_jobs=24)(delayed(sc.sparse_spectrum)(L) for L in laplacians)

                    # Precompute M spectra
                    pairs = [(i, j) for i in range(N) for j in range(i, N)]
                    print('Computing M spectra...')
                    M_spec = Parallel(n_jobs=24)(delayed(get_M)(i, j, laplacians[i], laplacians[j]) for (i, j) in pairs)
                    M_spec = {(p[0], p[1]): p[2] for p in M_spec}
                    
                    # Save computed spectra
                    with open(os.path.join(datsavepth, 'Mspectra_{}_{}_{}.pkl'.format(block, morphdim, sh)), 'wb') as f:
                        pickle.dump(M_spec, f)
                    with open(os.path.join(datsavepth, 'Laplacians_{}_{}_{}.pkl'.format(block, morphdim, sh)), 'wb') as f:
                        pickle.dump(laplacians_save, f)
                    with open(os.path.join(datsavepth, 'Lapspectra_{}_{}_{}.pkl'.format(block, morphdim, sh)), 'wb') as f:
                        pickle.dump(spectra, f)
                    # compute density matrices
            