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

daystr = datetime.datetime.now().strftime('%Y%m%d')
figsavepth = '/home/btheilma/DailyLog/'+daystr+'/'
print(figsavepth)

# Set up birds and block_paths
birds = ['B1083', 'B1056', 'B1235', 'B1075']
bps = {'B1083': '/home/btheilma/krista/B1083/P03S03/', 'B1075': '/home/btheilma/krista/B1075/P01S03/',
       'B1235': '/home/btheilma/krista/B1235/P02S01/', 'B1056': '/home/btheilma/krista/B1056/klusta/phy020516/Pen01_Lft_AP100_ML1300__Site03_Z2500__B1056_cat_P01_S03_1/',
       'B1056': '/home/btheilma/krista/B1056/klusta/phy020516/Pen01_Lft_AP100_ML1300__Site03_Z2500__B1056_cat_P01_S03_1/',
       'B1083-5': '/home/btheilma/krista/B1083/P03S05/'}


learned_stimuli = {'B1083': ['M_scaled_burung', 'N_scaled_burung', 'O_scaled_burung', 'P_scaled_burung'], 'B1056': ['A_scaled_burung', 'B_scaled_burung', 'C_scaled_burung', 'D_scaled_burung'], 'B1235': [], 'B1075': []}
peck_stimuli = {'B1083': {'L': ['N_40k','P_40k'], 'R': ['M_40k', 'O_40k']}, 'B1056': {'L': ['B_scaled_burung', 'D_scaled_burung'], 'R': ['A_scaled_burung', 'C_scaled_burung']}, 
                'B1235': {'L': ['F_scaled_burung', 'H_scaled_burung'], 'R': ['E_scaled_burung', 'G_scaled_burung'],}, 'B1075': {'L': ['F_40k', 'H_40k'], 'R': ['E_40k', 'G_40k']},
               'B1083-5': {'L': ['N_40k','P_40k'], 'R': ['M_40k', 'O_40k']}}

unfamiliar_stimuli = {'B1083': ['I_40k', 'J_40k', 'K_40k', 'L_40k'], 
                      'B1083-5': ['I_40k', 'J_40k', 'K_40k', 'L_40k'],
                      'B1235': ['A_scaled_burung', 'B_scaled_burung', 'C_scaled_burung', 'D_scaled_burung'], 
                      'B1075': ['A_40k', 'B_40k', 'C_40k', 'D_40k'], 
                      'B1056': ['E_scaled_burung', 'F_scaled_burung', 'G_scaled_burung', 'H_scaled_burung']
                     }

#bps =  {'B1056': '/home/AD/btheilma/krista/B1056/klusta/phy020516/Pen01_Lft_AP100_ML1300__Site03_Z2500__B1056_cat_P01_S03_1/',
#        'B1235': '/home/AD/btheilma/krista/B1235/P02S01/'}
#test_birds = ['B1056', 'B1235']
#test_birds = ['B1075', 'B1235']
#test_birds = ['B1056', 'B1235']
#test_birds =['B1056', 'B1083']
#test_birds = ['B1083']
#test_birds = ['B1083', 'B1083-5']
test_birds = ['B1056', 'B1235', 'B1083', 'B1083-5']
#test_birds = ['B1056']
# Binning Parameters
windt = 10.0                      # milliseconds
dtovr = 0.5*windt                 # milliseconds
segment_info = [0, 0]             # use full Trial
cluster_group = ['Good']          # use just good clusters
comment = 'JS_MI_TEST'            # BootStrap Populations
bdfs = {}                         # Dictionary to store bdf

# Loop through each bird in our list and bin the data
for bird in test_birds:
    block_path = bps[bird]
    bfdict = tp2.dag_bin(block_path, windt, segment_info, cluster_group=cluster_group, dt_overlap=dtovr, comment=comment)
    bdf = glob.glob(os.path.join(bfdict['raw'], '*.binned'))[0]
    print(bdf)
    bdfs[bird] = bdf

# extract left vs right stims
# extract population tensors for the populations of interest
# Do not sort the stims
population_tensors_familiar = {}
stimuli = []

for bird in test_birds:
    stimuli = peck_stimuli[bird]['L'] + peck_stimuli[bird]['R']
    print(stimuli)
    bdf = bdfs[bird]
    population_tensors_familiar[bird] = []
    # open the binned data file
    with h5.File(bdf, 'r') as f:
        #stimuli = f.keys()
        print(list(f.keys()))
        for stim in stimuli:
            poptens = np.array(f[stim]['pop_tens'])
            population_tensors_familiar[bird].append([poptens, stim])

# extract Unfamiliar stims
# extract population tensors for the populations of interest
# Do not sort the stims
population_tensors_unfamiliar = {}
stimuli = []

for bird in test_birds:
    stimuli = unfamiliar_stimuli[bird]
    print(stimuli)
    bdf = bdfs[bird]
    population_tensors_unfamiliar[bird] = []
    # open the binned data file
    with h5.File(bdf, 'r') as f:
        #stimuli = f.keys()
        print(list(f.keys()))
        for stim in stimuli:
            poptens = np.array(f[stim]['pop_tens'])
            population_tensors_unfamiliar[bird].append([poptens, stim])

# flatten the list of population tensors for each population
threshold = 6

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

def get_Lap(trial_matrix, sh):
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

poptens = {'familiar': population_tensors_familiar, 'unfamiliar': population_tensors_unfamiliar}

# mirroring cuda code
#Left vs right
reload(sc)
dim = 1

betas = [1, 0.25, 0.5, 2, 3]

for bird in test_birds:
    for sh in ['original', 'shuffled']:
        for fam in ['familiar', 'unfamiliar']:
            ntrials = 20 # Only do half the trials for each stim
            bird_tensors = poptens[fam][bird]
            SCG = []
            spectra = []
            laplacians_save = []
            print('Computing Laplacians for {} {} {}...'.format(bird, sh, fam))
            for bird_tensor, stim in bird_tensors:
                binmatlist = []
                print(bird, stim)
                ncells, nwin, _ = bird_tensor.shape
                bin_tensor = threshold_poptens(bird_tensor, threshold)
                laps = Parallel(n_jobs=24)(delayed(get_Lap)(bin_tensor[:, :, trial], sh) for trial in range(ntrials))
                laplacians_save.append((bird, stim, laps))
            laplacians = sum([s[2] for s in laplacians_save], [])
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
            with open(os.path.join(figsavepth, 'Mspectra_{}-{}-{}-{}.pkl'.format(bird, ntrials, sh, fam)), 'wb') as f:
                pickle.dump(M_spec, f)
            with open(os.path.join(figsavepth, 'Laplacians_{}-{}-{}-{}.pkl'.format(bird, ntrials, sh, fam)), 'wb') as f:
                pickle.dump(laplacians_save, f)
            with open(os.path.join(figsavepth, 'Lapspectra_{}-{}-{}-{}.pkl'.format(bird, ntrials, sh, fam)), 'wb') as f:
                pickle.dump(spectra, f)
            # compute density matrices
            
            for beta in betas:
                print('Computing JS Divergences with beta {}...'.format(beta))
                jsmat = np.zeros((N, N))
                
                jsdat = Parallel(n_jobs=24)(delayed(get_JS_spec)(i, j, spectra[i], spectra[j], M_spec[(i,j)], beta) for (i, j) in pairs)
                for d in jsdat:
                    jsmat[d[0], d[1]] = d[2]
            
                with open(os.path.join(figsavepth, 'JSpop_fast_{}-{}-{}-{}_LvsR-{}-{}.pkl'.format(bird, dim, beta, ntrials, fam, sh)), 'wb') as f:
                    pickle.dump(jsmat, f)
