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
import pandas as pd
import h5py as h5
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt


import neuraltda.topology2 as tp2
import neuraltda.spectralAnalysis as sa
import neuraltda.simpComp as sc
import pycuslsa as pyslsa

daystr = datetime.datetime.now().strftime('%Y%m%d')
figsavepth = '/home/brad/DailyLog/'+daystr+'/'
print(figsavepth)

# Set up birds and block_paths
birds = ['B1083', 'B1056', 'B1235', 'B1075']
bps = {'B1083': '/home/brad/krista/B1083/P03S03/', 'B1075': '/home/brad/krista/B1075/P01S03/',
       'B1235': '/home/brad/krista/B1235/P02S01/', 'B1056': '/home/brad/krista/B1056/klusta/phy020516/Pen01_Lft_AP100_ML1300__Site03_Z2500__B1056_cat_P01_S03_1/',
       'B1056': '/home/brad/krista/B1056/klusta/phy020516/Pen01_Lft_AP100_ML1300__Site03_Z2500__B1056_cat_P01_S03_1/',
       'B1083-5': '/home/brad/krista/B1083/P03S05/'}

bps = {'B1083': '/home/AD/btheilma/krista/B1083/P03S03/', 'B1075': '/home/AD/btheilma/krista/B1075/P01S03/',
       'B1235': '/home/AD/btheilma/krista/B1235/P02S01/', 'B1056': '/home/AD/btheilma/krista/B1056/klusta/phy020516/Pen01_Lft_AP100_ML1300__Site03_Z2500__B1056_cat_P01_S03_1/',
       'B1056': '/home/AD/btheilma/krista/B1056/klusta/phy020516/Pen01_Lft_AP100_ML1300__Site03_Z2500__B1056_cat_P01_S03_1/',
       'B1083-5': '/home/AD/btheilma/krista/B1083/P03S05/'}

learned_stimuli = {'B1083': ['M_scaled_burung', 'N_scaled_burung', 'O_scaled_burung', 'P_scaled_burung'], 'B1056': ['A_scaled_burung', 'B_scaled_burung', 'C_scaled_burung', 'D_scaled_burung'], 'B1235': [], 'B1075': []}
peck_stimuli = {'B1083': {'L': ['N_40k','P_40k'], 'R': ['M_40k', 'O_40k']}, 'B1056': {'L': ['B_scaled_burung', 'D_scaled_burung'], 'R': ['A_scaled_burung', 'C_scaled_burung']}, 
                'B1235': {'L': ['F_scaled_burung', 'H_scaled_burung'], 'R': ['E_scaled_burung', 'G_scaled_burung'],}, 'B1075': {'L': ['F_40k', 'H_40k'], 'R': ['E_40k', 'G_40k']},
               'B1083-5': {'L': ['N_40k','P_40k'], 'R': ['M_40k', 'O_40k']}}

unfamiliar_stimuli = {'B1083': ['I_40k', 'J_40k', 'K_40k', 'L_40k'], 'B1083-5': ['I_40k', 'J_40k', 'K_40k', 'L_40k']}

#bps =  {'B1056': '/home/AD/btheilma/krista/B1056/klusta/phy020516/Pen01_Lft_AP100_ML1300__Site03_Z2500__B1056_cat_P01_S03_1/',
#        'B1235': '/home/AD/btheilma/krista/B1235/P02S01/'}
# test_birds = ['B1075', 'B1083','B1056', 'B1235']
# test_birds = ['B1075', 'B1235']
# #test_birds = ['B1056', 'B1235']
# #test_birds =['B1056', 'B1083']
# #test_birds = ['B1083']

test_birds = ['B1083', 'B1083-5']
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


poptens = {'familiar': population_tensors_familiar, 'unfamiliar': population_tensors_unfamiliar}
# Compute JS popA:
#Left vs right

dim = 1
beta = -1.0

for bird in test_birds:
    for sh in ['original', 'shuffled']:
        for fam in ['familiar', 'unfamiliar']:
            ntrials = 20 # Only do half the trials for each stim
            bird_tensors = poptens[fam][bird]
            SCG = []
            for bird_tensor, stim in bird_tensors:
                binmatlist = []
                print(bird, stim)
                ncells, nwin, _ = bird_tensor.shape
                bin_tensor = threshold_poptens(bird_tensor, threshold)
                for trial in tqdm(range(ntrials)):
                    if sh == 'shuffled':
                        binmatlist.append(shuffle_binmat(bin_tensor[:, :, trial]))
                    else:
                        binmatlist.append(bin_tensor[:, :, trial])
                    ms = sc.binarytomaxsimplex(bin_tensor[:, :, trial], rDup=True)
                    SCG.append(pyslsa.build_SCG(ms))
            N = len(SCG)
            jsmat = np.zeros((N, N))
            for i in tqdm(range(N)):
                for j in tqdm(range(i, N)):
                    jsmat[i, j] = pyslsa.cuJS(SCG[i], SCG[j], dim, beta)
            with open(os.path.join(figsavepth, 'JSpop_{}-{}-{}-{}_LvsR-{}-{}'.format(bird, dim, beta, ntrials, fam, sh)), 'wb') as f:
                pickle.dump(jsmat, f)
            
