# standard
import numpy as np
import glob
import os
import pickle
import tqdm
from itertools import product
from joblib import Parallel, delayed
import datetime

# My code
import neuraltda.simpComp as sc
import neuraltda.spectralAnalysis as sa
import neuraltda.topology2 as tp2
import pyslsa


# file save path
daystr = datetime.datetime.now().strftime('%Y%m%d')
figsavepth = '/home/brad/DailyLog/'+daystr+'/'
if not os.path.exists(figsavepth):
    os.makedirs(figsavepth)
print(figsavepth)

# The goal is to compute the pairwise distances using the JS divergence between all trials of each stimuli.
# First we set up the parameters for the analysis

# Bird parameters
bps = {'B1083': '/home/brad/krista/B1083/P03S03/', 'B1075': '/home/brad/krista/B1075/P01S03/',
       'B1235': '/home/brad/krista/B1235/P02S01/', 'B1056': '/home/brad/krista/B1056/klusta/phy020516/Pen01_Lft_AP100_ML1300__Site03_Z2500__B1056_cat_P01_S03_1/',
       'B1056': '/home/brad/krista/B1056/klusta/phy020516/Pen01_Lft_AP100_ML1300__Site03_Z2500__B1056_cat_P01_S03_1/'}

birds = ['B1235']

# Binning parameters
windt = 10.0                      # milliseconds
dtovr = 0.5*windt                 # milliseconds
segment_info = [0, 0]             # use full Trial
cluster_group = ['Good']          # use just good clusters
comment = 'ForSLSE-JS'            # SLSE Computations
bdfs = {}                         # Dictionary to store bdf
scgfs = {}                         # Dictionary to store simplicial complexes

# Simplicial complex parameters
thresh = 6.0

# Now, bin the data
for bird in birds:
    block_path = bps[bird]
    bfdict = tp2.dag_bin(block_path, windt, segment_info, cluster_group=cluster_group, dt_overlap=dtovr, comment=comment)
    bdf = glob.glob(os.path.join(bfdict['raw'], '*.binned'))[0]
    print(bdf)
    bdfs[bird] = bdf

# Ok, data binned, now we compute chain groups
for bird in birds:
    block_path = bps[bird]
    #scg_f = sa.computeChainGroups(block_path, bdfs[bird], thresh, comment=comment)
    scg_f = sa.pyslsa_compute_chain_groups_binned(block_path, bdfs[bird], thresh, comment=comment)
    scgfs[bird] = scg_f

# C version - parameter sweep
# Organization of scg file is [stim]
#with open(scgfs[bird], 'rb') as scgf:
#    scg_data = pickle.load(scgf)
#    print(scg_data.keys())

scg_data = scgfs[bird]
# Logic of computation.  For each pair of stimuli,
# we compute the JS divergence between all pairs of trials.
# so we need an Nstim x Nstim x (Ntrials*Ntrials) matrix to store all the values
dims = [1,2,3]
betas = [-2.0, -1.5, -0.95,  -0.65]
stims = list(scg_data.keys())   # Get list of stimuli
Nstim = len(stims)              # Get number of stimuli
Ntrials = 15
                     # Hard coded for now...

# Create result array
# Stimulus x Stimulus x TrialPairs x Dimension x Beta
JS_divs = np.zeros((Nstim, Nstim, Ntrials*Ntrials, len(dims), len(betas)))

for k, beta in enumerate(betas):
    for l, dim in enumerate(dims):
        for i, stim1 in enumerate(stims):
            for j, stim2 in enumerate(stims):
                print("Beginning Stimulus pair: ({}, {})".format(stim1, stim2))
                # Extract scgs for each stimulis
                scg1_dat = scg_data[stim1]
                scg2_dat = scg_data[stim2]
                for (trial1, trial2) in tqdm.tqdm(product(range(Ntrials), range(Ntrials))):
                    scg1 = scg1_dat[trial1]
                    scg2 = scg2_dat[trial2]
                    JS = pyslsa.JS(scg1, scg2, dim, beta)
                    JS_divs[i, j, trial1*Ntrials + trial2, l, k] = JS
with open(os.path.join(figsavepth, 'X104_JS_divs_{}_parameter_sweep.pkl'.format(bird)), 'wb') as f:
    pickle.dump([JS_divs, betas, dims, stims, Ntrials], f)
