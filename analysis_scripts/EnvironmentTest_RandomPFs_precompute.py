##
## This script generates simulated environments with holes,
## generates simulated place fields, simulates an animal walking through tese environments,
## and simulates the resulting spike trains.  Then, it computes the simplicial complexes for these spike trains
## and computes the JS divergence betweenspike trains of the different trials
## The script then saves the results


import numpy as np
import matplotlib.pyplot as plt
import neuraltda.topology2 as tp2
import neuraltda.simpComp as sc
import neuraltda.spectralAnalysis as sa
import pandas as pd
import h5py as h5
import pickle
from joblib import Parallel, delayed

import tqdm

import os
import datetime
daystr = datetime.datetime.now().strftime('%Y%m%d')
figsavepth = '/home/btheilma/DailyLog/'+daystr+'/'
print(figsavepth)

datsavepth = '/home/btheilma/pcsim/'

# Class to define environments with holes
class TPEnv:
    
    def __init__(self, n_holes, hole_rad):
        self.xlim = [-1, 1]
        self.ylim = [-1, 1]
        self.holes = []
        self.hole_rad = hole_rad
        c = 0.75*(2*np.random.rand(2) - 1)
        for hole in range(n_holes):
            while self.hole_collide(c):
                c = 0.75*(2*np.random.rand(2) - 1)
            self.holes.append(c)
        #self.holes = 0.75*(2*np.random.rand(n_holes, 2) - 1) # keep centers in range -1, 1
        self.hole_rad = hole_rad # radius of holes
        
    def in_hole(self, x, y):
        '''
        Check to see if a point is in a hole
        '''
        for hole in self.holes:
            if np.linalg.norm(np.subtract([x, y], hole)) < self.hole_rad:
                return True
        return False
        
    def hole_collide(self, c):
        '''
        Check to see if a hole will collide with already existing holes 
        '''

        for h in self.holes:
            if np.sqrt((h[0] - c[0])**2 + (h[1] - c[1])**2) <= 2*self.hole_rad:
                return True
        return False
 
def generate_environments(N, h, numrepeats=1):
    envs = []
    for nholes in range(N):
        for r in range(numrepeats):
            envs.append(TPEnv(nholes, h))
    return envs

def convert_env_to_img(env,  NSQ):
    img = np.ones((NSQ, NSQ))
    X, Y = np.meshgrid(np.linspace(-1, 1, NSQ), np.linspace(-1, 1, NSQ))
    
    for hole in env.holes:
        hx = hole[0]
        hy = hole[1]
        diffx = X - hx*np.ones(np.shape(X))
        diffy = Y - hy*np.ones(np.shape(Y))
        dists = np.sqrt(np.power(diffx, 2) + np.power(diffy, 2))
        img[dists < env.hole_rad] = 0
        
    return img 

def compute_env_img_correlations(imgs):
    nsq, _ = np.shape(imgs[0])
    dat_mat = np.zeros((len(imgs), nsq*nsq))
    for ind,img in enumerate(imgs):
        dat_mat[ind, :] = img.flatten()
        
    cormat = np.corrcoef(dat_mat)
    return cormat
    

def generate_paths(space, n_steps, ntrials, dl):
    # pick a starting point
    final_pts = np.zeros((ntrials, n_steps, 2))
    for trial in range(ntrials):
        pts = []
        pt = (2*np.random.rand(1, 2) - 1)[0]
        while space.in_hole(pt[0], pt[1]):
            pt = (2*np.random.rand(1, 2) - 1)[0]
        #pts.append(pt)
        steps_to_go = n_steps
        while steps_to_go > 0:
            if steps_to_go % 10000 == 0:
                pass
                #print("Steps to go: {}".format(steps_to_go))
            # pick a new point
            #theta = np.pi*np.random.rand(1)[0] - np.pi/2
            
            theta = 2*np.pi*np.random.rand(1)[0]
            dx = dl*np.cos(theta)
            dy = dl*np.sin(theta)

            if (abs(pt[0]+dx) < 1 and 
               abs(pt[1]+dy) < 1 and
                not space.in_hole(pt[0]+dx, pt[1]+dy)):
                
                steps_to_go -= 1

                pt[0] = pt[0] + dx
                pt[1] = pt[1] + dy
                pts.append(np.copy(pt))
        pts = np.vstack(pts)
        final_pts[trial, :, :] = pts
    return final_pts

def generate_place_fields_random(n_fields, rad):
    
    centers =2*np.random.rand(n_fields, 2) - 1
    return (centers, rad)

def generate_place_fields(n_fields, rad):
    
    nf = np.round(np.sqrt(n_fields))
    cx = np.linspace(-1, 1, nf)
    cy = np.linspace(-1, 1, nf)
    centers = np.array([np.array((x, y)) for x in cx for y in cy])
    rads = rad*np.ones(n_fields)
    return (centers, rads)

def generate_place_fields_perturbed_lattice(n_fields, rad, stddev=0.1):
    ''' generates place fields with centers normally perturbed around a lattice'''
    nf = np.round(np.sqrt(n_fields))
    cx = np.linspace(-1, 1, nf)
    cy = np.linspace(-1, 1, nf)
    cx = cx + stddev*np.random.randn(len(cx))
    cy = cy + stddev*np.random.randn(len(cy))
    centers = np.array([np.array((x, y)) for x in cx for y in cy])
    return (centers, rad)

def generate_place_fields_CI(n_fields, rad_range,exclusion_param):
    radii = (rad_range[1] - rad_range[0])*np.random.random_sample(n_fields) + rad_range[0]
    field_c = []
    for field in range(n_fields):
        #print('field = ', field)
        # pick a center in range -1, 1
        c = 2*np.random.rand(2) - 1
        if field == 0:
            field_c.append(c)
            continue
        added = False
        collision = False
        trie = 0
        maxtries = 100
        #print('field c', field_c)
        while trie < maxtries and added == False:
            #print('trie', trie)
            for cbar_ind, cbar in enumerate(field_c):
                #print('cbar', cbar)
                if np.linalg.norm(c-cbar) < exclusion_param*radii[cbar_ind]:
                    # already a field there, try again
                    #print('collision')
                    c = 2*np.random.rand(2) - 1
                    collision = True
                    break
            if collision:
                trie+=1
                collision = False
                continue
            else:
                field_c.append(c)
                added = True
        if not added:
            field_c.append(c)
    return (np.array(field_c), radii)
            
def generate_spikes_gaussian(paths, fields, max_rate, sigma):
    
    ncell, dim = fields.shape
    ntrial, nwin, _ = paths.shape
    
    spikes = np.zeros((ncell, nwin, ntrial))

    P1 = paths[:, :, np.newaxis, :]
    C1 = fields[np.newaxis, np.newaxis, :, :]

    P1 = np.tile(P1, [1, 1, ncell, 1])
    C1 = np.tile(C1, [ntrial, nwin, 1, 1])

    S = P1 - C1
    M = np.einsum('ijkl, ijkl->ijk', S, S)
    probs = max_rate*np.exp(-1*M / (2*sigma**2))
    spikes = 1*np.greater(probs, np.random.random(np.shape(probs)))
    return np.einsum('ijk->kji', spikes)

def generate_spikes(paths, fields, max_rate, rads):
    
    ncell, dim = fields.shape
    ntrial, nwin, _ = paths.shape
    
    spikes = np.zeros((ncell, nwin, ntrial))

    P1 = paths[:, :, np.newaxis, :]
    C1 = fields[np.newaxis, np.newaxis, :, :]

    P1 = np.tile(P1, [1, 1, ncell, 1])
    C1 = np.tile(C1, [ntrial, nwin, 1, 1])

    S = P1 - C1
    M = np.einsum('ijkl, ijkl->ijk', S, S)
    M = np.sqrt(M)
    #SIGMA = sigma*np.ones(M.shape)
    SIGMA = np.tile(rads[np.newaxis, np.newaxis, :], (ntrial, nwin, 1))
    # if distance is less than sigma, then p = max_rate
    probs = max_rate*np.less(M, SIGMA)
    spikes = 1*np.greater(probs, np.random.random(np.shape(probs)))
    return np.einsum('ijk->kji', spikes)

def spikes_to_dataframe(spikes, fs, nsecs):
    (ncells, nwin, ntrial) = spikes.shape
    spikes_frame = pd.DataFrame(columns=['cluster', 'time_samples', 'recording'])
    trials_frame = pd.DataFrame(columns=['stimulus', 'time_samples', 'stimulus_end'])
    for trial in range(ntrial):
        for cell in range(ncells):
            cellspikes = np.nonzero(spikes[cell, :, trial])[0] + trial*(nsecs+2)*fs
            celldict = {'cluster': len(cellspikes)*[cell], 'time_samples': cellspikes, 'recording': len(cellspikes)*[0]}
            cellframe = pd.DataFrame(celldict)
            spikes_frame = spikes_frame.append(cellframe, ignore_index=True)
        trial_frame = pd.DataFrame({'stimulus': 'joe', 'time_samples': trial*(nsecs+2)*fs, 'stimulus_end': (trial*(nsecs+2) + nsecs)*fs, 'recording': 0}, index=[0])
        trials_frame = trials_frame.append(trial_frame, ignore_index=True)
    clusters_frame = pd.DataFrame({'cluster': range(ncells), 'quality': ncells*['Good']})
    return (spikes_frame.sort_values(by='time_samples'), trials_frame, clusters_frame)


def get_lap(trial_matrix, sh):
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

####### SIMULATION CODE ########

L = 2 # meter
vel = 0.1*L # meters / second

hole_rad = 0.3
nsecs = 30*60
fs = 10
nwin = nsecs*fs
ncells = 100

dl = vel/fs
dl = vel/1
ntrials = 10

max_rate_hz = 4 # spikes/ second
max_rate_phys = fs # spikes / second
max_rate = max_rate_hz / max_rate_phys
sigma = 0.1*L

beta = -1.0
dim = 1

max_hole = 5
nrepeats = 5
num_envs = max_hole*nrepeats 
exclusion_param = 1.1 # how far away centers of pfs must be as a multiple of hole radius. 

NSQ = 100

windt = 100.0
thresh = 6.0
period = [0,0]
dtovr = 5.0

# Generate environments
print('Generating environments...')
envs = generate_environments(max_hole, hole_rad, nrepeats)

# Generate images
print('Generating images...')
imgs = []
for env in envs:
    imgs.append(convert_env_to_img(env, NSQ))

# compute environment correlations
print('Computing environment correlations...')
corrmat = compute_env_img_correlations(imgs)

# Run Simulations
print('Generating placefields...')
(fields, rads) = generate_place_fields_CI(ncells, [sigma, 0.1*L], exclusion_param)

spikes = []

print('Generating spikes...')
for env1 in tqdm.tqdm(envs):

    fields1, rads1 = generate_place_fields_CI(ncells, [sigma, 0.1*L], exclusion_param)
    pths1 = generate_paths(env1, nwin, ntrials, dl)  # 10 walks through environment 1
    spikes1 = generate_spikes(pths1, fields1, max_rate, rads1)
    spikes.append(spikes1)


# Bin Data
SCGs = []  # storing simplicial complexes for each environment
SCGs_trials = []
laplacians_trials = []
trial_binmats = []
for ind, spikes1 in enumerate(spikes):
    print('Binning data...')
    E1s = []  # storing simplicial complexes for each trial for an environment
    tspikes, ttrials, tclust = spikes_to_dataframe(spikes1, fs=fs, nsecs=nsecs)
    tp2.build_binned_file_quick(tspikes, ttrials, tclust, windt, fs, ['Good'], period, '/home/btheilma/pcsim/placecellsimdatcudasqrt{}.binned'.format(ind), dt_overlap=dtovr)

# compute simplicial complexes
    print('Computing simplicial complexes...')
    with h5.File('/home/btheilma/pcsim/placecellsimdat{}.binned'.format(ind), 'r') as bf:
        poptens = np.array(bf['joe']['pop_tens'])
        ncell, nwin, ntrial = poptens.shape
        for trial in range(ntrial):
            binmat = sc.binnedtobinary(poptens[:, :, trial], thresh)
            trial_binmats.append(binmat)
            print(np.amax(np.sum(binmat, axis = 0)))

# precompute laplacians and laplacian spectra
laplacians_trials = Parallel(n_jobs=24)(delayed(get_lap)(binmat, None) for binmat in trial_binmats)
n_laplacians = len(laplacians_trials)

# precompute laplacian spectra
laplacian_spectra_trials = Parallel(n_jobs=24)(delayed(sc.sparse_spectrum)(L) for L in laplacians_trials)

# precompute M spectra
pairs = [(i, j) for i in range(n_laplacians) for j in range(i, n_laplacians)]
M_spec = Parallel(n_jobs=24)(delayed(get_M)(i, j, laplacians_trials[i], laplacians_trials[j]) for (i, j) in pairs)
M_spec = {(p[0], p[1]): p[2] for p in M_spec}

# Save computed spectra
with open(os.path.join(datsavepth, 'Mspectra.pkl'), 'wb') as f:
    pickle.dump(M_spec, f)
with open(os.path.join(datsavepth, 'Laplacians.pkl'), 'wb') as f:
    pickle.dump(laplacians_trials, f)
with open(os.path.join(datsavepth, 'Lapspectra.pkl'), 'wb') as f:
    pickle.dump(laplacian_spectra_trials, f)

# dists = np.zeros((num_envs*ntrials, num_envs*ntrials))

# print('Computing JS Divergences...')
# for d1 in tqdm.tqdm(range(num_envs*ntrials)):
#     for d2 in tqdm.tqdm(range(d1, num_envs*ntrials)):
#         #print(d1, d2)
#         #envA = int(d1 / ntrials)
#         #envB = int(d2 / ntrials)
#         #trialA = d1 % ntrials
#         #trialB = d2 % ntrials
#         #dists[d1, d2] = pyslsa.JS(SCGs[envA][trialA], SCGs[envB][trialB], dim, beta)
#         #print(SCGs_trials[d1].L_dim(dim), SCGs_trials[d2].L_dim(dim))
#         # print(SCGs_trials[d2].L_dim(dim))
#         dists[d1, d2] = sc.sparse_JS_SCG(SCGs_trials[d1], SCGs_trials[d2], dim, -1.0*beta)

# print('Saving...')
# with open(os.path.join(figsavepth,'environment_out_random_place_fields_cuda_sqrt_multipop_{}_{}_{}.pkl'.format(ntrials, max_hole, nrepeats)), 'wb') as f:
#     pickle.dump((dists, spikes, fields, envs, corrmat), f)
