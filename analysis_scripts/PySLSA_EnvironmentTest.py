import numpy as np
import pyslsa
import matplotlib.pyplot as plt
import neuraltda.topology2 as tp2
import neuraltda.simpComp as sc
import neuraltda.spectralAnalysis as sa
import pandas as pd
import h5py as h5
import pickle

import tqdm


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
        for hole in self.holes:
            if np.linalg.norm(np.subtract([x, y], hole)) < self.hole_rad:
                return True
        return False
        
    def hole_collide(self, c):
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
                print("Steps to go: {}".format(steps_to_go))
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
    return (centers, rad)

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

def generate_spikes(paths, fields, max_rate, sigma):
    
    ncell, dim = fields.shape
    ntrial, nwin, _ = paths.shape
    
    spikes = np.zeros((ncell, nwin, ntrial))

    P1 = paths[:, :, np.newaxis, :]
    C1 = fields[np.newaxis, np.newaxis, :, :]

    P1 = np.tile(P1, [1, 1, ncell, 1])
    C1 = np.tile(C1, [ntrial, nwin, 1, 1])

    S = P1 - C1
    M = np.einsum('ijkl, ijkl->ijk', S, S)
    SIGMA = sigma*np.ones(M.shape)
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


L = 2 # meter
vel = 0.1*L # meters / second

hole_rad = 0.2
nsecs = 10*60
fs = 10
nwin = nsecs*fs
ncells = 70

dl = vel/fs
dl = vel/1
ntrials = 10

max_rate_hz = 3 # spikes/ second
max_rate_phys = fs # spikes / second
max_rate = max_rate_hz / max_rate_phys
sigma = 0.1*L

beta = -1.0
dim = 1

max_hole = 5
nrepeats = 5
num_envs = max_hole*nrepeats 

NSQ = 100

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
(fields, rad) = generate_place_fields(ncells, sigma)

spikes = []

print('Generating spikes...')
for env1 in envs:

    pths1 = generate_paths(env1, nwin, ntrials, dl)  # 10 walks through environment 1
    spikes1 = generate_spikes(pths1, fields, max_rate, sigma)
    spikes.append(spikes1)


# Bin Data
SCGs = []  # storing simplicial complexes for each environment
SCGs_trials = []
for ind, spikes1 in enumerate(spikes):
    print('Binning data...')
    E1s = []  # storing simplicial complexes for each trial for an environment
    tspikes, ttrials, tclust = spikes_to_dataframe(spikes1, fs=fs, nsecs=nsecs)
    tp2.build_binned_file_quick(tspikes, ttrials, tclust, 100.0, fs, ['Good'], [0, 0], '/home/brad/placecellsimdat{}.binned'.format(ind), dt_overlap=5.0)

# compute simplicial complexes
    print('Computing simplicial complexes...')
    with h5.File('/home/brad/placecellsimdat{}.binned'.format(ind), 'r') as bf:
        poptens = np.array(bf['joe']['pop_tens'])
        ncell, nwin, ntrial = poptens.shape
        for trial in range(ntrial):
            binmat = sc.binnedtobinary(poptens[:, :, trial], 6.0)
            print(np.amax(np.sum(binmat, axis = 0)))
            maxsimps = sc.binarytomaxsimplex(binmat, rDup=True)
            E1 = pyslsa.build_SCG(maxsimps)
            E1s.append(E1)
            SCGs_trials.append(E1)
    SCGs.append(E1s)


dists = np.zeros((num_envs*ntrials, num_envs*ntrials))

print('Computing JS Divergences...')
for d1 in tqdm.tqdm(range(num_envs*ntrials)):
    for d2 in range(d1, num_envs*ntrials):
        #print(d1, d2)
        #envA = int(d1 / ntrials)
        #envB = int(d2 / ntrials)
        #trialA = d1 % ntrials
        #trialB = d2 % ntrials
        #dists[d1, d2] = pyslsa.JS(SCGs[envA][trialA], SCGs[envB][trialB], dim, beta)
        dists[d1, d2] = pyslsa.JS(SCGs_trials[d1], SCGs_trials[d2], dim, beta)

print('Saving...')
with open('/home/brad/environment_out22.pkl', 'wb') as f:
    pickle.dump((dists, spikes, fields, envs, corrmat), f)
