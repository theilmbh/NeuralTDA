# My code
import neuraltda.simpComp as sc
import neuraltda.topology2 as tp2
import neuraltda.spectralAnalysis as sa
from ephys import rasters
import pyslsa

# System Libs
from importlib import reload
import pickle
import glob
import os
import datetime

# Scientific Libs
import numpy as np
from scipy.optimize import brentq
import tqdm
# file save path
daystr = datetime.datetime.now().strftime('%Y%m%d')
figsavepth = '/home/brad/DailyLog/'+daystr+'/'
print(figsavepth)

def binmat_to_scg_C(sptrain):
    msimps = sc.binarytomaxsimplex(sptrain, True)
    return pyslsa.build_SCG(msimps)


def loss_C(a, beta, dim, ncells, nsamples2):
    # take a probabilities,
    # generate random configurations,
    # measure KL divergence to data,
    # report loss

    # Declare variables
    KLsave = []
    JSsave = []
    probs = (a*np.ones((ncells, 1)))

    # Generate new spike trains
    samples = np.random.rand(ncells, nwin, nsamples2)
    probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
    probmat = np.tile(probmat, (1, 1, nsamples2))
    binMatsamples = np.greater(probmat, samples).astype(int)
    #Emodels = precompute_test_scg(binMatsamples)

    # Compute simplicial complex
    SCGs = []
    for ind in range(nsamples2):
        maxsimps = sc.binarytomaxsimplex(binMatsamples[:, :, ind], rDup=True)
        # Compute SCG for test spike trains
        #Emodel = Emodels[ind]
        Emodel = pyslsa.build_SCG(maxsimps)

        # Compute KL divergence for this test spike train and store it.
        #%time div = pyslsa.KL(Etarget, Emodel, dim, beta)
        div = pyslsa.KL(Etarget, Emodel, dim, beta)
        KLsave.append(div)

    # Compute mean and stderr over all the test spike trains
    m = np.mean(KLsave)
    std = np.std(KLsave)
    stderr = std / np.sqrt(nsamples2)
    return (m, stderr)

# Generate binary matrix with given probabilities for each "cell"
ncellss = [10, 20, 50, 100]
nwin = 1000
a_s = [600.0/(x*1000) for x in ncellss] # conserve number of spikes at target FR
betas = np.linspace(-0.15, -2, 25)
dims = [1,2,3,4]
KL = np.zeros((100, len(ncellss), len(betas), len(dims)))
KLerr = np.zeros((100, len(ncellss), len(betas), len(dims)))
Ptests = np.zeros((100, len(ncellss), len(betas), len(dims)))
for I, ncell in enumerate(ncellss):

    a = 600.0/(ncell*1000)
    probs = (a*np.ones((ncell, 1)))
    nsamples = 1
    samples = np.random.rand(ncell, nwin, nsamples)
    probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
    probmat = np.tile(probmat, (1, 1, nsamples))
    binMatsamples = np.greater(probmat, samples).astype(int)

    # Compute SCG for each sample
    SCGs = []
    for ind in range(nsamples):
        E = binmat_to_scg_C(binMatsamples[:, :, ind])
        SCGs.append(E)
    Etarget = SCGs[0]

    ntrials = 1
    x_low = 10.0/(ncell*1000.)
    x_hi = 1600.0/(ncell*1000.)
    X = np.linspace(x_low, x_hi, 100)
    for K, beta in enumerate(betas):
        for L, dim in enumerate(dims):
            print((I, K, L))
            for t in range(ntrials):
                for M, x in tqdm.tqdm(enumerate(X)):
                    #print(x)
                    (m, stderr) = loss_C(x, beta, dim, ncell, 10)
                    KL[M, I, K, L] = m
                    KLerr[M, I, K, L] = stderr
                    Ptests[M, I, K, L] = x
with open(os.path.join(figsavepth, 'pyslsa_modelfitting_paramsweep_out2.pkl'), 'wb') as f:
    pickle.dump([KL, KLerr, Ptests, ncellss, betas, dims], f)
