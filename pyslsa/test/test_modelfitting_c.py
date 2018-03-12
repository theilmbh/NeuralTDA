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

# Scientific Libs
import numpy as np
from scipy.optimize import brentq
import tqdm


# Do Model Fitting with C code
dim = 1
nsamples2 = 10

def binmat_to_scg_C_old(sptrain):

    # turn a binmat to a C scg
    scg = pyslsa.SCG()
    msimps = sc.binarytomaxsimplex(binMat=sptrain, rDup=True)
    for msimp in msimps:
        t = pyslsa.Simplex()
        for vert in msimp:
            t.add_vertex(vert)
        scg.add_max_simplex(t)
    return scg

def binmat_to_scg_C(sptrain):
    msimps = sc.binarytomaxsimplex(sptrain, True)
    return pyslsa.build_SCG(msimps)

def precompute_test_scg(tests):
    Emodels = []
    for ind in range(nsamples2):
        Emodels.append(binmat_to_scg_C(tests[:, :, ind]))

    return Emodels


def loss_C(a, beta):
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
ncells = 20
nwin = 1000
a = 0.02
probs = (a*np.ones((ncells, 1)))
nsamples = 2
samples = np.random.rand(ncells, nwin, nsamples)
probmat = np.tile(probs,  (1, nwin))[:, :, np.newaxis]
probmat = np.tile(probmat, (1, 1, nsamples))
binMatsamples = np.greater(probmat, samples).astype(int)

# Compute SCG for each sample
SCGs = []
for ind in range(nsamples):
    E = binmat_to_scg_C(binMatsamples[:, :, ind])
    SCGs.append(E)
Etarget = SCGs[0]
Etest = SCGs[1]

SCGs_old = []
for ind in range(nsamples):
    msimps = sc.binarytomaxsimplex(binMat=binMatsamples[:, :, ind], rDup=True)
    E = sc.simplicialChainGroups(msimps)
    SCGs_old.append(E)
Etarget_old = SCGs_old[0]
Etest_old = SCGs_old[1]


est_save = []
ntrials = 1
X = np.linspace(0.0005, 0.08, 300)

beta = -0.45
beta2 = -0.15

for t in range(ntrials):
    #print(t)
    KL=[]
    KL2 = []
    JS = []
    KLerr = []
    KLerr2 = []
    JSerr = []
    for i, x in tqdm.tqdm(enumerate(X)):
        #print(x)
        (m, stderr) = loss_C(x, beta)
        (mb2, stderrb2) = loss_C(x, beta2)
        KL.append(m)
        KLerr.append(stderr)

        KL2.append(mb2)
        KLerr2.append(stderrb2)
