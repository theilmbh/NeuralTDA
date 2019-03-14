# My code
import neuraltda.simpComp as sc
import neuraltda.topology2 as tp2
import neuraltda.spectralAnalysis as sa
from ephys import rasters

# System Libs
from importlib import reload
import pickle
import glob
import os

# Scientific Libs
import numpy as np
from scipy.optimize import brentq
import tqdm


def poisson_model_loss_generic(a, beta, E_data, ncells, nwin, n_samples, metric_func):
    """
    Produce the mean and stderr KL divergence for comparing Poisson models
    generated with parameter a compared to the SCG given  by E_data 

    Parameters
    ----------
    a : float
        Poisson firing rate parameter
    beta : float 
        Parameter for SLSE density matrix
    E_data : SCG
        Simplicial complex for target spike train
    ncells : int 
        number of cells in the population
    nwin : int
        number of time windows
    n_samples : int
        number of test spike trains to compute for each value of the parameter
    metric_func : function
        function taking (laplacian, laplacian, beta) 
        and producing the "distance"

    Returns 
    -------
    (mean, stderr)
        Mean and stderr for the samples KL divergence
    
    """

    # take a probabilities,
    # generate random configurations,
    # measure KL divergence to data,
    # report loss

    # Declare variables
    KLsave = []
    JSsave = []
    probs = a * np.ones((ncells, 1))

    # Generate new spike trains
    samples = np.random.rand(ncells, nwin, n_samples)
    probmat = np.tile(probs, (1, nwin))[:, :, np.newaxis]
    probmat = np.tile(probmat, (1, 1, n_samples))
    binMatsamples = np.greater(probmat, samples).astype(int)

    # Compute simplicial complex
    SCGs = []
    for ind in range(n_samples):

        # Compute SCG for test spike trains
        msimps = sc.binarytomaxsimplex(binMat=binMatsamples[:, :, ind], rDup=True)
        E_model = sc.simplicialChainGroups(msimps)

        # Compute Laplacians for target and tests
        Lsamp = sc.compute_laplacian(E_model, d)
        Ldata = sc.compute_laplacian(E_data, d)

        # Reconcile Laplacian dimensions
        if np.size(Lsamp) > np.size(Ldata):
            (Ldata, Lsamp) = sc.reconcile_laplacians(Ldata, Lsamp)
        else:
            (Lsamp, Ldata) = sc.reconcile_laplacians(Lsamp, Ldata)

        # Compute KL divergence for this test spike train and store it.
        KLsave.append(metric_func(Ldata, Lsamp, beta))

    # Compute mean and stderr over all the test spike trains
    m = np.mean(KLsave)
    std = np.std(KLsave)
    stderr = std / np.sqrt(n_samples)
    return (m, stderr)
