import spectralAnalysis as sa 
import simpComp as sc 
import numpy as np 
from joblib import Parallel, delayed

class SLSEMetric:

    def __init__(self, targ_scg, d, thresh):
        self.targ_scg = targ_scg
        self.d = d
        self.thresh = thresh

    def error(self, spikes, beta):
        JSdivs = []
        (ncells, nwin, ntrials) = spikes.shape()
        scgGenSave = Parallel(n_jobs=14)(delayed(computeChainGroup)(spikes, self.thresh, trial) for trial in range(ntrials))
        for scg in scgGenSave:
            JSdivs.append(sa.compute_JS_expanded(self.targ_scg, scg, self.d, beta))
        return np.mean(np.array(JSdivs))