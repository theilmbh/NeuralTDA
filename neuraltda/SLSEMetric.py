import neuraltda.spectralAnalysis as sa 
import neuraltda.simpComp as sc 
import numpy as np 
from joblib import Parallel, delayed

class SLSEMetric:

    def __init__(self, targ_scg, d, thresh):
        self.targ_scg = targ_scg
        self.d = d
        self.thresh = thresh

    def loss(self, spikes, beta):
        JSdivs = []
        (ncells, nwin, ntrials) = spikes.shape
        scgGenSave = [sa.computeChainGroup(spikes, self.thresh, trial) for trial in range(ntrials)]
        for scg in scgGenSave:
            JSdivs.append(sa.compute_JS_expanded(self.targ_scg, scg, self.d, beta))
        return np.mean(np.array(JSdivs))

class FrobeniusLoss:

    def __init__(self, targ_scg, d, thresh):
        self.targ_scg = targ_scg
        self.d = d
        self.thresh = thresh
        self.DA = sc.boundaryOperatorMatrix(self.targ_scg)
        self.LA = sc.laplacian(self.DA, d)

    def loss(self, spikes, beta):

        norms = []
        (ncells, nwin, ntrials) = spikes.shape
        scgGenSave = [sa.computeChainGroup(spikes, self.thresh, trial) for trial in range(ntrials)]
        for scg in scgGenSave:
            
            DB = sc.boundaryOperatorMatrix(scg)
            LB = sc.laplacian(DB, self.d)
            (LA, LB) = sc.reconcile_laplacians(self.LA, LB)
            Ldiff = (LA - LB)
            norms.append(np.linalg.norm(Ldiff)**2)
        return np.mean(np.array(norms))