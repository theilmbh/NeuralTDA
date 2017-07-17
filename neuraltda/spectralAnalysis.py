#########################################################################################
## Routines for computing simplicla complex structures on neural data                  ##
## Notes: .scg means "Simplicial Complex Generators"                                   ##
##                                                                                     ##
## Bradley Theilman 6 February 2017                                                    ##
#########################################################################################

import neuraltda.simpComp as sc
import neuraltda.topology2 as tp2
import h5py
import os
import pickle
import numpy as np
from joblib import Parallel, delayed

def computeChainGroup(poptens, thresh, trial):
    '''
    Computes the Chain complex for the population data in poptens
    '''
    
    print(trial)
    popmat = poptens[:, :, trial]
    popmatbinary = sc.binnedtobinary(popmat, thresh)
    maxsimps = sc.binarytomaxsimplex(popmatbinary, rDup=True)
    # filter max simplices
    maxsimps = sorted(maxsimps, key=len)
    newms = maxsimps
    r = 1
    scgGens = sc.simplicialChainGroups(newms)
    return scgGens

def parallel_compute_chain_group(bdf, stim, thresh):
    poptens = np.array(bdf[stim]['pop_tens'])
    try:
        (ncell, nwin, ntrial) = np.shape(poptens)
    except ValueError:
        print('Empty Poptens')
        return
    if  nwin == 0:
        return
    scgGenSave = dict()
    scgGenSave = Parallel(n_jobs=14)(delayed(computeChainGroup)(poptens, thresh, trial) for trial in range(ntrial))

def computeChainGroups(blockPath, binned_datafile, thresh, comment='', shuffle=False, clusters=None):
    ''' Takes a binned data file and computes the chain group generators and saves them
        Output file has 3 params in name:  Winsize-dtOverlap-Thresh.scg
    '''
    print('Computing Chain Groups...')
    with h5py.File(binned_datafile, 'r') as bdf:
        stims = bdf.keys()
        print(stims)
        stimGenSave = dict()
        for ind, stim in enumerate(stims):
            binned_clusters = np.array(bdf[stim]['clusters'])
            poptens = np.array(bdf[stim]['pop_tens'])
            print('Stim: {}, Clusters:{}'.format(stim, str(clusters)))
            try:
                if clusters is not None:
                    poptens = poptens[np.in1d(binned_clusters, clusters), :, :]
                    print("Selecting Clusters: poptens:" + str(np.shape(poptens)))
                (ncell, nwin, ntrial) = np.shape(poptens)
            except (ValueError, IndexError):
                print('Poptens Error')
                continue


            if shuffle:
                poptens = tp2.build_shuffled_data_tensor(poptens, 1)
                poptens = poptens[:, :, :, 0]

            if  nwin == 0:
                continue

            scgGenSave = Parallel(n_jobs=14)(delayed(computeChainGroup)(poptens, thresh, trial) for trial in range(ntrial))
            print('SCGGenSave: '+str(len(scgGenSave)))
#            for trial in range(ntrial):
#                print('Stim: {} Trial: {}').format(stim, trial)
#                popmat = poptens[:, :, trial]
#                popmatbinary = sc.binnedtobinary(popmat, thresh)
#                maxsimps = sc.binarytomaxsimplex(popmatbinary, rDup=True)
#                scgGens = sc.simplicialChainGroups(maxsimps)
#                scgGenSave[trial] = scgGens
            stimGenSave[stim] = scgGenSave

    # Create output filename
    (binFold, binFile) = os.path.split(binned_datafile)
    (binFileName, binExt) = os.path.splitext(binFile)
    scg_prefix = '-{}'.format(thresh)
    if not (comment == ''):
        scg_prefix = scg_prefix + '-{}'.format(comment)
    scgGenFile = binFileName + scg_prefix + '.scg'

    # Create scg Folder
    scgFold = os.path.join(blockPath, 'scg/')
    if not os.path.exists(scgFold):
        os.makedirs(scgFold)

    # Create output file
    scgGenFile = os.path.join(scgFold, scgGenFile)
    with open(scgGenFile, 'w') as scggf:
        print(stimGenSave)
        pickle.dump(stimGenSave, scggf)

def computeSimplicialLaplacians(scgf):
    ''' Takes a path to a Simplicial Complex Generator File
        Computes Laplacians
        Stores the matrcies
    '''

    # create output file
    (scgFold, scgFile) = os.path.split(scgf)
    (scgFileName, scgExt) = os.path.splitext(scgFile)
    LapFile = scgFileName + '.laplacians'

    # Load the SCGs
    with open(scgf, 'r') as scgff:
        E = pickle.load(scgff)
    LapDict = dict()

    for stim in E.keys():
        stimSCGs = E[stim]
        trialDict = dict()
        for trial in stimSCGs.keys():
            print('Stim: {}  Trial: {}'.format(stim, trial))
            scg = stimSCGs[trial]
            D = sc.boundaryOperatorMatrix(scg)
            trialDict[trial] = sc.laplacians(D)
        LapDict[stim] = trialDict

    LapFile = os.path.join(scgFold, LapFile)
    with open(LapFile, 'w') as lpf:
        pickle.dump(LapDict, lpf)

def compute_JS_expanded(scgA, scgB, d, beta):
    '''
    Computes the Jensen-Shannon Divergence between
    simplicial complexes A and B in dimension d
    using parameter beta.
    The bases are expanded according to reconcile_laplacians
    '''
    DA = sc.boundaryOperatorMatrix(scgA)
    DB = sc.boundaryOperatorMatrix(scgB)
    LA = sc.laplacian(DA, d)
    LB = sc.laplacian(DB, d)

    (LA, LB) = sc.reconcile_laplacians(LA, LB)

    rho1 = sc.densityMatrix(LA, beta)
    rho2 = sc.densityMatrix(LB, beta)

    div = sc.JSdivergence(rho1, rho2)
    return div

def compute_JS_expanded_negativeL(scgA, scgB, d, beta):

    DA = sc.boundaryOperatorMatrix(scgA)
    DB = sc.boundaryOperatorMatrix(scgB)
    LA = sc.laplacian(DA, d)
    LB = sc.laplacian(DB, d)
    (LA, LB) = sc.reconcile_laplacians(LA, LB)
    rho1 = sc.densityMatrix(-1.0*LA, beta)
    rho2 = sc.densityMatrix(-1.0*LB, beta)
    div = sc.JSdivergence(rho1, rho2)
    return div

def compute_entropy(scgA, d, beta):

    DA = sc.boundaryOperatorMatrix(scgA)
    LA = sc.laplacian(DA, d)
    rho1 = sc.densityMatrix(LA, beta)
    div = sc.Entropy(rho1)
    return div
