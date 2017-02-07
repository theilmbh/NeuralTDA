#########################################################################################
## Routines for computing simplicla complex structures on neural data                  ##
## Notes: .scg means "Simplicial Complex Generators"                                   ##
##                                                                                     ##
## Bradley Theilman 6 February 2017                                                    ##
#########################################################################################

import simpComp as sc
import topology2 as tp2
import h5py
import os
import pickle
import numpy as np

def computeChainGroups(blockPath, binned_datafile, thresh):
    ''' Takes a binned data file and computes the chain group generators and saves them
        Output file has 3 params in name:  Winsize-dtOverlap-Thresh.scg
    '''

    with h5py.File(binned_datafile, 'r') as bdf:
        stims = bdf.keys()
        stimGenSave = dict()
        for stim in stims:
            poptens = np.array(bdf[stim]['pop_tens'])
            (ncell, nwin, ntrial) = np.shape(poptens)
            scgGenSave = dict()
            for trial in range(ntrial):
                print('Stim: {} Trial: {}').format(stim, trial)
                popmat = poptens[:, :, trial]
                popmatbinary = sc.binnedtobinary(popmat, thresh)
                maxsimps = sc.binarytomaxsimplex(popmatbinary, rDup=True)
                scgGens = sc.simplicialChainGroups(maxsimps)
                scgGenSave[trial] = scgGens
            stimGenSave[stim] = scgGenSave

    # Create output filename
    (binFold, binFile) = os.path.split(binned_datafile)
    (binFileName, binExt) = os.path.splitext(binFile)
    scgGenFile = binFileName + '-{}.scg'.format(thresh)

    # Create scg Folder
    scgFold = os.path.join(blockPath, 'scg/')
    if not os.path.exists(scgFold):
        os.makedirs(scgFold)

    # Create output file
    scgGenFile = os.path.join(scgFold, scgGenFile)
    with open(scgGenFile, 'w') as scggf:
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

