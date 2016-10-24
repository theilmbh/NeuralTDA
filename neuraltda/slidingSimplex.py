'''
slidingSimplex.py 

Compute topology on neural simplicial complex 
that decays over time

Brad Theilman 2016-10-24
'''
import os
import pickle
import h5py
import numpy as np 
import topology as tp

def sspComputeTopology(cellGroupList):
	
	# First get the cell groups:
	popvec = np.array(dataGroup['pop_vec'])
	masterCellGroups = tp.calc_cell_groups_from_binned_data(popvec,
		                                                    thresh)
	(ncells, nwin) = popvec.shape
	cgTimePoints = np.array([s[0] for s in masterCellGroups])
	for winNum in range(nwin):
		cgInds = np.logical_and(cgTimePoints > winNum, 
			                    cgTimePoints < winNum+simplexWinSize)
		simplexCGs = [masterCellGroups[i] for i in np.nonzero(cgInds)]
		tp.build_persues_input(simplexCGs, pfile)
		bettiFile = tp.run_perseus(pfile)
		bettis = []
		with open(bettiFile, 'r') as bf:
        	for bf_line in bf:
            	if len(bf_line) < 2:
                	continue
            	betti_data = bf_line.split()
            	betti_numbers = map(int, betti_data[1:])
            	bettis.append([winNum, betti_numbers])
    return bettis

def sspComputeBarcode():
	pass

def sspSlidingSimplexRecursive(dataGroup, pfileStem, hStem, 
							   bettiDict, analysisPath,
							   thresh, simplexWinSize):

	if 'pop_vec' in dataGroup.keys():
		pfileName = pfileStem +'-simplex.txt'
		pfile = os.path.join(analysisPath, pfileName)
		bettiDict['hstr'] = hStem
		bettis = sspComputeTopology(dataGroup, pfile, thresh, simplexWinSize)
		nbetti = len(betti)
		barcodeDict = dict()
		barcodeDict = sspComputeBarcode()
		bettiDict['bettis'] = bettis 
		return bettiDict
	else:
		for perm, permkey in enumerate(dataGroup.keys()):
			newDataGroup = dataGroup[permkey]
			newPfileStem = pfileStem+'-%s' % permkey
			newHStem = hStem +'-%s' % permkey 
			newBettiDict = dict()
			bettis = sspSlidingSimplexRecursive(newDataGroup, newPfileStem, 
				                                newHStem, newBettiDict, 
				                                analysisPath, thresh, 
				                                simplexWinSize)
			bettiDict[permkey] = bettis 
		return bettiDict 
	

def sspSlidingSimplex(topologyID, tpBinnedDataFile, 
					  block_path, thresh, simplexWinSize):
	
	# Set up directory hierarchy
	bdfName, ext = os.path.splitext(os.path.basename(tpBinnedDataFile))
	analysisPath = os.path.join(block_path, 'topology/%s/' % topologyID)
	if not os.path.exists(analysisPath):
		os.makedirs(analysisPath)

	bettiResultsFile = os.path.join(analysisPath, 
		                                topologyID+'-bettiResultsDict.pkl')
	bpdStim = dict()

	with h5py.File(tpBinnedDataFile, 'r') as bdf:

		stims = bdf.keys()
		nstims = len(stims)
		for stim in stims:
			bettiDict = dict()
			stimData = bdf[stim]
			hStem = stim+'-'
			nreps = len(stimData.keys())

			bettiSaveFileName = topologyID + '-stim-%s' % stim + '-betti.csv'
			bettiSaveFile = os.path.join(analysisPath, bettiSaveFileName)

			bettiPklName = topologyID + '-stim-%s' % stim + '-bettiPkl.pkl'
			bettiPkl = os.path.join(analysisPath, bettiPklName)

			pfileStem = topologyID + '-stim-%s' % stim + '-rep-'

			bettiDict = sspSlidingSimplexRecursive(stimData, pfileStem, hStem, 
												   bettiDict, analysisPath,
												   thresh, simplexWinSize)
			bpdStim[stim] = bettiDict
			with open(bettiPkl, 'r') as bp: 
				pickle.dump(bettiDict, bp)
		with open(bettiResultsFile, 'w') as bpResFile:
			pickle.dump(bpdStim, bpResFile)


