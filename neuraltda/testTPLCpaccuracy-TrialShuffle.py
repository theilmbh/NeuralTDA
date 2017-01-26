import numpy as np
import neuraltda.TopologicalLogisticClassPredictor as tplcp
import glob
import os
import pickle

blockPath = './'
stimclasses = {'E_40k':'R','F_40k':'L','G_40k':'R','H_40k':'L'}

Nlogistic = 50
accStore = np.zeros(Nlogistic)
shuffaccStore = np.zeros(Nlogistic)

masterResults = glob.glob(os.path.join(blockPath, '20161005T012837Z*masterResults.pkl'))

for resf in masterResults:

	tmod = tplcp.TopologicalLogisticClassPredictor(resf, stimclasses, predNTimes=30)
	try:
		tmod.extractTrainedStims('permuted-trialShuffled')
	except KeyError:
		print('Nope')
	tmod.buildPredictionDataMatrix()

	for tr in range(Nlogistic):

		tmod.fitLogistic()
		accStore[tr] = tmod.modelScore
		shuffaccStore[tr] = tmod.shuffmodelScore

	savef = resf+'-predictionAccuracy-TrialS'
	shuffsavef = resf+'-predictionAccuracyShuff-TrialS'
	with open(savef, 'w') as f:
		pickle.dump(accStore, f)
	with open(shuffsavef, 'w') as f:
		pickle.dump(shuffaccStore, f)
	print(accStore)
	print(shuffaccStore)
	tmod = tplcp.TopologicalLogisticClassPredictor(resf, stimclasses, predNTimes=30)
        try:
            	tmod.extractTrainedStims('permuted')
        except KeyError:
                print('Nope')
        tmod.buildPredictionDataMatrix()

        for tr in range(Nlogistic):

                tmod.fitLogistic()
                accStore[tr] = tmod.modelScore
                shuffaccStore[tr] = tmod.shuffmodelScore

        savef = resf+'-predictionAccuracy-TrialS-permutedonly'
        shuffsavef = resf+'-predictionAccuracyShuff-TrialS-permutedonly'
        with open(savef, 'w') as f:
                pickle.dump(accStore, f)
        with open(shuffsavef, 'w') as f:
                pickle.dump(shuffaccStore, f)
        print(accStore)
        print(shuffaccStore)
