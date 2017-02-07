import numpy as np
import neuraltda.TopologicalLogisticClassPredictor as tplcp
import glob
import os
import pickle

blockPath = './'
stimclasses = {'A_40k':'R','J_40k':'L','P_40k':'R','O_40k':'L'}

Nlogistic = 30
accStore = np.zeros(Nlogistic)
shuffaccStore = np.zeros(Nlogistic)

masterResults = glob.glob(os.path.join(blockPath, '*masterResults.pkl'))

for resf in masterResults:

	tmod = tplcp.TopologicalLogisticClassPredictor(resf, stimclasses, predNTimes=30)
	tmod.extractTrainedStims('permuted')
	tmod.buildPredictionDataMatrix()

	for tr in range(Nlogistic):

		tmod.fitLogistic()
		accStore[tr] = tmod.modelScore
		shuffaccStore[tr] = tmod.shuffmodelScore

	savef = resf+'-predictionAccuracy-ArbUnfam'
	shuffsavef = resf+'-predictionAccuracyShuff-ArbUnfam'
	with open(savef, 'w') as f:
		pickle.dump(accStore, f)
	with open(shuffsavef, 'w') as f:
		pickle.dump(shuffaccStore, f)
	print(accStore)
	print(shuffaccStore)
