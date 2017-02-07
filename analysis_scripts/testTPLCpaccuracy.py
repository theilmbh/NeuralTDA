import TopologicalLogisticClassPredictor as tplcp
import glob
import os
import pickle

blockPath = './'
stimclasses = {}

Nlogistic = 30
accStore = np.zeros(Nlogistic)
shuffaccStore = np.zeros(Nlogistic)

masterResults = glob.glob(os.path.join(blockPath, '*masterResults.pkl'))

for resf in masterResults:

	tmod = tplcp.TopologicalLogisticClassPredictor(resf, stimclasses)
	tmod.extractTrainedStims('permuted')
	tmod.buildPredictionDataMatrix()

	for tr in range(Nlogistic):

		tmod.fitLogistic()
		accStore[tr] = tmod.modelScore
		shuffaccStore[tr] = tmod.shuffmodelScore

	savef = os.path.join(resf, '-predictionAccuracy')
	shuffsavef = os.path.join(resf, '-predictionAccuracyShuff')

	with open(savef, 'w') as f:
		pickle.dump(accStore, f)
	with open(shuffsavef, 'w') as f:
		pickle.dump(shuffaccStore, f)
	print(accStore)
	print(shuffaccStore)
