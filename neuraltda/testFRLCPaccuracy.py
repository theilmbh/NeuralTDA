import neuraltda.FiringRateLogisticClassPredictor as frlcp
import glob
import os
import pickle

binnedPath = './binned_data/'
stimclasses = {}

Nlogistic = 30
accStore = np.zeros(Nlogistic)
shuffaccStore = np.zeros(Nlogistic)

tmod = frlcp.FiringRateLogisticClassPredictor(binnedPath, stimclasses)
tmod.buildPredictionDataMatrix()

for tr in range(Nlogistic):
	tmod.fitLogistic()
	accStore[tr] = tmod.modelScore
	shuffaccStore[tr] = tmod.shuffmodelScore
savef = os.path.join(resf, '-FRpredAcc')
shuffsavef = os.path.join(resf, '-FRpredAccShuff')
with open(savef, 'w') as f:
	pickle.dump(accStore, f)
with open(shuffsavef, 'w') as f:
	pickle.dump(shuffaccStore, f)
print(accStore)
print(shuffaccStore)
