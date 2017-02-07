import neuraltda.FiringRateLogisticClassPredictor as frlcp
import glob
import os
import pickle
import numpy as np
binnedPath = './binned_data/20160923T165344Z/20160923T165344Z-25.0.binned'
stimclasses = {'E_40k':'R','F_40k':'L','G_40k':'R','H_40k':'L'}

Nlogistic = 30
accStore = np.zeros(Nlogistic)
shuffaccStore = np.zeros(Nlogistic)

tmod = frlcp.FiringRateLogisticClassPredictor(binnedPath, stimclasses)
print(tmod.colnames)
tmod.buildPredictionDataMatrix()

for tr in range(Nlogistic):
	tmod.fitLogistic()
	accStore[tr] = tmod.modelScore
	shuffaccStore[tr] = tmod.shuffmodelScore
savef = './FRpredAcc.pkl'
shuffsavef = './FRpredAccShuff.pkl'
with open(savef, 'w') as f:
	pickle.dump(accStore, f)
with open(shuffsavef, 'w') as f:
	pickle.dump(shuffaccStore, f)
print(accStore)
print(shuffaccStore)
