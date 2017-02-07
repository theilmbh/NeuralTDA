import numpy as np
import neuraltda.TopologicalLogisticClassPredictor as tplcp
import glob
import os
import pickle

blockPath = './'
stimclasses = {'M_40k':'R','N_40k':'R','O_40k':'R','P_40k':'R', 'A_40k': 'L', 'B_40k': 'L', 'C_40k': 'L', 'D_40k':'L', 'E_40k':'L', 'F_40k':'L', 'G_40k':'L', 'H_40k':'L', 'I_40k': 'L', 'J_40k': 'L', 'K_40k':'L', 'L_40k':'L'}

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

	savef = resf+'-FamUnFamPredAcc'
	shuffsavef = resf+'-FamUnFamPredAccShuff'
	with open(savef, 'w') as f:
		pickle.dump(accStore, f)
	with open(shuffsavef, 'w') as f:
		pickle.dump(shuffaccStore, f)
	print(accStore)
	print(shuffaccStore)
