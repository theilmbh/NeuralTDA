import numpy as np
from scipy.interpolate import interp1d
import topology as tp
import pickle
import pandas as pd
import patsy
import statsmodels.api as sm 
import itertools
from sklearn.linear_model import LogisticRegression
import h5py as h5

class FiringRateLogisticClassPredictor:

    def __init__(self, binnedData, stimuliClasses, predNTimes=10, nCellsPerm=None, **kwargs):

        self.stimClasses = stimuliClasses
        self.nCellsPerm = nCellsPerm
        self.binnedData = h5.File(binnedData, 'r')
        self.nclus = self.binnedData.attrs['nclus']
        self.predMaxBetti = 5
        self.predNTimes = predNTimes
        self.trainedStimuliData = {}
        self.FRVecArray = []
        self.predClassArray = []
        self.trainedBinnedData = {}
        self.colnames = ['U%d' % s for s in range(self.nclus)]

    def buildPredictionDataMatrixRecursive(self, bpd, stmcls):

        if 'pop_vec' in bpd.keys():
            # We are at bottom of hierarchy
            popVec = np.array(bpd['pop_vec'])
            if nCellsPerm:
                cellSubset = np.random.permutation(self.nclus)
                cellSubset = cellSubset[0:nCellsPerm]
                popVec = popVec[cellSubset, :]
                self.colnames = self.colnames[cellSubset]

            avgFRVec = np.mean(popVec, 1)[np.newaxis, :]
            self.FRVecArray.append(avgFRVec)
            newDF = pd.DataFrame(data=avgFRVec, columns=self.colnames, index=[1])
            self.persistentBettiFrame = self.persistentBettiFrame.append(newDF, ignore_index=True)
            self.predClassArray.append(stmcls)
            return
        else:
            for indx, hLevel in enumerate(bpd.keys()):
                if hLevel in self.stimClasses.keys():
                    stmcls = self.stimClasses[hLevel]
                self.buildPredictionDataMatrixRecursive(bpd[hLevel], stmcls)
            return

    def getStimClass(self, row):

        hstr = row['hierarchy']
        for stim in self.stimClasses.keys():
            if stim in hstr:
                return self.stimClasses[stim]

    def buildPredictionDataMatrix(self):

        self.persistentBettiFrame = pd.DataFrame(columns=self.colnames)
        for stim in self.stimClasses.keys():
            self.trainedBinnedData[stim] = self.binnedData[stim]
        self.buildPredictionDataMatrixRecursive(self.trainedBinnedData, '')
        self.persistentBettiFrame['stimClass'] = self.predClassArray

    def formatModelInput(self):

        self.trainX = np.array(self.persistentBettiFrame.sample(frac=1))
        (samps, feats) = np.shape(self.trainX)
        testInd = np.round(0.2*samps) # select 20%
        self.testX = self.trainX[0:testInd, :]
        self.trainX = self.trainX[testInd:, :]

        self.trainY = self.trainX[:, -1]
        self.trainX = self.trainX[:, 0:-1]

        self.testY = self.testX[:, -1]
        self.testX = np.array(self.testX[:, 0:-1])

        self.trainY = 1.0*np.array([s is 'R' for s in self.trainY])
        self.testY = 1.0*np.array([s is 'R' for s in self.testY])

        self.shufftrainY = np.random.permutation(self.trainY)

    def fitLogistic(self):

        self.formatModelInput()
        self.tMod = LogisticRegression()
        self.tMod.fit(self.trainX, self.trainY)
        self.modelScore = self.tMod.score(self.testX, self.testY)

        self.shufftMod = LogisticRegression()
        self.shufftMod.fit(self.trainX, self.shufftrainY)
        self.shuffmodelScore = self.shufftMod.score(self.testX, self.testY)        


