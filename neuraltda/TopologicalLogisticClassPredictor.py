import numpy as np
from scipy.interpolate import interp1d
import topology as tp
import pickle
import pandas as pd
import patsy
import statsmodels.api as sm 
import itertools
from sklearn.linear_model import LogisticRegression

class TopologicalLogisticClassPredictor:

    def __init__(self, masterResults, stimuliClasses, predNTimes=10, shuffle='ClassLabels', **kwargs):

        self.stimClasses = stimuliClasses
        self.shuffle = shuffle
        with open(masterResults, 'r') as f:
            self.resultsDict = pickle.load(f)
        self.predMaxBetti = 5
        self.predNTimes = predNTimes
        self.trainedStimuliData = {}
        self.persistentBettiArray = []
        self.predClassArray = []
        self.colnames = ['B%d_T%d' % s for s in itertools.product(range(self.predMaxBetti), range(self.predNTimes))]

    def extractTrainedStims(self, computationClass):

        clData = self.resultsDict[computationClass]
        for stim in clData.keys():
            if stim in self.stimClasses.keys():
                self.trainedStimuliData[stim] = clData[stim]

    def bptd_recursive(self, bpd, bpdf):

        if 'hstr' in bpd.keys():
            barcode = bpd['barcodes']
            
            betti_dict = dict()
            for dim in barcode.keys():
                nPersist = sum([-1 in s for s in barcode[dim]])
                betti_dict['B%s' % dim] = nPersist
            betti_dict['hierarchy'] = bpd['hstr']
            filtdataframe = pd.DataFrame(data=betti_dict, index=[1])
            bpdf = bpdf.append(filtdataframe, ignore_index=True)
            return bpdf
        else:
            for indx, h_level in enumerate(bpd.keys()):
                bpdf = self.bptd_recursive(bpd[h_level], bpdf)
            return bpdf

    def buildPredictionDataMatrixRecursive(self, bpd):

        if 'hstr' in bpd.keys():
            # We are at bottom of hierarchy
            bettiData = bpd['bettis']
            curveStore = np.zeros((self.predMaxBetti, self.predNTimes))
            filts = [s[0] for s in bettiData]
            bettis = [s[1] for s in bettiData]
            t = np.linspace(min(filts), max(filts), self.predNTimes)
            for dim in range(self.predMaxBetti):
                try:
                    bettiVals = [s[1][dim] for s in bettiData]
                except IndexError:
                    bettiVals = [0 for s in bettiData]
                bettifunc = interp1d(filts, bettiVals, kind='zero',
                                     bounds_error=False,
                                     fill_value=(bettiVals[0], bettiVals[-1]))
                bettiCurve = bettifunc(t)
                curveStore[dim, :] = bettiCurve
            bettiDataRow = np.reshape(curveStore, (1, self.predMaxBetti*self.predNTimes))
            self.persistentBettiArray.append(bettiDataRow)
            newDF = pd.DataFrame(data=bettiDataRow, columns=self.colnames, index=[1])
            self.persistentBettiFrame = self.persistentBettiFrame.append(newDF, ignore_index=True)
            hstr = bpd['hstr']
            for stim in self.stimClasses.keys():
                if stim in hstr:
                    stmcls = self.stimClasses[stim]
                    self.predClassArray.append(stmcls)
            return
        else:
            for indx, hLevel in enumerate(bpd.keys()):
                self.buildPredictionDataMatrixRecursive(bpd[hLevel])
            return

    def getStimClass(self, row):

        hstr = row['hierarchy']
        for stim in self.stimClasses.keys():
            if stim in hstr:
                return self.stimClasses[stim]

    def buildPredictionDataMatrix(self):

        self.persistentBettiFrame = pd.DataFrame(columns=self.colnames)
        self.buildPredictionDataMatrixRecursive(self.trainedStimuliData)
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
        if self.shuffle = 'ClassLabels':

            self.shufftrainY = np.random.permutation(self.trainY)
            self.shuffletrainX = np.copy(self.trainX)
        else:
            self.shufftrainY = np.copy(self.trainY)
            (a,b) = np.shape(self.trainX)
            self.shuffletrainX = np.copy(self.trainX)
            self.shuffletrainX = np.reshape(np.random.permutation(np.reshape(self.shuffletrainX), (a*b, 1)), (a,b))


    def fitLogistic(self):

        self.formatModelInput()
        self.tMod = LogisticRegression()
        self.tMod.fit(self.trainX, self.trainY)
        self.modelScore = self.tMod.score(self.testX, self.testY)

        self.shufftMod = LogisticRegression()
        self.shufftMod.fit(self.shuffletrainX, self.shufftrainY)
        self.shuffmodelScore = self.shufftMod.score(self.testX, self.testY)        


