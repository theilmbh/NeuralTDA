import numpy as np
import topology as tp
import pickle
import pandas as pd
import patsy
import statsmodels.api as sm 
import itertools

class TopologicalLogisticClassPredictor:

    def __init__(self, masterResults, stimuliClasses, **kwargs):

        self.stimClasses = stimuliClasses
        with open(masterResults, 'r') as f:
            self.resultsDict = pickle.load(f)
        self.predMaxBetti = 5
        self.predNTimes = 10
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
            newDF = pd.DataFrame(data=bettiDataRow, columns=colnames, index=[1])
            self.persistentBettiFrame = self.persistentBettiFrame.append(bettiDataRow, ignore_index=True)
            hstr = bpd['hstr']
            for stim in self.stimClasses.keys():
                if stim in hstr:
                    stmcls = self.stimClasses[stim]
                    self.predClassArray.append(stmcls)
            return
        else:
            for indx, hLevel in enumerate(bpd.keys()):
                self.buildPredictionDataMatrixRecursive(bpd)
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

        formula = 'C(stimClass) ~ B0 + B1 + B2 + B3 + B4 + B5'

        self.y, self.X = patsy.dmatrices(formula, self.persistentBettiFrame, return_type='dataframe')

    def fitLogistic(self):

        self.formatModelInput()
        sm.Logit(self.y['C(stimClass)[L]'], self.X).fit().summary()



