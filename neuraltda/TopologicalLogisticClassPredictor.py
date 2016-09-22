import numpy as np
import topology as tp
import pickle
import pandas as pd

class TopologicalLogisticClassPredictor:

    def __init__(self, masterResults, stimuliClasses, **kwargs):

        self.stimClasses = stimuliClasses
        with open(masterResults, 'r') as f:
            self.resultsDict = pickle.load(f)

        self.trainedStimuliData = {}

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
                betti_dict[dim] = betti_num
            betti_dict['hierarchy'] = bpd['hstr']
            filtdataframe = pd.DataFrame(data=betti_dict, index=[1])
            bpdf = bpdf.append(filtdataframe, ignore_index=True)
            return bpdf
        else:
            for indx, h_level in enumerate(bpd.keys()):
                bpdf = self.bptd_recursive(bpd[h_level], bpdf)
            return bpdf

    def getStimClass(self, row):

        hstr = row['hierarchy']
        for stim in self.stimClasses.keys():
            if stim in hstr:
                return self.stimClasses[stim]


    def buildPredictionDataMatrix(self):

        self.persistentBettiFrame = pd.DataFrame(columns=['hierarchy','0', '1', '2'])
        self.persistentBettiFrame = self.bptd_recursive(self.trainedStimuliData, self.persistentBettiFrame)
        self.persistentBettiFrame['stimClass'] = self.persistentBettiFrame.apply(lambda row: self.getStimClass(row), axis=1)




