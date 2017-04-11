import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pickle
import os
import sys
from ephys import core, events, rasters
from neuraltda import topology
import glob
import string
from scipy.io import wavfile
import scipy.signal as signal
from scipy.interpolate import interp1d
import seaborn as sns
import scipy.stats as st
sns.set_context('poster')
sns.set_style('white')

def plotBarcodeRecursive(resDict):
    
    if 'barcodes' in resDict.keys():
        barcodeDat = resDict['barcodes']
        bettis = barcodeDat.keys()
        for betti in bettis:
            bettiBarcode = barcodeDat[betti]
            nlines = len(bettiBarcode)
            ycoords = [[s,s] for s in range(nlines)]
            xcoords = [[s[0], s[1]+maxt*(s[1] < 0)] for s in bettiBarcode]
            fig = plt.figure()
            plt.plot(np.transpose(xcoords), np.transpose(ycoords))
            plt.title('Hstr: %s  Betti: %s' % (resDict['hstr'], betti))
            plt.show()
    else:
        for ind, k in enumerate(resDict.keys()):
            plotBarcodeRecursive(resDict[k])
            
def plotPersistenceDiagramRecursive(resDict):
    
    if 'barcodes' in resDict.keys():
        barcodeDat = resDict['barcodes']
        bettis = barcodeDat.keys()
        for betti in bettis:
            bettiBarcode = barcodeDat[betti]
            nlines = len(bettiBarcode)
            ycoords = [s[1]+max(max(bettiBarcode))*(s[1] < 0) for s in bettiBarcode]
            xcoords = [s[0] for s in bettiBarcode]
            fig = plt.figure()
            plt.plot(np.transpose(xcoords), np.transpose(ycoords), '.')
            plt.plot(np.linspace(0, maxt, 10), np.linspace(0, maxt, 10), 'k')
            plt.title('Hstr: %s  Betti: %s' % (resDict['hstr'], betti))
            plt.xlim([-2, maxt])
            plt.ylim([-2, maxt])
            plt.show()
    else:
        for ind, k in enumerate(resDict.keys()):
            plotPersistenceDiagramRecursive(resDict[k])


def plotBettiCurveRecursive(resDict):
    
    if 'bettis' in resDict.keys():
        bettiDat = resDict['bettis']
        bettis = len(bettiDat[0][1])
        for betti in range(bettis):
            ycoords = [s[1][betti] for s in bettiDat]
            xcoords = [s[0] for s in bettiDat]
            fig = plt.figure()
            plt.plot(np.transpose(xcoords), np.transpose(ycoords))
            plt.title('Hstr: %s  Betti: %s' % (resDict['hstr'], betti))
            plt.show()
    else:
        for ind, k in enumerate(resDict.keys()):
            plotBettiCurveRecursive(resDict[k])
            
def avgBettiRecursive(bettiDict, bettinum, runningSum, N, maxT, windt):
    if 'bettis' in bettiDict.keys():
        bettiT = np.array([s[0] for s in bettiDict['bettis']])*(windt/1000.)
        maxT = max(bettiT)
        try:
            bettiB = np.array([s[1][bettinum] for s in bettiDict['bettis']])
        except:
            
            bettiB = np.zeros(len(bettiT))
        bfunc = interp1d(bettiT, bettiB, kind='zero', bounds_error=False, fill_value=(bettiB[0], bettiB[-1]))
        t = np.linspace(0, maxT, 1000)
        bvals = bfunc(t)
        return (runningSum + bvals, N+1, maxT)
    else:
        for k in bettiDict.keys():
            runningSum, N, maxT = avgBettiRecursive(bettiDict[k], bettinum, runningSum, N, maxT, windt)
        return (runningSum, N, maxT)
    
def avgstdBettiRecursive(bettiDict, bettinum, bettiSave, N, maxT, windt ):
    if 'bettis' in bettiDict.keys():
        bettiT = np.array([s[0] for s in bettiDict['bettis']])*(windt/1000.)
        maxT = max(bettiT)
        try:
            bettiB = np.array([s[1][bettinum] for s in bettiDict['bettis']])
        except:
            
            bettiB = np.zeros(len(bettiT))
        bfunc = interp1d(bettiT, bettiB, kind='zero', bounds_error=False, fill_value=(bettiB[0], bettiB[-1]))
        t = np.linspace(0, maxT, 1000)
        bvals = np.array(bfunc(t))
        bettiSave = np.vstack((bettiSave, bvals))
        return (bettiSave, N+1, maxT)
    else:
        for k in bettiDict.keys():
            bettiSave, N, maxT = avgstdBettiRecursive(bettiDict[k], bettinum, bettiSave, N, maxT, windt)
        return (bettiSave, N, maxT) 
    
def computeAvgStdBettiCurve(resDict, stim, betti, windt):
    
    stimdata = resDict[stim]
    bettiSave = np.zeros(1000)
    bettiSave, N, maxT = avgstdBettiRecursive(stimdata, betti, bettiSave, 0, 0, windt)
    avgBetti = np.mean(bettiSave, axis=0)
    stdBetti = np.std(bettiSave, axis=0)
    semBetti = stdBetti / np.sqrt(N)
    ci95Betti = semBetti*st.t.interval(0.95, N-1)[1]
    t = np.linspace(0, maxT, len(avgBetti))
    
    return (avgBetti, stdBetti, ci95Betti, t, maxT)
    
def computeAvgBettiCurve(resDict, stim, betti, windt):
    
    stimdata = resDict[stim]
    
    runningSum, N, maxT = avgBettiRecursive(stimdata, betti, np.zeros(1000), 0, 0, windt)
    avgBetti = runningSum / float(N)
    t = np.linspace(0, maxT, len(avgBetti))
    
    return (avgBetti, t, maxT)
    
def plotAvgBettiCurves(avgBetti, t, betti, stim): 
    
    plt.figure()
    plt.plot(t, avgBetti)
    plt.title('Stim: %s Betti: %d' %(stim, betti))
    plt.ylim([0, max(avgBetti)+2])
    plt.xlim([0, max(t)])
    plt.show()

def plotStimAvgMinusTotalAvg(resDict, stim, betti):
    
    stimdata = resDict[stim]
    t = np.linspace(0, 2, 1000)
    runningSum, N = avgBettiRecursive(stimdata, betti, np.zeros(len(t)), 0, t)
    avgBetti = runningSum / float(N)
    
    totAvg, N = avgBettiRecursive(resDict, betti, np.zeros(len(t)), 0, t)
    totAvg = totAvg/float(N)
    
    
    plt.figure(figsize=(11, 8))
    plt.plot(t, avgBetti-totAvg)
    sns.despine()
    plt.title('Stim: %s Betti: %d' %(stim, betti))
    plt.ylim([-5, 5])
    plt.xlabel('Time (seconds)')
    plt.ylabel('Difference')
    plt.savefig('{}-Stim_{}-Betti_{}.png'.format(prefix, stim, betti))
    plt.show()
    