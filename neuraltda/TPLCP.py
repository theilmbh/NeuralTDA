import numpy as np
import scipy as sp
from importlib import reload
import neuraltda.topology2 as tp2
import glob
import os
from sklearn.linear_model import LogisticRegression

def predict_stimuli_classes(betti_curves, npercurve, stimuli, stim_classes, pc_test, n_predict, shuff_Y=False):
    n_curves = len(betti_curves)
    dats = []
    pred_y = np.array([])
    pred_x = np.empty((0, n_curves*npercurve))

    for stim in stimuli:
        dats=[]
        for bc in range(n_curves):
            dats.append(betti_curves[bc][stim])
        dat = np.hstack(dats)
        n = np.shape(dat)[0]
        stim_vec = np.array(n*[stim_classes[stim]])
        pred_y = np.hstack([pred_y,stim_vec])
        pred_x = np.vstack([pred_x, dat])

    pred_Y = np.array(pred_y)
    pred_X = np.array(pred_x)

    if shuff_Y:
        pred_Y = np.random.permutation(pred_Y)
    accuracies = []
    for pred in range(n_predict):

        accuracies.append(run_prediction(pred_Y, pred_X, pc_test));

    return accuracies

def run_prediction(pred_Y, pred_X, pc_test):
    total_pts = len(pred_Y)
    ntrain = int(np.round((1-pc_test)*total_pts))
    L = LogisticRegression()
    inds = np.random.permutation(np.arange(len(pred_Y)))
    inds_train = inds[0:ntrain]
    inds_predict = inds[ntrain:]
    L.fit(pred_X[inds_train, :], pred_Y[inds_train])
    test = L.predict(pred_X[inds_predict, :])
    acc = [test[x] == pred_Y[inds_predict][x] for x in range(len(inds_predict))]
    accuracy = np.sum(acc) / len(inds_predict)
    return accuracy
