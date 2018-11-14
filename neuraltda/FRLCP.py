import numpy as np
import scipy as sp
from importlib import reload
import neuraltda.topology2 as tp2
import glob
import os
from sklearn.linear_model import LogisticRegression


def predict_stimuli_classes_population_FR(
    pop_tensors,
    dimensionality,
    stimuli,
    stim_classes,
    pc_test,
    n_predict,
    shuff_Y=False,
):

    (ncells, nwin, ntrials, nperms) = pop_tensors[stimuli[0]].shape
    print(nperms)
    wins = [int(round(x)) for x in np.linspace(0, nwin - 1, dimensionality)]
    pred_y = np.array([])
    pred_x = np.empty((0, dimensionality))

    for stim in stimuli:
        pt = np.reshape(pop_tensors[stim], (ncells, nwin, ntrials * nperms))
        print(pt.shape)
        dat = np.mean(pt, axis=0)
        dat = dat[wins, :].T
        stim_vec = np.array(ntrials * nperms * [stim_classes[stim]])
        pred_y = np.hstack([pred_y, stim_vec])
        pred_x = np.vstack([pred_x, dat])

    pred_Y = np.array(pred_y)
    pred_X = np.array(pred_x)
    print(len(pred_Y))

    if shuff_Y:
        pred_Y = np.random.permutation(pred_Y)
    accuracies = []
    for pred in range(n_predict):
        accuracies.append(run_prediction(pred_Y, pred_X, pc_test))

    return accuracies


def run_prediction(pred_Y, pred_X, pc_test):
    """
    Perform a prediciton based on logistic regression
    """
    total_pts = len(pred_Y)
    ntrain = int(np.round((1 - pc_test) * total_pts))
    print("total pts: {}, ntrain: {}".format(total_pts, ntrain))
    L = LogisticRegression()
    inds = np.random.permutation(np.arange(len(pred_Y)))
    inds_train = inds[0:ntrain]
    inds_predict = inds[ntrain:]
    L.fit(pred_X[inds_train, :], pred_Y[inds_train])
    test = L.predict(pred_X[inds_predict, :])
    acc = [test[x] == pred_Y[inds_predict][x] for x in range(len(inds_predict))]
    accuracy = np.sum(acc) / len(inds_predict)
    return accuracy
