import numpy as np
import scipy as sp
from importlib import reload
import neuraltda.topology2 as tp2
import glob
import os
from sklearn.linear_model import LogisticRegression


def predict_stimuli_classes(
    betti_curves, stimuli, stim_classes, pc_test, n_predict, shuff_Y=False
):
    """
    Run n_predict logist regressions to classify stimuli according to their
    betti curves.  Return the list of accuracies for each model
    """

    (ndims, ntimes, ntrials) = betti_curves[list(betti_curves.keys())[0]].shape
    pred_y = np.array([])
    pred_x = np.empty((0, ndims * ntimes))

    for stim in stimuli:
        dat = np.reshape(betti_curves[stim], (ndims * ntimes, ntrials)).T
        stim_vec = np.array(ntrials * [stim_classes[stim]])
        pred_y = np.hstack([pred_y, stim_vec])
        pred_x = np.vstack([pred_x, dat])

    pred_Y = np.array(pred_y)
    pred_X = np.array(pred_x)

    if shuff_Y:
        pred_Y = np.random.permutation(pred_Y)
    accuracies = []
    for pred in range(n_predict):
        accuracies.append(run_prediction(pred_Y, pred_X, pc_test))

    return accuracies


def assign_arbitary_classes(stimuli, class_labels):
    """
    Assign arbitrary class labels to the stimuli
    """

    n_labels = len(class_labels)

    assert len(stimuli) % n_labels == 0
    stride = int(len(stimuli) / n_labels)
    perms = np.random.permutation(np.arange(len(stimuli)))
    stimuli_classes = {}
    for labn in range(n_labels):
        stim_num = perms[(labn * stride) : (labn * stride) + stride]
        for num in stim_num:
            stimuli_classes[stimuli[num]] = class_labels[labn]
    return stimuli_classes


def predict_arbitrary_classes(
    betti_curves, stimuli, stim_class_labels, pc_test, n_predict, shuff_Y=False
):
    """
    Attempt to predict arbitary class labels from stimuli betti curves
    """
    (ndims, ntimes, ntrials) = betti_curves[list(betti_curves.keys())[0]].shape
    print(stimuli)
    accuracies = []
    stim_classes = assign_arbitary_classes(list(stimuli), stim_class_labels)
    for pred in range(n_predict):
        pred_y = np.array([])
        pred_x = np.empty((0, ndims * ntimes))
        #     stim_classes = assign_arbitary_classes(list(stimuli), stim_class_labels)
        for stim in stimuli:
            dat = np.reshape(betti_curves[stim], (ndims * ntimes, ntrials)).T
            stim_vec = np.array(ntrials * [stim_classes[stim]])
            pred_y = np.hstack([pred_y, stim_vec])
            pred_x = np.vstack([pred_x, dat])

        pred_Y = np.array(pred_y)
        pred_X = np.array(pred_x)

        if shuff_Y:
            pred_Y = np.random.permutation(pred_Y)
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
    acc = L.score(pred_X[inds_predict, :], pred_Y[inds_predict])
    # acc = [test[x] == pred_Y[inds_predict][x] for x in range(len(inds_predict))]
    # accuracy = np.sum(acc) / len(inds_predict)
    return acc
