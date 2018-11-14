import os
import subprocess
import time
import glob
import pickle
import logging
import datetime
import tqdm
import tempfile

import numpy as np
import h5py
from scipy.interpolate import interp1d

from ephys import events, core

import neuraltda.simpComp as sc
import neuraltda.topology2 as tp2


def get_population_firing_rate(binned_datafile, shuffle=False, clusters=None):

    stim_tensors = tp2.extract_population_tensors(binned_datafile, shuffle, clusters)
    stim_firing_rates = {}

    for stim in stim_tensors.keys():
        stim_tensor = stim_tensors[stim]
        stim_frs = np.mean(stim_tensor, axis=0)
        stim_firing_rates[stim] = stim_frs
    return stim_firing_rates
