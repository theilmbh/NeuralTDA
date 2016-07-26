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

def compute_avg_betti_recursive(bettidata, betticurves, t):
    
    if type(bettidata) is dict:
        for permnum, perm in enumerate(bettidata.keys()):
            betticurves = compute_avg_betti_recursive(bettidata[perm], betticurves, t)
        return betticurves
    else:
        betti1 = np.zeros([len(bettidata), 2])
        for ind, filt in enumerate(bettidata):
            betti1[ind, 0] = filt[0]*windt/1000.
            try: 
                betti1[ind, 1] = filt[1][betti]
            except IndexError:
                betti1[ind, 1] = 0
                
        bettifunc = interp1d(betti1[:, 0], betti1[:, 1], kind='zero', bounds_error=False, fill_value=(betti1[0, 1], betti1[-1, 1]))
        betticurve = bettifunc(t)
        betticurves = np.vstack((betticurves, betticurve))
        return betticurves

def plot_betti_trace_recursive(trialdata, ax, upper):
    
    if type(trialdata) is dict:
        for permnum, perm in enumerate(trialdata.keys()):
            new_trialdata = trialdata[perm]
            new_upper=plot_betti_trace_recursive(new_trialdata, ax, upper)
            upper = new_upper
        return upper
    else:
        betti1 = np.zeros([len(trialdata), 2])
        for ind, filt in enumerate(trialdata):
            betti1[ind, 0] = filt[0]*(windt/1000.)
            try: 
                betti1[ind, 1] = filt[1][betti]
            except IndexError:
                betti1[ind, 1] = 0
            
        upper = max([upper, max(betti1[:, 1])])
        pltcolor='b'
        ax.plot(betti1[:, 0], betti1[:,1], pltcolor, lw=2)
        ax.set_title('Stimulus: {}'.format(stimname))
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Betti Number Value')
        ax.set_ylim([0, upper+1])
        return upper

def get_persistence_files(topology_folder):

	persistence_files = sorted(glob.glob(os.path.join(topology_folder, '*.pkl')))
	return persistence_files

def extract_metadata(topology_file):

	tf_path, tf_name = os.path.split(topology_file)
	tf_name_split = tf_name.split('-')

	pen = tf_name_split[0]
	cluster_group = tf_name_split[1]
	dt = tf_name_split[2]
	period = tf_name_split[3]
	analysis_id = tf_name_split[5]
	stimname = tf_name_split[7]

	pen_split = pen.split('_')
	birdID = pen_split[0]
	penID = pen_split[2]
	siteID = pen_split[3]

	metadata_dict = dict()
	metadata_dict['birdID'] = birdID
	metadata_dict['penID'] = penID
	metadata_dict['siteID'] = siteID

	metadata_dict['dt'] = dt
	metadata_dict['period'] = period
	metadata_dict['stimname'] = stimname
	metadata_dict['analysis_id'] = analysis_id
	metadata_dict['cluster_group'] = cluster_group
	return metadata_dict
	

