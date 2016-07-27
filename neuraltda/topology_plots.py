import numpy as np
import scipy as sp
import matplotlib
matplotlib.use('Agg')
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
from itertools import groupby

def compute_avg_betti_recursive(bettidata, betticurves, betti, windt, t):
    
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

    dt_parsed = [''.join(g) for _, g in groupby(dt, str.isalpha)]
    dt_value = float(dt_parsed[0])
    dt_units = dt_parsed[1]
    metadata_dict = dict()
    metadata_dict['birdID'] = birdID
    metadata_dict['penID'] = penID
    metadata_dict['siteID'] = siteID

    metadata_dict['dt'] = dt_value
    metadata_dict['dt_units'] = dt_units
    metadata_dict['period'] = period
    metadata_dict['stimname'] = stimname
    metadata_dict['analysis_id'] = analysis_id
    metadata_dict['cluster_group'] = cluster_group
    metadata_dict['tf_name'] = tf_name
    return metadata_dict


def plot_average_betti(persistence_files, betti, maxt, figsize, plot_savepath):
    plot_savepath = os.path.abspath(plot_savepath)
    t = np.linspace(0, maxt, num=1000)
    bettiStimspline=[]
    
    nplots = len(persistence_files)
    print(nplots)
    nsubplotrows = np.round(nplots/4)
    print(nsubplotrows)
    subplot_shape = (nsubplotrows, 4)
    fig, axs = plt.subplots(nsubplotrows, 4, figsize=figsize)


    for pf_num, pf in enumerate(persistence_files):
        pf_metadata = extract_metadata(pf)
        stimname = pf_metadata['stimname']
        dt = pf_metadata['dt']
        prd = pf_metadata['period']
        pdata = pickle.load(open(pf, 'r'))
        upper=0
        bettiTrialspline=[]
        ax = axs.flatten()[pf_num]
        betticurves = np.empty_like(t)
        betticurves = compute_avg_betti_recursive(pdata, betticurves, betti, dt, t)
        avgbetticurve = np.mean(betticurves[1:], axis=0)
        ax.plot(t, avgbetticurve, lw=2)
        ax.set_title('Stimulus: {}'.format(stimname))
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Betti {} Value'.format(betti))
    plt.savefig(plot_savepath+'B{}_betti{}_{}ms_{}_permuted_avg_withshuffled.png'.format(bird, betti, dt, prd))


def plot_all_bettis(persistence_files, maxbetti, maxt, figsize, plot_savepath):

    for betti in range(maxbetti):
        plot_average_betti(persistence_files, betti, maxt, figsize, plot_savepath)

def plot_average_betti_with_shuffled(persistence_files, persistence_files_shuffled, betti, maxt, figsize, plot_savepath, difference=False):
    plot_savepath = os.path.abspath(plot_savepath)
    t = np.linspace(0, maxt, num=1000)
    bettiStimspline=[]
    
    nplots = len(persistence_files)
    print(nplots)
    nsubplotrows = np.round(nplots/4)
    print(nsubplotrows)
    subplot_shape = (nsubplotrows, 4)
    fig, axs = plt.subplots(nsubplotrows, 4, figsize=figsize)


    for pf_num, pf, pf_shuff in enumerate(zip(persistence_files, persistence_files_shuffled)):
        pf_metadata = extract_metadata(pf)
        stimname = pf_metadata['stimname']
        dt = pf_metadata['dt']
        prd = pf_metadata['period']

        ax = axs.flatten()[pf_num]

        pdata = pickle.load(open(pf, 'r'))
        pdata_shuff = pickle.load(open(pf_shuff, 'r'))
        betticurves = np.empty_like(t)
        betticurves = compute_avg_betti_recursive(pdata, betticurves, betti, dt, t)
        avgbetticurve = np.mean(betticurves[1:], axis=0)
    
        bc_shuff = np.empty_like(t)
        bc_shuff = compute_avg_betti_recursive(pdata_shuff, bc_shuff, betti, dt, t)
        avgbcshuff = np.mean(bc_shuff[1:], axis=0)
        if difference:
            ax.plot(t, avgbetticurve-avgbcshuff, 'b', lw=2)
        else:
            ax.plot(t, avgbetticurve, 'r',lw=2)
            ax.plot(t, avgbcshuff, 'b', lw=2)
        ax.set_title('Stimulus: {}'.format(stimname))
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Betti {} Value'.format(betti))
    plt.savefig(plot_savepath+'B{}_betti{}_{}ms_{}_permuted_avg_withshuffled.png'.format(bird, betti, dt, prd))

def plot_all_bettis_with_shuffled(persistence_files, persistence_files_shuffled, maxbetti, maxt, figsize, plot_savepath):

    for betti in range(maxbetti):
        plot_average_betti_with_shuffled(persistence_files, persistence_files_shuffled, betti, maxt, figsize, plot_savepath)

def plot_all_bettis_together(persistence_files, maxbetti, maxt, figsize, plot_savepath):

    plot_savepath = os.path.abspath(plot_savepath)
    t = np.linspace(0, maxt, num=1000)
    bettiStimspline=[]
    
    nplots = len(persistence_files)
    print(nplots)
    nsubplotrows = np.round(nplots/4)
    print(nsubplotrows)
    subplot_shape = (nsubplotrows, 4)
    fig, axs = plt.subplots(nsubplotrows, 4, figsize=figsize)


    for pf_num, pf in enumerate(persistence_files):
        pf_metadata = extract_metadata(pf)
        stimname = pf_metadata['stimname']
        dt = pf_metadata['dt']
        prd = pf_metadata['period']
        pdata = pickle.load(open(pf, 'r'))
        upper=0
        bettiTrialspline=[]
        ax = axs.flatten()[pf_num]
        for betti in range(maxbetti):
            betticurves = np.empty_like(t)
            betticurves = compute_avg_betti_recursive(pdata, betticurves, betti, dt, t)
            avgbetticurve = np.mean(betticurves[1:], axis=0)
            ax.plot(t, avgbetticurve, lw=2)
        ax.set_title('Stimulus: {}'.format(stimname))
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Betti Value'.format(betti))
    plt.savefig(plot_savepath+'B{}_AllBetti_{}ms_{}_permuted_avg.png'.format(bird, betti, dt, prd))

def make_all_plots(block_path, analysis_id, maxbetti, maxt, figsize):
    block_path = os.path.abspath(block_path)
    real_topology_folder = os.path.join(block_path, 'topology/{}-real/'.format(analysis_id))
    shuffled_topology_folder = os.path.join(block_path, 'topology/{}-shuffled/'.format(analysis_id))
    print(real_topology_folder)
    # make figures dir
    figs_folder = os.path.join(block_path,  'figures/{}/'.format(analysis_id))
    if not os.path.exists(figs_folder):
        os.makedirs(figs_folder)

    real_topology_folders = sorted(glob.glob(real_topology_folder+'*'))
    shuffled_topology_folders = sorted(glob.glob(shuffled_topology_folder+'*'))

    for s_real, s_shuff in zip(real_topology_folders, shuffled_topology_folders):
        print('Real: {}     Shuffled: {}'.format(s_real, s_shuff))
        real_pfs = get_persistence_files(s_real)
        shuff_pfs = get_persistence_files(s_shuff)
        print('Plotting Bettis with Shuffled...')
        plot_all_bettis_with_shuffled(real_pfs, shuff_pfs, maxbetti, maxt, figsize, figs_folder)
        print('Plotting All Bettis Together...')
        plot_all_bettis_together(real_pfs, maxbetti, maxt, figsize, figs_folder)

