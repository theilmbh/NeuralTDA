import logging
import datetime
import numpy as np
import pandas as pd
import os
import sys
import subprocess
import pickle
import h5py
import time
import glob
from scipy.interpolate import interp1d
from scipy import integrate

from ephys import events, core

################################################
###### Old Topology Computation Functions ######
################################################

def calc_CI_bettis_on_dataset(block_path, analysis_id, cluster_group=None, windt_ms=50., n_subwin=5,
                              threshold=6, segment_info=DEFAULT_SEGMENT_INFO, persistence=False):
    '''
    Calculate bettis for each trial in a dataset and report statistics

    Parameters
    ------
    block_path : str
        Path to directory containing data files
    analysis_id : str
        A string to identify this particular analysis
    cluster_group : list
        list of cluster qualities to include in analysis
    windt_ms : float, optional
        window width in milliseconds
    n_subwin : int, optional
        number of sub-subwindows
    segment_info : dict
        dictionary containing information on which segment to compute topology
        'period' (stim or ?)
        'segstart' : time in ms of segment start relative to trial start
        'segend' : time in ms of segment end relative to trial start

    Yields
    ------
    betti_savefile : file
        File containing betti numbers for each trial for a given stimulus
        For all stimuli 
    '''
    # Create topology analysis folder
    global alogf
    analysis_path = os.path.join(block_path, 'topology/{}'.format(analysis_id))
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)

    analysis_logfile_name = '{}.log'.format(analysis_id)
    alogf = os.path.join(analysis_path, analysis_logfile_name)

    maxbetti = 10
    kwikfile = core.find_kwik(block_path)
    kwikname, ext = os.path.splitext(os.path.basename(kwikfile))



    spikes = core.load_spikes(block_path)
    clusters = core.load_clusters(block_path)
    trials = events.load_trials(block_path)
    fs = core.load_fs(block_path)

    windt_samps = np.floor(windt_ms*(fs/1000.))
    topology_log(alogf, 'Fs: {}'.format(str(fs)))
    topology_log(alogf, 'Window length: {} ms; {} samples'.format(str(windt_ms), str(windt_samps)))
    topology_log(alogf, 'Cluster group: {}'.format(str(cluster_group)))
    nclu = len(clusters[clusters['quality'].isin(cluster_group)])
    topology_log(alogf, 'N Clusters: {}'.format(str(nclu)))
    topology_log(alogf, 'Threshold: {}'.format(str(threshold)))

    stims = set(trials['stimulus'].values)
    for stim in stims:
        print('Calculating bettis for stim: {}'.format(stim))
        topology_log(alogf, 'Calculating bettis for stim: {}'.format(stim))
        stim_trials = trials[trials['stimulus'] == stim]
        nreps = len(stim_trials.index)
        topology_log(alogf, 'Number of repetitions for stim {} : {}'.format(stim, str(nreps)))
        stim_bettis = np.zeros([nreps, maxbetti])

        betti_savefile = kwikname + '_stim{}'.format(stim) + '_betti.csv'
        betti_savefile = os.path.join(analysis_path, betti_savefile)
        topology_log(alogf, 'Betti savefile: {}'.format(betti_savefile))
        betti_persistence_savefile = kwikname + '_stim{}'.format(stim) + '_bettiPersistence.pkl'
        betti_persistence_savefile = os.path.join(analysis_path, betti_persistence_savefile)
        topology_log(alogf, 'Betti persistence savefile: {}'.format(betti_persistence_savefile))
        betti_persistence_dict = dict()

        for rep in range(nreps):
            topology_log(alogf, 'Repetition {}'.format(str(rep)))
            pfile = kwikname + '_stim{}'.format(stim) + \
                    '_rep{}'.format(int(rep)) + '_simplex.txt'
            pfile = os.path.join(analysis_path, pfile)
            topology_log(alogf, 'pfile: {}'.format(pfile))

            trial_start = stim_trials.iloc[rep]['time_samples']
            trial_end = stim_trials.iloc[rep]['stimulus_end']

            cg_params = DEFAULT_CG_PARAMS
            cg_params['subwin_len'] = windt_samps
            cg_params['cluster_group'] = cluster_group
            cg_params['n_subwin'] = n_subwin
            cg_params['threshold'] = threshold

            segment = get_segment([trial_start, trial_end], fs, segment_info)
            print('Trial bounds: {}  {}'.format(str(trial_start),
                                                str(trial_end)))
            print('Segment bounds: {}  {}'.format(str(segment[0]),
                                                  str(segment[1])))
            topology_log(alogf, 'Trial bounds: {}  {}'.format(str(trial_start),
                                                str(trial_end)))
            topology_log(alogf, 'Segment bounds: {}  {}'.format(str(segment[0]), 
                                                                str(segment[1])))
            
            bettis = calc_bettis(spikes, segment,
                                 clusters, pfile, cg_params, persistence)
            # The bettis at the last step of the filtration are our 'total bettis'
            trial_bettis = bettis[-1][1]
            stim_bettis[rep, :len(trial_bettis)] = trial_bettis
            # save time course of bettis
            betti_persistence_dict['{}'.format(str(rep))] = bettis

        stim_bettis_frame = pd.DataFrame(stim_bettis)
        stim_bettis_frame.to_csv(betti_savefile, index_label='rep')
        if persistence:
            with open(betti_persistence_savefile, 'w') as bpfile:
                pickle.dump(betti_persistence_dict, bpfile)
        topology_log(alogf, '****** Curto+Itskov Topology Computation Completed ******')

def calc_CI_bettis_on_loaded_dataset(spikes, clusters, trials, fs, kwikfile, kwikname,
                                     cluster_group=None, windt_ms=50., n_subwin=5,
                                     segment_info=DEFAULT_SEGMENT_INFO, persistence=False):

    maxbetti = 10
    windt_samps = np.floor(windt_ms*(fs/1000.))

    stims = set(trials['stimulus'].values)
    for stim in stims:
        print('Calculating bettis for stim: {}'.format(stim))
        stim_trials = trials[trials['stimulus'] == stim]
        nreps = len(stim_trials.index)
        stim_bettis = np.zeros([nreps, maxbetti])

        betti_savefile = kwikname + '_stim{}'.format(stim) + '_betti.csv'
        betti_savefile = os.path.join(block_path, betti_savefile)
        betti_persistence_savefile = kwikname + '_stim{}'.format(stim) + '_bettiPersistence.pkl'
        betti_persistence_savefile = os.path.join(block_path, betti_persistence_savefile)
        betti_persistence_dict = dict()
        for rep in range(nreps):
            pfile = kwikname + '_stim{}'.format(stim) + \
                    '_rep{}'.format(int(rep)) + '_simplex.txt'
            pfile = os.path.join(block_path, pfile)

            trial_start = stim_trials.iloc[rep]['time_samples']
            trial_end = stim_trials.iloc[rep]['stimulus_end']

            cg_params = DEFAULT_CG_PARAMS
            cg_params['subwin_len'] = windt_samps
            cg_params['cluster_group'] = cluster_group
            cg_params['n_subwin'] = n_subwin

            segment = get_segment([trial_start, trial_end], fs, segment_info)
            print('Trial bounds: {}  {}'.format(str(trial_start),
                                                str(trial_end)))
            print('Segment bounds: {}  {}'.format(str(segment[0]),
                                                  str(segment[1])))

            bettis = calc_bettis(spikes, segment,
                                 clusters, pfile, cg_params, persistence)
            # The bettis at the last step of the filtration are our 'total bettis'
            trial_bettis = bettis[-1][1]
            stim_bettis[rep, :len(trial_bettis)] = trial_bettis
            # save time course of bettis
            betti_persistence_dict['{}'.format(str(rep))] = bettis

        stim_bettis_frame = pd.DataFrame(stim_bettis)
        stim_bettis_frame.to_csv(betti_savefile, index_label='rep')
        if persistence:
            with open(betti_persistence_savefile, 'w') as bpfile:
                pickle.dump(betti_persistence_dict, bpfile)

def calc_CI_bettis_on_dataset_average_activity(block_path, cluster_group=None, 
                                               windt_ms=50., n_subwin=5,
                                               segment_info=DEFAULT_SEGMENT_INFO,
                                               persistence=False):
    '''
    Calculate bettis for each trial in a dataset and report statistics

    Parameters
    ------
    block_path : str 
        Path to directory containing data files 
    cluster_group : list
        list of cluster qualities to include in analysis 
    windt_ms : float, optional
        window width in milliseconds
    n_subwin : int, optional
        number of sub-subwindows
    segment_info : dict 
        dictionary containing information on which segment to compute topology
        'period' (stim or ?).  If stim, defaults to looking at just stimulus period
        'segstart' : time in ms of segment start relative to trial start 
        'segend' : time in ms of segment end relative to trial start
    
    Yields
    ------
    betti_savefile : file
        File containing betti numbers for each trial for a given stimulus
        For all stimuli 
    '''
    maxbetti = 10
    kwikfile = core.find_kwik(block_path)
    kwikname, ext = os.path.splitext(os.path.basename(kwikfile))

    spikes = core.load_spikes(block_path)
    clusters = core.load_clusters(block_path)
    trials = events.load_trials(block_path)
    fs = core.load_fs(block_path)

    windt_samps = np.floor(windt_ms*(fs/1000.))

    stims = set(trials['stimulus'].values)
    for stim in stims:
        print('Calculating bettis for stim: {}'.format(stim))
        stim_trials = trials[trials['stimulus']==stim]
        nreps       = len(stim_trials.index)
        stim_bettis = np.zeros([nreps, maxbetti])

        betti_savefile = kwikname + '_stim{}'.format(stim) + '_avg_betti.csv'
        betti_savefile = os.path.join(block_path, betti_savefile)
        betti_persistence_savefile = kwikname + '_stim{}'.format(stim) + '_avg_bettiPersistence.pkl'
        betti_persistence_savefile = os.path.join(block_path, betti_persistence_savefile)
        betti_persistence_dict = dict()

        pfile = kwikname + '_stim{}'.format(stim) + \
                '_AllReps' + '_simplex.txt'
        pfile = os.path.join(block_path, pfile)

        first_trial_start = stim_trials.iloc[0]['time_samples']
        first_trial_end   = stim_trials.iloc[0]['stimulus_end']
        segment = get_segment([first_trial_start, first_trial_end], fs, segment_info)
        cg_params                   = DEFAULT_CG_PARAMS
        cg_params['subwin_len']     = windt_samps
        cg_params['cluster_group']  = cluster_group
        cg_params['n_subwin']       = n_subwin

        print('Segment bounds: {}  {}'.format(str(segment[0]), 
                                              str(segment[1])))
        # realign spikes to all fall within one trial
        for rep in range(nreps)[1:]:
            trial_start = stim_trials.iloc[rep]['time_samples']
            trial_end   = stim_trials.iloc[rep]['stimulus_end']
            spikes['time_samples'] = spikes.apply(lambda row: spike_time_subtracter(row, trial_start, trial_end, first_trial_start), axis=1)

        bettis = calc_bettis(spikes, segment, 
                                 clusters, pfile, cg_params, persistence)
        # The bettis at the last step of the filtration are our 'total bettis'
        trial_bettis                         = bettis[-1][1]
        stim_bettis[rep, :len(trial_bettis)] = trial_bettis
        # save time course of bettis
        betti_persistence_dict['{}'.format(str(rep))] = bettis

        stim_bettis_frame = pd.DataFrame(stim_bettis)
        stim_bettis_frame.to_csv(betti_savefile, index_label='rep')
        if persistence:
            with open(betti_persistence_savefile, 'w') as bpfile:
                pickle.dump(betti_persistence_dict, bpfile)


def calc_CI_bettis_binned_data(analysis_id, binned_data_file, block_path, thresh):
    '''
    Given a binned data file, compute the betti numbers of the Curto-Itskov style complex 

    Parameters
    ------
    analysis_id : str 
        A string to identify this particular analysis run 
    binned_data_file : str 
        Path to the binned data file on which to compute topology 
    block_path : str 
        Path to the folder containing the data for the block 
    thresh : float 
        Threshold to use when identifying cell groups 
    '''
    global alogf 

    bdf_name, ext = os.path.splitext(os.path.basename(binned_data_file))
    analysis_path = os.path.join(block_path, 'topology/{}/{}'.format(analysis_id, bdf_name))
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)

    analysis_files_prefix = '{}-{}'.format(bdf_name, analysis_id)
    analysis_logfile_name = '{}-{}.log'.format(bdf_name, analysis_id)
    alogf = os.path.join(analysis_path, analysis_logfile_name)

    maxbetti      = 50
    kwikfile      = core.find_kwik(block_path)
    kwikname, ext = os.path.splitext(os.path.basename(kwikfile))

    topology_log(alogf, '****** Beginning Curto+Itskov Topological Analysis of: {} ******'.format(kwikfile))
    topology_log(alogf, '****** Using Previously Binned Dataset: {} ******'.format(bdf_name))
    topology_log(alogf, 'Theshold: {}'.format(thresh))
    with h5py.File(binned_data_file, 'r') as bdf:

        stims = bdf.keys()
        nstims = len(stims)

        for stim in stims:
            topology_log(alogf, 'Calculating bettis for stim: {}'.format(stim))
            stim_trials = bdf[stim]
            nreps       = len(stim_trials)
            topology_log(alogf, 'Number of repetitions for stim {} : {}'.format(stim, str(nreps)))
            stim_bettis = np.zeros([nreps, maxbetti])

            betti_savefile = analysis_files_prefix + '-stim-{}'.format(stim) + '-betti.csv'
            betti_savefile = os.path.join(analysis_path, betti_savefile)
            topology_log(alogf, 'Betti savefile: {}'.format(betti_savefile))
            betti_persistence_savefile = analysis_files_prefix + '-stim-{}'.format(stim) + '-bettiPersistence.pkl'
            betti_persistence_savefile = os.path.join(analysis_path, betti_persistence_savefile)
            topology_log(alogf, 'Betti persistence savefile: {}'.format(betti_persistence_savefile))
            betti_persistence_dict = dict()

            for rep, repkey in enumerate(stim_trials.keys()):
                pfile = analysis_files_prefix + '-stim-{}'.format(stim) + \
                    '-rep-{}'.format(repkey) + '-simplex.txt'
                pfile = os.path.join(analysis_path, pfile)
                bettis = calc_bettis_from_binned_data(stim_trials[repkey], pfile, thresh)

                # The bettis at the last step of the filtration are our 'total bettis'
                trial_bettis                         = bettis[-1][1]
                stim_bettis[int(rep), :len(trial_bettis)] = trial_bettis
                # save time course of bettis
                betti_persistence_dict['{}'.format(str(rep))] = bettis

            stim_bettis_frame = pd.DataFrame(stim_bettis)
            stim_bettis_frame.to_csv(betti_savefile, index_label='rep')
            with open(betti_persistence_savefile, 'w') as bpfile:
                pickle.dump(betti_persistence_dict, bpfile)
        topology_log(alogf, 'Completed All Stimuli')

def calc_CI_bettis_permuted_binned_data(analysis_id, binned_data_file, block_path, thresh):
    '''
    Given a binned data file, compute the betti numbers of the Curto-Itskov style complex 

    Parameters
    ------
    analysis_id : str 
        A string to identify this particular analysis run 
    binned_data_file : str 
        Path to the binned data file on which to compute topology 
    block_path : str 
        Path to the folder containing the data for the block 
    thresh : float 
        Threshold to use when identifying cell groups 
    '''
    global alogf 

    bdf_name, ext = os.path.splitext(os.path.basename(binned_data_file))
    analysis_path = os.path.join(block_path, 'topology/{}/{}'.format(analysis_id, bdf_name))
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)

    analysis_files_prefix = '{}-{}'.format(bdf_name, analysis_id)
    analysis_logfile_name = '{}-{}.log'.format(bdf_name, analysis_id)
    alogf = os.path.join(analysis_path, analysis_logfile_name)

    maxbetti      = 50
    kwikfile      = core.find_kwik(block_path)
    kwikname, ext = os.path.splitext(os.path.basename(kwikfile))

    topology_log(alogf, '****** Beginning Curto+Itskov Topological Analysis of: {} ******'.format(kwikfile))
    topology_log(alogf, '****** Using Previously Binned Dataset: {} ******'.format(bdf_name))
    topology_log(alogf, 'Theshold: {}'.format(thresh))
    with h5py.File(binned_data_file, 'r') as bdf:

        stims = bdf.keys()
        nstims = len(stims)

        for stim in stims:
            topology_log(alogf, 'Calculating bettis for stim: {}'.format(stim))
            stim_trials = bdf[stim]
            nreps       = len(stim_trials)
            topology_log(alogf, 'Number of repetitions for stim {} : {}'.format(stim, str(nreps)))
            stim_bettis = np.zeros([nreps, maxbetti])

            betti_savefile = analysis_files_prefix + '-stim-{}'.format(stim) + '-betti.csv'
            betti_savefile = os.path.join(analysis_path, betti_savefile)
            topology_log(alogf, 'Betti savefile: {}'.format(betti_savefile))
            betti_persistence_savefile = analysis_files_prefix + '-stim-{}'.format(stim) + '-bettiPersistence.pkl'
            betti_persistence_savefile = os.path.join(analysis_path, betti_persistence_savefile)
            topology_log(alogf, 'Betti persistence savefile: {}'.format(betti_persistence_savefile))
            betti_persistence_dict = dict()

            for rep, repkey in enumerate(stim_trials.keys()):
                stim_trial_rep = stim_trials[repkey]
                betti_persistence_perm_dict = dict()
                for perm, permkey in enumerate(stim_trial_rep.keys()):

                    pfile = analysis_files_prefix + '-stim-{}'.format(stim) + \
                        '-rep-{}-perm-{}'.format(repkey, permkey) + '-simplex.txt'
                    pfile = os.path.join(analysis_path, pfile)
                    bettis = calc_bettis_from_binned_data(stim_trial_rep[permkey], pfile, thresh)

                    # The bettis at the last step of the filtration are our 'total bettis'
                    # save time course of bettis
                    betti_persistence_perm_dict['{}'.format(str(perm))] = bettis
                betti_persistence_dict['{}'.format(str(rep))] = betti_persistence_perm_dict
            with open(betti_persistence_savefile, 'w') as bpfile:
                pickle.dump(betti_persistence_dict, bpfile)
        topology_log(alogf, 'Completed All Stimuli')


###################################
###### Old Bin Data Routines ######
###################################

def shuffle_control_binned_data(binned_data_file, permuted_data_file, nshuffs):
    '''
    Bins the data using build_population_embedding 
    Parameters are given in bin_def_file
    Each line of bin_def_file contains the parameters for each binning

    Parameters
    ------
    binned_data_file : str
        Path to an hdf5 file containing the previously binned population vectors
    permuted_data_file : str
        Path to store the resulting hdf5 file that contains the shuffled vectors 
    nshuffs : int 
        Number of shuffles to perform
    '''

    # Try to make a folder to store the binnings
    global alogf
    
    with h5py.File(binned_data_file, "r") as popvec_f:
        win_size = popvec_f.attrs['win_size'] 
        fs = popvec_f.attrs['fs'] 
        nclus = popvec_f.attrs['nclus']
        stims = popvec_f.keys()
        with h5py.File(permuted_data_file, "w") as perm_f:
            perm_f.attrs['win_size'] = win_size
            perm_f.attrs['permuted'] = '0'
            perm_f.attrs['shuffled'] = '1'
            perm_f.attrs['fs']  = fs 

            for stim in stims:
                perm_stimgrp = perm_f.create_group(stim)
                stimdata = popvec_f[stim]
                trials = stimdata.keys()
                for trial in trials:
                    trialdata = stimdata[trial]
                    clusters = trialdata['clusters']
                    popvec = trialdata['pop_vec']
                    windows = trialdata['windows']
                    nwins = len(windows)
                    for perm_num in range(nshuffs):
                        clusters_to_save = clusters
                        popvec_save = popvec
                        perm_permgrp = perm_stimgrp.create_group('trial'+str(trial)+'perm'+str(perm_num))
                        for clu_num in range(nclus):
                            permt = np.random.permutation(nwins)
                            np.random.shuffle(popvec_save[clu_num, :])
                        perm_permgrp.create_dataset('pop_vec', data=popvec_save)
                        perm_permgrp.create_dataset('clusters', data=clusters_to_save)
                        perm_permgrp.create_dataset('windows', data=windows)

def permute_binned_data(binned_data_file, permuted_data_file, n_cells_in_perm, n_perm):
    '''
    Given a binned data file, make a new binned data file containing the vectors 
    taken from random subsets of the whole population 

    Parameters
    ------
    binned_data_file : str 
        Path to the binned data file on which to act 
    permuted_data_file : str 
        Path to file to store the resulting permuations 
    n_cells_in_perm : int 
        Number of cells to include in each subset 
    n_perm : int 
        Number of subsets to extract 
    '''

    # Try to make a folder to store the binnings
    global alogf
    
    with h5py.File(binned_data_file, "r") as popvec_f:
        win_size = popvec_f.attrs['win_size'] 
        fs = popvec_f.attrs['fs'] 
        nclus = popvec_f.attrs['nclus']
        permt = np.random.permutation(nclus)
        permt = permt[0:n_cells_in_perm]
        stims = popvec_f.keys()
        with h5py.File(permuted_data_file, "w") as perm_f:
            perm_f.attrs['win_size'] = win_size
            perm_f.attrs['permuted'] = '1'
            perm_f.attrs['shuffled'] = '0'
            perm_f.attrs['fs']  = fs 
            perm_f.attrs['nclus'] = nclus

            for stim in stims:
                perm_stimgrp = perm_f.create_group(stim)
                stimdata = popvec_f[stim]
                trials = stimdata.keys()
                for trial in trials:
                    perm_trialgrp = perm_stimgrp.create_group(trial)
                    trialdata = stimdata[trial]
                    clusters = trialdata['clusters']
                    popvec = trialdata['pop_vec']
                    windows = trialdata['windows']
                    for perm_num in range(n_perm):
                            permt = np.random.permutation(nclus)
                            permt = permt[0:n_cells_in_perm].tolist()
                            clusters_to_save = np.zeros(clusters.shape)
                            popvec_save = np.zeros(popvec.shape)
                            popvec.read_direct(popvec_save)
                            clusters.read_direct(clusters_to_save)
                            clusters_to_save = clusters_to_save[permt]
                            popvec_save = popvec_save[permt]
                            perm_permgrp = perm_trialgrp.create_group(str(perm_num))
                            perm_permgrp.create_dataset('pop_vec', data=popvec_save)
                            perm_permgrp.create_dataset('clusters', data=clusters_to_save)
                            perm_permgrp.create_dataset('windows', data=windows)


def make_shuffled_controls(path_to_binned, nshuffs):
    '''
    Takes a folder containing .binned files and makes shuffled controls from them

    Parameters
    ------
    path_to_binned : str 
        Path to a folder containing all the .binned hdf5 files you'd like to make controls for 
    nshuffs : int
        Number of shuffles per control 
    '''

    # get list of binned data files
    path_to_binned = os.path.abspath(path_to_binned)
    binned_data_files = glob.glob(os.path.join(path_to_binned, '*.binned'))
    if not binned_data_files:
        print('Error: No binned data files!')
        sys.exit(-1)

    # Try to make shuffled_controls folder
    shuffled_controls_folder = os.path.join(path_to_binned, 'shuffled_controls')
    if not os.path.exists(shuffled_controls_folder):
        os.makedirs(shuffled_controls_folder)

    for binned_data_file in binned_data_files:

        bdf_fold, bdf_full_name = os.path.split(binned_data_file)
        bdf_name, bdf_ext = os.path.splitext(bdf_full_name)
        scf_name = bdf_name + '_shuffled-control.binned'
        shuffled_control_file = os.path.join(shuffled_controls_folder, scf_name)

        shuffle_control_binned_data(binned_data_file, shuffled_control_file, nshuffs)

def make_permuted_binned_data(path_to_binned, n_cells_in_perm, n_perms):
    '''
    Takes a folder containing .binned files and makes permuted subsets of them. 

    Parameters
    ------
    path_to_binned : str 
        Path to a folder containing all the .binned hdf5 files you'd like to make controls for 
    nperms : int
        Number of permutations
    '''

    path_to_binned = os.path.abspath(path_to_binned)
    binned_data_files = glob.glob(os.path.join(path_to_binned, '*.binned'))
    if not binned_data_files:
        print('Error: No binned data files!')
        sys.exit(-1) 
    
    permuted_binned_folder = os.path.join(path_to_binned, 'permuted_binned')
    if not os.path.exists(permuted_binned_folder):
        os.makedirs(permuted_binned_folder)

    for binned_data_file in binned_data_files:

        bdf_fold, bdf_full_name = os.path.split(binned_data_file)
        bdf_name, bdf_ext = os.path.splitext(bdf_full_name)
        pbd_name = bdf_name + '-permuted.binned'
        permuted_data_file = os.path.join(permuted_binned_folder, pbd_name)

        permute_binned_data(binned_data_file, permuted_data_file, n_cells_in_perm, n_perms)
