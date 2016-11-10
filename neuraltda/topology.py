import os
import sys
import subprocess
import time
import glob
import pickle
import logging
import datetime
import csv

import numpy as np
import pandas as pd
import h5py
from scipy.interpolate import interp1d
from scipy import integrate

from ephys import events, core

################################
###### Module Definitions ######
################################

TOPOLOGY_LOG = logging.getLogger('NeuralTDA')

DEFAULT_CG_PARAMS = {'cluster_group': None, 'subwin_len': 100,
                     'threshold': 6.0, 'n_subwin': 5}

DEFAULT_SEGMENT_INFO = {'period': 1}

#################################
###### Auxiliary Functions ######
#################################

def setup_logging(func_name):
    '''
    Sets up a logger object for logging events during topology computations.

    Parameters
    ----------
    func_name : str
        Name of the script using the NeuralTDA library
    '''

    # Make logging dir if doesn't exist
    logging_dir = os.path.join(os.getcwd(), 'logs/')
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    logging_fname = '{}_'.format(func_name) + \
                    datetime.datetime.utcnow().strftime('%Y-%m-%dT%H%M%SZ') + \
                    '.log'
    logging_file = os.path.join(logging_dir, logging_fname)

    # create logger
    logger = logging.getLogger('NeuralTDA')
    logger.setLevel(logging.DEBUG)

    # create file and stream handlers and set level to debug
    ch = logging.FileHandler(logging_file)
    ch.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    formatter.converter = time.gmtime

    # add formatter to ch and sh
    ch.setFormatter(formatter)
    sh.setFormatter(formatter)

    # add ch and sh to logger
    logger.addHandler(ch)
    logger.addHandler(sh)

    # Log initialization
    logger.info('Starting {}.'.format(func_name))

def get_spikes_in_window(spikes, window):
    '''
    Returns a spike DataFrame containing all spikes within a time window.
    The time window is specified by sample number.

    Parameters
    ----------
    spikes : pandas DataFrame
        Contains the spike data (see ephys.core)
    window : tuple
        The lower and upper bounds, in samples,
        of the time window for extracting spikes

    Returns
    -------
    spikes_in_window : pandas DataFrame
        DataFrame with same layout as input spikes
        but containing only spikes within window
    '''
    mask = ((spikes['time_samples'] <= window[1]) &
            (spikes['time_samples'] >= window[0]))
    return spikes[mask]

def calc_mean_fr(cluster_row, spikes, window):
    '''
    Computes the mean firing rate of a unit within the given spikes DataFrame

    Parameters
    ----------
    row : pandas dataframe row
        Row of clusters dataframe containing clusterID to compute
        (see ephys.clusters)
    spikes : pandas dataframe
        Contains the spike data (see ephys.core).
        Firing rate computed from this data over the window
    window : tuple
        Time, in samples, over which the spikes in 'spikes' occur.

    Returns
    -------
    mean_fr : float
        Mean firing rate over the interval.
    '''
    cluster = cluster_row['cluster']
    spikes = spikes[spikes['cluster'] == cluster]
    spikes = get_spikes_in_window(spikes, window)
    nspikes = len(spikes.index)
    dt = window[1] - window[0]
    mean_fr = (1.0*nspikes) / dt
    retframe = pd.DataFrame({'mean_fr': [mean_fr]})
    return retframe.iloc[0]

def calc_mean_fr_int(cluster, spikes, window):
    ''' Does the same as above, but for ints.  This is bad.  Fix this
    '''
    spikes = spikes[spikes['cluster'] == cluster]
    spikes = get_spikes_in_window(spikes, window)
    nspikes = len(spikes.index)
    dt = window[1] - window[0]
    mean_fr = (1.0*nspikes) / dt
    return mean_fr


def create_subwindows(segment, subwin_len, noverlap=0):
    '''
    Create list of subwindows for cell group identification.

    Parameters
    ----------
    segment : list
        Sample numbers of the beginning
        and end of the segment to subdivide into windows.
    subwin_len : int
        Number of samples to include in a subwindow.
    noverlap : int
        Number of samples to overlap the windows

    Returns
    ------
    subwindows : list
        List of subwindows.
        Each subwindow is a list containing
        the starting sample and ending sample.
    '''
    dur = segment[1]-segment[0]
    skip = subwin_len - noverlap
    maxK = int(np.floor(float(dur)/float(skip)))
    starts = [segment[0] + k*skip for k in range(maxK)]
    windows = [[w, min(w + (subwin_len-1), segment[1])] for w in starts]
    return windows

def calc_population_vectors(spikes, clusters, windows, thresh):
    '''
    Builds population vectors according to Curto and Itskov 2008.

    Parameters
    ------
    spikes : pandas DataFrame
        DataFrame containing spike data.
        Must have 'fr_mean' column containing mean firing rate.
    clusters : pandas DataFrame
        Dataframe containing cluster information.
    windows : tuple
        The set of windows for which to compute population vectors.
    thresh : float
        Multiple of mean firing rate for a cluster to be included in vector.

    Returns
    ------
    popvec_list : list
        Population vector list.
        Each element is a list containing the window and the population vector.
        The population vector is an array containing cluster ID and firing rate.
    '''
    TOPOLOGY_LOG.info('Building population vectors...')
    total_win = len(windows)
    popvec_list = []
    for win_num, win in enumerate(windows):
        if np.mod(win_num, 500) == 0:
            TOPOLOGY_LOG.info("Window {} of {}".format(str(win_num),
                                                       str(total_win)))
        popvec = np.zeros([len(clusters.index), 3])
        for ind, cluster in enumerate(clusters['cluster'].values):
            fr = calc_mean_fr_int(cluster, spikes, win)
            cluster_fr = clusters[clusters['cluster'] == cluster]
            fr_mask = (1.0*thresh*cluster_fr['mean_fr']).values
            fr_vec = fr > fr_mask
            popvec[ind, 1] = fr
            popvec[ind, 0] = cluster
            popvec[ind, 2] = fr_vec
        popvec_list.append([win, popvec])
    return popvec_list

def calc_cell_groups(spikes, segment, clusters, cg_params=DEFAULT_CG_PARAMS):
    '''
    Creates cell group DataFrame according to Curto and Itskov 2008

    Parameters
    ----------
    spikes : pandas dataframe
        Dataframe containing the spikes to analyze.
    segment : tuple or list of floats
        Time window for which to create cell groups.
    clusters : pandas DataFrame
        DataFrame containing cluster information
    cg_params : dict, optional
        Parameters for cell group creation.  Includes:
        cluster_group : str
            Quality of the clusters to include in the analysis (Good, MUA, etc)
        subwin_len : int
            length (samples) for each subwin
        threshold : float, optional
            Multiples above baseline firing rate to include activity
        n_subwin : int
            Number of subwindows to use to generate population vectors

    Returns
    -------
    cell_groups : list
        list where each entry is a list containing a time window
        and the ID's of the cells in that group
    '''
    cluster_group = cg_params['cluster_group']
    subwin_len = cg_params['subwin_len']
    threshold = cg_params['threshold']
    n_subwin = cg_params['n_subwin']

    spikes = get_spikes_in_window(spikes, segment)
    if cluster_group != None:
        mask = np.ones(len(clusters.index)) < 0
        for grp in cluster_group:
            mask = np.logical_or(mask, clusters['quality'] == grp)
        clusters = clusters[mask]
        spikes = spikes[spikes['cluster'].isin(clusters['cluster'].values)]

    topology_subwindows = create_subwindows(segment, subwin_len, n_subwin)
    mean_fr_row_func = lambda row: calc_mean_fr(row, spikes, segment)['mean_fr']
    clusters['mean_fr'] = clusters.apply(mean_fr_row_func, axis=1)

    # Build population vectors
    population_vector_list = calc_population_vectors(spikes, clusters,
                                                     topology_subwindows,
                                                     threshold)
    cell_groups = []
    for population_vector_win in population_vector_list:
        win = population_vector_win[0]
        popvec = population_vector_win[1]
        active_cells = popvec[(popvec[:, 2].astype(bool)), 0].astype(int)
        cell_groups.append([win, active_cells])

    return cell_groups

def build_perseus_input(cell_groups, savefile):
    '''
    Formats cell group information as an input file
    for the Perseus persistent homology software

    Parameters
    ------
    cell_groups : list
        cell_group information returned by calc_cell_groups
    savefile : str
        File in which to put the formatted cellgroup information

    Yields
    ------
    savefile : text File
        file suitable for running perseus on
    '''
    with open(savefile, 'w+') as pfile:
        pfile.write('1\n')
        for win_grp in cell_groups:
            grp = list(win_grp[1])
            #debug_print('Cell group: ' + str(grp) +'\n')
            grp_dim = len(grp) - 1
            if grp_dim < 0:
                continue
            vert_str = str(grp)
            vert_str = vert_str.replace('[', '')
            vert_str = vert_str.replace(']', '')
            vert_str = vert_str.replace(' ', '')
            vert_str = vert_str.replace(',', ' ')
            out_str = str(grp_dim) + ' ' + vert_str + ' 1\n'
            pfile.write(out_str)
    return savefile

def build_perseus_persistent_input(cell_groups, savefile):
    '''
    Formats cell group information as an input file
    for the Perseus persistent homology software, but assigns filtration
    levels for each cell group based on the time order of their appearance
    in the signal.

    Parameters
    ----------
    cell_groups : list
        cell_group information returned by calc_cell_groups
    savefile : str
        File in which to put the formatted cellgroup information

    Yields
    ------
    savefile : text File
        file suitable for running perseus on
    '''
    with open(savefile, 'w+') as pfile:
        pfile.write('1\n')
        for ind, win_grp in enumerate(cell_groups):
            grp = list(win_grp[1])
            grp_dim = len(grp) - 1
            if grp_dim < 0:
                continue
            vert_str = str(grp)
            vert_str = vert_str.replace('[', '')
            vert_str = vert_str.replace(']', '')
            vert_str = vert_str.replace(' ', '')
            vert_str = vert_str.replace(',', ' ')
            out_str = str(grp_dim) + ' ' + vert_str + ' {}\n'.format(str(ind+1))
            pfile.write(out_str)
    return savefile

def run_perseus(pfile):
    '''
    Runs Perseus persistent homology software on the data in pfile

    Parameters
    ------
    pfile : str
        File on which to compute homology

    Returns
    ------
    betti_file : str
        File containing resultant betti numbers

    '''
    TOPOLOGY_LOG.info('In run_perseus')
    pfile_split = os.path.splitext(pfile)
    of_string = pfile_split[0]
    perseus_command = "perseus"
    perseus_return_code = subprocess.call([perseus_command, 'nmfsimtop', pfile,
                                           of_string])
    TOPOLOGY_LOG.info('Perseus return code: {}'.format(perseus_return_code))
    betti_file = of_string+'_betti.txt'
    #betti_file = os.path.join(os.path.split(pfile)[0], betti_file)
    TOPOLOGY_LOG.info('Betti file from run_perseus: %s' % betti_file)
    return betti_file

def calc_bettis(spikes, segment, clusters, pfile, cg_params=DEFAULT_CG_PARAMS,
                persistence=False):
    '''
    Calculate betti numbers for spike data in segment.

    Parameters
    ------
    spikes : pandas DataFrame
        Dataframe containing spike data
    segment : list
        Time window, in samples, of data to calculate betti numbers
    clusters : pandas Dataframe
        Dataframe containing cluster data
    pfile : str
        File in which to save simplex data
    cg_params : dict
        Parameters for cell group generation
    persistence : bool, optional
        If true, will compute the time dependence of the bettis
        by including filtration times in the cell groups

    Returns
    ------
    bettis : list
        Betti numbers.  Each member is a list with the first member being
        filtration time and the second being betti numbers
    '''
    cell_groups = calc_cell_groups(spikes, segment, clusters, cg_params)
    if persistence:
        build_perseus_persistent_input(cell_groups, pfile)
    else:
        build_perseus_input(cell_groups, pfile)
    betti_file = run_perseus(pfile)
    bettis = []
    with open(betti_file, 'r') as bf:
        for bf_line in bf:
            if len(bf_line) < 2:
                continue
            betti_data = bf_line.split()
            filtration_time = int(betti_data[0])
            betti_numbers = map(int, betti_data[1:])
            bettis.append([filtration_time, betti_numbers])
    return bettis

def get_segment(trial_bounds, fs, segment_info):
    '''
    Convert a segment info specifier into a segment.

    Parameters
    ------
    trial_bounds : list
        List containing trial start and trial end in samples
    fs : int
        The sampling rate
    segment_info : dict
        Dictionary containing:
        'period' (1 for stim, 0 for other)
        'segstart' : time in ms of segment start relative to trial start
        'segend' : time in ms of segment end relative to trial start

    Returns
    ------
    segment : list
        bounds for the segment for which to compute topology, in samples.

    '''
    if segment_info['period'] == 1:
        return trial_bounds
    else:
        seg_start = trial_bounds[0] \
                    + np.floor(segment_info['segstart']*(fs/1000.))
        seg_end = trial_bounds[0] \
                  + np.floor(segment_info['segend']*(fs/1000.))
    return [seg_start, seg_end]

#############################################
###### Binned Data Auxiliary Functions ######
#############################################

def calc_cell_groups_from_binned_data(binned_dataset, thresh):
    '''
    Given a binned dataset and a firing rate threshold,
    generate the time sequence of cell groups.

    Parameters
    ----------
    binned_dataset : h5py group
        This is the group at the lowest level of the bin hierarchy.
        It must contain three datasets: 'pop_vec', 'clusters', and 'windows'.
        - 'pop_vec' is an nclu x nwin matrix with each element containing
          the firing rate in Hz of that unit for that window.
        - 'clusters' is an nclu x 1 matrix containing the cluster ids of each
          row of 'pop_vec'
        - 'windows' is an nwin x 2 matrix containing the start and end samples
          of each window used for the binning.
    thresh : float
        Multiple of average firing rate in order for a cluster to be included
        in a cell group.

    Returns
    -------
    cell_groups : list
        A list containing entries for each cell group.
        The first element of each entry is the window number of the cell group.
        The second element of each entry is a list of cluster ids for the cells
        in the cell group.
    '''
    bds = np.array(binned_dataset['pop_vec'])
    clusters = np.array(binned_dataset['clusters']).astype(int)
    nwin = bds.shape[1]
    mean_frs = np.mean(bds, 1)
    cell_groups = []
    for win in range(nwin):
        acty = bds[:, win]
        above_thresh = np.greater(acty, thresh*mean_frs)
        clus_in_group = clusters[above_thresh]
        cell_groups.append([win, clus_in_group])
    return cell_groups

def calc_bettis_from_binned_data(binned_dataset, pfile, thresh):
    '''
    Calculates betti numbers from a previously binned dataset.

    Parameters
    ----------
    binned_dataset : h5py group
        This is the group at the lowest level of the bin hierarchy.
        It must contain three datasets: 'pop_vec', 'clusters', and 'windows'.
        - 'pop_vec' is an nclu x nwin matrix with each element containing
          the firing rate in Hz of that unit for that window.
        - 'clusters' is an nclu x 1 matrix containing the cluster ids of each
          row of 'pop_vec'
        - 'windows' is an nwin x 2 matrix containing the start and end samples
          of each window used for the binning.
    pfile : str
        The name of the file to store the Perseus input.
    thresh : float
        Multiple of average firing rate in order for a cluster to be included
        in a cell group.

    Returns
    -------
    bettis : list
        List containing betti numbers.
        The first entry in each element is the filtration time point.
        The second entry in each element is a list containing the values of the
        betti numbers for each dimension.
    '''
    cell_groups = calc_cell_groups_from_binned_data(binned_dataset, thresh)
    build_perseus_persistent_input(cell_groups, pfile)
    betti_file = run_perseus(pfile)
    bettis = []
    with open(betti_file, 'r') as bf:
        for bf_line in bf:
            if len(bf_line) < 2:
                continue
            betti_data = bf_line.split()
            filtration_time = int(betti_data[0])
            betti_numbers = map(int, betti_data[1:])
            bettis.append([filtration_time, betti_numbers])
    return bettis

#######################################
###### Current Binning Functions ######
#######################################

def build_population_embedding(spikes, trials, clusters, win_size, fs,
                               cluster_group, segment_info, popvec_fname, dtOverlap=0.0):
    '''
    Embeds binned population activity into R^n

    Parameters
    ------
    spikes : pandas dataframe
        Spike frame from ephys.core
    trials : pandas dataframe
        Trials dataframe from ephys.trials
    clusters : Pandas DataFrame
        Clusters frame from ephys.core
    win_size : float
        Window size in milliseconds
    fs : float
        Sampling rate in Hz
    cluster_group : list
        List containing cluster sort quality strings to include in embedding
        Possible entries include 'Good', 'MUA'
    segment_info : dict
        Dictionary containing parameters for segment generation
    popvec_fname : str
        File in which to store the embedding
    '''
    with h5py.File(popvec_fname, "w") as popvec_f:

        popvec_f.attrs['win_size'] = win_size
        popvec_f.attrs['fs'] = fs
        if cluster_group != None:
            mask = np.ones(len(clusters.index)) < 0
            for grp in cluster_group:
                mask = np.logical_or(mask, clusters['quality'] == grp)
        clusters_to_use = clusters[mask]
        clusters_list = clusters_to_use['cluster'].unique()
        spikes = spikes[spikes['cluster'].isin(clusters_list)]
        nclus = len(clusters_to_use.index)
        popvec_f.attrs['nclus'] = nclus
        stims = trials['stimulus'].unique()

        for stim in stims:
            stimgrp = popvec_f.create_group(stim)
            stim_trials = trials[trials['stimulus'] == stim]
            nreps = len(stim_trials.index)

            for rep in range(nreps):
                trialgrp = stimgrp.create_group(str(rep))
                trial_start = stim_trials.iloc[rep]['time_samples']
                trial_end = stim_trials.iloc[rep]['stimulus_end']
                trial_bounds = [trial_start, trial_end]
                seg_start, seg_end = get_segment(trial_bounds,
                                                 fs, segment_info)
                this_segment = [seg_start, seg_end]
                win_size_samples = int(np.round(win_size/1000. * fs))
                overlap_samples = int(np.round(dtOverlap/1000. * fs))
                windows = create_subwindows(this_segment, win_size_samples, overlap_samples)
                nwins = len(windows)
                popvec_dset_init = np.zeros((nclus, nwins))
                popvec_dset = trialgrp.create_dataset('pop_vec',
                                                      data=popvec_dset_init)
                trialgrp.create_dataset('clusters', data=clusters_list)
                trialgrp.create_dataset('windows', data=np.array(windows))
                popvec_dset.attrs['fs'] = fs
                popvec_dset.attrs['win_size'] = win_size
                for win_ind, win in enumerate(windows):
                    spikes_in_win = get_spikes_in_window(spikes, win)
                    clus_that_spiked = spikes_in_win['cluster'].unique()
                    if len(clus_that_spiked) > 0:
                        for clu in clus_that_spiked:
                            clu_msk = (spikes_in_win['cluster'] == clu)
                            pvclu_msk = (clusters_list == clu)
                            nsp_clu = float(len(spikes_in_win[clu_msk]))
                            win_s = win_size/1000.
                            popvec_dset[pvclu_msk, win_ind] = nsp_clu/win_s

def prep_and_bin_data(block_path, bin_def_file, bin_id, nshuffs):
    '''
    Loads dataset using ephys and bins it
    according to the parameters in a bin definition file

    Parameters
    ----------
    block_path : str
        Path to the directory containing the sorted spike data
    bin_def_file : str
        Path to the file containing bin definitions strings
        The format is:
        <bin_id> <winsize (ms)> <clu_group> <seg_period> <seg_start> <seg_end>
    bin_id : str
        Unique identifier for this binning session
        Typically a date/time string is used
    '''

    spikes = core.load_spikes(block_path)
    clusters = core.load_clusters(block_path)
    trials = events.load_trials(block_path)
    fs = core.load_fs(block_path)
    kwikfile = core.find_kwik(block_path)
    binning_folder = do_bin_data(block_path, spikes,
                                 clusters, trials,
                                 fs, kwikfile,
                                 bin_def_file, bin_id)
    if nshuffs:
        TOPOLOGY_LOG.info('Making shuffled controls.')
        make_shuffled_controls_recursive(binning_folder, nshuffs)

def do_bin_data(block_path, spikes, clusters, trials,
                fs, kwikfile, bin_def_file, bin_id):
    '''
    Bins the data using build_population_embedding
    Parameters are given in bin_def_file
    For each line of bin_def_file, bin the data
    according to the parameters in the line.

    Parameters
    ------
    block_path : str
        Path to the folder containing all the original datafiles
    spikes : Pandas DataFrame
        Dataframe containing spike data
    clusters : Pandas DataFrame
        DataFrame containing cluster information
    trials : PD DataFrame
        Containing trial information
    fs : int
        sampling rate
    kwikfile: str
        path to kwikfile
    bin_def_file : str
        Path to file containing parameters for each binning
    bin_id : str
        Identifier for this particular binning run
    '''
    binning_folder = os.path.join(block_path, 'binned_data/{}'.format(bin_id))
    if not os.path.exists(binning_folder):
        os.makedirs(binning_folder)
    kwikname = os.path.splitext(os.path.basename(kwikfile))[0]

    with open(bin_def_file, 'r') as bdf:
        for bdf_line in bdf:
            binning_params = bdf_line.split(' ')
            binning_id = binning_params[0]
            win_size = float(binning_params[1])
            cluster_groups = binning_params[2]
            segment = int(binning_params[3])
            # bin_str = ''.join(cluster_groups) + '-' + str(win_size) + '-' 
            TOPOLOGY_LOG.info('segment specifier: {}'.format(segment))
            seg_start = float(binning_params[4])
            seg_end = float(binning_params[5])
            segment_info = {'period': segment,
                            'segstart':seg_start,
                            'segend': seg_end}
            cluster_group = cluster_groups.split(',')
            binning_fname = '{}-{}.binned'.format(kwikname, binning_id)
            binning_path = os.path.join(binning_folder, binning_fname)
            if os.path.exists(binning_path):
                TOPOLOGY_LOG.warn('Binning file {} \
                                    already exists.'.format(binning_path))
                continue
            TOPOLOGY_LOG.info('Binning data into {}'.format(binning_fname))
            build_population_embedding(spikes, trials,
                                       clusters, win_size,
                                       fs, cluster_group,
                                       segment_info, binning_path)
            TOPOLOGY_LOG.info('Done')
    return binning_folder

def do_bin_data_direct(block_path, spikes, clusters, trials,
                fs, kwikfile, win_size, cgroups, segment, sstart, send, bin_id):
    '''
    Bins the data using build_population_embedding
    Parameters are given directly, without a definition file.

    Parameters
    ------
    block_path : str
        Path to the folder containing all the original datafiles
    spikes : Pandas DataFrame
        Dataframe containing spike data
    clusters : Pandas DataFrame
        DataFrame containing cluster information
    trials : PD DataFrame
        Containing trial information
    fs : int
        sampling rate
    kwikfile: str
        path to kwikfile
    bin_id : str
        Identifier for this particular binning run
    '''
    binning_folder = os.path.join(block_path, 'binned_data/{}'.format(bin_id))
    if not os.path.exists(binning_folder):
        os.makedirs(binning_folder)
    kwikname = os.path.splitext(os.path.basename(kwikfile))[0]

    binning_params = bdf_line.split(' ')
    binning_id = binning_params[0]
    win_size = float(binning_params[1])
    cluster_groups = binning_params[2]
    segment = int(binning_params[3])
    TOPOLOGY_LOG.info('segment specifier: {}'.format(segment))
    seg_start = float(binning_params[4])
    seg_end = float(binning_params[5])
    segment_info = {'period': segment,
                    'segstart':seg_start,
                    'segend': seg_end}
    cluster_group = cluster_groups.split(',')
    binning_fname = '{}-{}.binned'.format(kwikname, binning_id)
    binning_path = os.path.join(binning_folder, binning_fname)
    if os.path.exists(binning_path):
        TOPOLOGY_LOG.warn('Binning file {} \
                            already exists.'.format(binning_path))
    TOPOLOGY_LOG.info('Binning data into {}'.format(binning_fname))
    build_population_embedding(spikes, trials,
                               clusters, win_size,
                               fs, cluster_group,
                               segment_info, binning_path)
    TOPOLOGY_LOG.info('Done')
    return binning_folder

def permute_recursive(data_group, perm_group, n_cells_in_perm, nperms):

    if 'pop_vec' in data_group.keys():
        clusters = data_group['clusters']
        popvec = data_group['pop_vec']
        windows = data_group['windows']
        nclus = len(clusters)
        for perm_num in range(nperms):
            permt = np.random.permutation(nclus)
            if len(clusters) >= n_cells_in_perm:
                permt = permt[0:n_cells_in_perm].tolist()
            else:
                permt = permt[0:].tolist()
            clusters_to_save = np.zeros(clusters.shape)
            popvec_save = np.zeros(popvec.shape)
            
            if 0 in popvec.shape:
                print(popvec.name)
            popvec.read_direct(popvec_save)
            clusters.read_direct(clusters_to_save)
            clusters_to_save = clusters_to_save[permt]
            popvec_save = popvec_save[permt]
            perm_permgrp = perm_group.create_group(str(perm_num))
            perm_permgrp.create_dataset('pop_vec', data=popvec_save)
            perm_permgrp.create_dataset('clusters', data=clusters_to_save)
            perm_permgrp.create_dataset('windows', data=windows)
    else:
        for inst_num, inst in enumerate(data_group.keys()):
            new_perm_group = perm_group.create_group(inst)
            permute_recursive(data_group[inst], new_perm_group,
                              n_cells_in_perm, nperms)

def trialshuffle_recursive(data_group, perm_group, ntrials):

    if 'pop_vec' in data_group.keys():
        clusters = data_group['clusters']
        popvec = data_group['pop_vec']
        windows = data_group['windows']
        nclus = len(clusters)
        for perm_num in range(nperms):
            permt = np.random.permutation(nclus)
            if len(clusters) >= n_cells_in_perm:
                permt = permt[0:n_cells_in_perm].tolist()
            else:
                permt = permt[0:].tolist()
            clusters_to_save = np.zeros(clusters.shape)
            popvec_save = np.zeros(popvec.shape)
            popvec.read_direct(popvec_save)
            clusters.read_direct(clusters_to_save)
            clusters_to_save = clusters_to_save[permt]
            popvec_save = popvec_save[permt]
            perm_permgrp = perm_group.create_group(str(perm_num))
            perm_permgrp.create_dataset('pop_vec', data=popvec_save)
            perm_permgrp.create_dataset('clusters', data=clusters_to_save)
            perm_permgrp.create_dataset('windows', data=windows)
    else:
        for inst_num, inst in enumerate(data_group.keys()):
            new_perm_group = perm_group.create_group(inst)
            permute_recursive(data_group[inst], new_perm_group,
                              n_cells_in_perm, nperms)

def shuffle_recursive(data_group, perm_group, nshuffs):

    if 'pop_vec' in data_group.keys():
        clusters = data_group['clusters']
        popvec = data_group['pop_vec']
        windows = data_group['windows']
        nwins = len(windows)
        nclus = len(clusters)
        for perm_num in range(nshuffs):
            clusters_to_save = clusters
            popvec_save = np.array(popvec)
            perm_permgrp = perm_group.create_group(str(perm_num))
            for clu_num in range(nclus):
                permt = np.random.permutation(nwins)
                np.random.shuffle(popvec_save[clu_num, :])
            perm_permgrp.create_dataset('pop_vec', data=popvec_save)
            perm_permgrp.create_dataset('clusters', data=clusters_to_save)
            perm_permgrp.create_dataset('windows', data=windows)
    else:
        for inst_num, inst in enumerate(data_group.keys()):
            new_perm_group = perm_group.create_group(inst)
            shuffle_recursive(data_group[inst], new_perm_group, nshuffs)

def shuffle_binned_data_recursive(binned_data_file,
                                  permuted_data_file,
                                  nshuffs):

    with h5py.File(binned_data_file, "r") as popvec_f:
        win_size = popvec_f.attrs['win_size']
        fs = popvec_f.attrs['fs']
        nclus = popvec_f.attrs['nclus']
        stims = popvec_f.keys()
        with h5py.File(permuted_data_file, "w") as perm_f:
            perm_f.attrs['win_size'] = win_size
            perm_f.attrs['permuted'] = '0'
            perm_f.attrs['shuffled'] = '1'
            perm_f.attrs['fs'] = fs
            perm_f.attrs['nclus'] = nclus
            perm_f.attrs['nshuffs'] = nshuffs
            for stim in stims:
                perm_stimgrp = perm_f.create_group(stim)
                stimdata = popvec_f[stim]
                shuffle_recursive(stimdata, perm_stimgrp, nshuffs)

def permute_binned_data_recursive(binned_data_file, permuted_data_file,
                                  n_cells_in_perm, nperms):

    with h5py.File(binned_data_file, "r") as popvec_f:
        win_size = popvec_f.attrs['win_size']
        fs = popvec_f.attrs['fs']
        nclus = popvec_f.attrs['nclus']
        stims = popvec_f.keys()
        with h5py.File(permuted_data_file, "w") as perm_f:
            perm_f.attrs['win_size'] = win_size
            perm_f.attrs['permuted'] = '1'
            perm_f.attrs['shuffled'] = '0'
            perm_f.attrs['fs'] = fs
            perm_f.attrs['nclus'] = nclus
            perm_f.attrs['n_cells_in_perm'] = n_cells_in_perm
            for stim in stims:
                perm_stimgrp = perm_f.create_group(stim)
                stimdata = popvec_f[stim]
                permute_recursive(stimdata, perm_stimgrp,
                                  n_cells_in_perm, nperms)

def shuffle_trials(binned_data_file, trialshuffle_file):
    ''' Has to be done on original binned file'''
    with h5py.File(binned_data_file, "r") as popvec_f:
        win_size = popvec_f.attrs['win_size']
        fs = popvec_f.attrs['fs']
        nclus = popvec_f.attrs['nclus']
        stims = popvec_f.keys()
        with h5py.File(trialshuffle_file, "w") as ts_f:
            ts_f.attrs['win_size'] = win_size
            ts_f.attrs['trialshuffled'] = '1'
            ts_f.attrs['fs'] = fs
            ts_f.attrs['nclus'] = nclus
            for stim in stims:
                ts_stimgrp = ts_f.create_group(stim)
                stimdata = popvec_f[stim]
                oldTrials = stimdata.keys()
                for trial in oldTrials:
                    wins = np.array(stimdata[trial]['windows'])
                    clus = np.array(stimdata[trial]['clusters'])
                    ts_newTrialGroup = ts_stimgrp.create_group(str(trial))
                    newPopVec = np.zeros(stimdata[stimdata.keys()[0]]['pop_vec'].shape)
                    randTrialIDVec = np.random.random_integers(0, high=len(oldTrials)-1, size=nclus)
                    for unitIndx in range(nclus):
                        randTrialID = randTrialIDVec[unitIndx]
                        newPopVec[unitIndx, :] = np.array(stimdata[str(randTrialID)]['pop_vec'])[unitIndx, :]
                    ts_newTrialGroup.create_dataset('pop_vec', data=newPopVec)
                    ts_newTrialGroup.create_dataset('clusters', data=clus)
                    ts_newTrialGroup.create_dataset('windows', data=wins)


def cij_recursive(data_group, tmax, fs):

    if 'pop_vec' in data_group.keys():
        nclus = len(data_group['clusters'])
        Cij_mat = compute_Cij_matrix(data_group['pop_vec'],
                                     data_group['windows'],
                                     fs, nclus, tmax)
        data_group.create_dataset('Cij', data=Cij_mat)
        TOPOLOGY_LOG.info('Cij matrix computed.')
    else:
        for inst_num, inst in enumerate(data_group.keys()):
            cij_recursive(data_group[inst], tmax, fs)

def compute_cij_recusive(binned_data_file, tmax):

    with h5py.File(binned_data_file, "r+") as popvec_f:
        fs = popvec_f.attrs['fs']
        stims = popvec_f.keys()
        for stim in stims:
            TOPOLOGY_LOG.info('Computing Cij matrix \
                                for stim: {}'.format(stim))
            stimdata = popvec_f[stim]
            cij_recursive(stimdata, tmax, fs)

def make_cij(path_to_binned, tmax):

    path_to_binned = os.path.abspath(path_to_binned)
    binned_data_files = glob.glob(os.path.join(path_to_binned, '*.binned'))
    if not binned_data_files:
        TOPOLOGY_LOG.error('NO BINNED DATA FILES')
        sys.exit(-1)

    for binned_data_file in binned_data_files:
        TOPOLOGY_LOG.info('Computing Cij for: {}'.format(binned_data_file))
        compute_cij_recusive(binned_data_file, tmax)
        TOPOLOGY_LOG.info('Cij computation complete.')


def make_shuffled_controls_recursive(path_to_binned, nshuffs):
    '''
    Takes a folder containing .binned files and makes shuffled controls.

    Parameters
    ------
    path_to_binned : str
        Path to a folder containing all the .binned hdf5 files
        for which controls are desired.
    nshuffs : int
        Number of shuffles per control
    '''

    path_to_binned = os.path.abspath(path_to_binned)
    binned_data_files = glob.glob(os.path.join(path_to_binned, '*.binned'))
    if not binned_data_files:
        TOPOLOGY_LOG.error('NO BINNED DATA FILES')
        sys.exit(-1)
    shuffled_controls_folder = os.path.join(path_to_binned, 'shuffled_controls')
    if not os.path.exists(shuffled_controls_folder):
        os.makedirs(shuffled_controls_folder)

    for binned_data_file in binned_data_files:
        bdf_fold, bdf_full_name = os.path.split(binned_data_file)
        bdf_name, bdf_ext = os.path.splitext(bdf_full_name)
        scf_name = bdf_name + '-shuffled_control.binned'
        shuffled_control_file = os.path.join(shuffled_controls_folder, scf_name)
        shuffle_binned_data_recursive(binned_data_file,
                                      shuffled_control_file,
                                      nshuffs)

def make_permuted_binned_data_recursive(path_to_binned,
                                        n_cells_in_perm,
                                        n_perms):
    '''
    Takes a folder containing .binned files and makes permuted subsets of them.

    Parameters
    ------
    path_to_binned : str
        Path to a folder containing all the .binned hdf5 files
        for which permutations are desired.
    nperms : int
        Number of permutations
    '''

    path_to_binned = os.path.abspath(path_to_binned)
    binned_data_files = glob.glob(os.path.join(path_to_binned, '*.binned'))
    if not binned_data_files:
        TOPOLOGY_LOG.error('NO BINNED DATA FILES')
        sys.exit(-1)
    permuted_binned_folder = os.path.join(path_to_binned, 'permuted_binned')
    if not os.path.exists(permuted_binned_folder):
        os.makedirs(permuted_binned_folder)

    for binned_data_file in binned_data_files:
        bdf_fold, bdf_full_name = os.path.split(binned_data_file)
        bdf_name, bdf_ext = os.path.splitext(bdf_full_name)
        pbd_name = bdf_name + '-permuted.binned'
        permuted_data_file = os.path.join(permuted_binned_folder, pbd_name)
        permute_binned_data_recursive(binned_data_file, permuted_data_file,
                                      n_cells_in_perm, n_perms)

def make_trialshuffled(path_to_binned):
    '''
    Takes a folder containing .binned files and makes permuted subsets of them.

    Parameters
    ------
    path_to_binned : str
        Path to a folder containing all the .binned hdf5 files
        for which permutations are desired.
    nperms : int
        Number of permutations
    '''

    path_to_binned = os.path.abspath(path_to_binned)
    binned_data_files = glob.glob(os.path.join(path_to_binned, '*.binned'))
    if not binned_data_files:
        TOPOLOGY_LOG.error('NO BINNED DATA FILES')
        sys.exit(-1)
    trialshuffled_folder = os.path.join(path_to_binned, 'trialshuffle/')
    if not os.path.exists(trialshuffled_folder):
        os.makedirs(trialshuffled_folder)

    for binned_data_file in binned_data_files:
        bdf_fold, bdf_full_name = os.path.split(binned_data_file)
        bdf_name, bdf_ext = os.path.splitext(bdf_full_name)
        pbd_name = bdf_name + '-trialshuffled.binned'
        trialshuffled_file = os.path.join(trialshuffled_folder, pbd_name)
        shuffle_trials(binned_data_file, trialshuffled_file)

def compute_avg_acty_binned(binned_file, avgfile):
    '''
    Takes an already-binned data file and computes average activity over trials
    '''

    with h5py.File(binned_file, 'r') as bdf:
        win_size = bdf.attrs['win_size']
        fs = bdf.attrs['fs']
        nclus = bdf.attrs['nclus']
        stims = bdf.keys()
        with h5py.File(avgfile, 'w') as avgf:
            # Copy metadata in binned file
            avgf.attrs['win_size'] = win_size
            avgf.attrs['avg'] = 1
            avgf.attrs['fs'] = fs
            avgf.attrs['nclus'] = nclus
            for stim in stims:
                avg_stimgrp = avgf.create_group(stim)
                stimdata = bdf[stim]
                wins = np.array(stimdata['0']['windows'])
                clus = np.array(stimdata['0']['clusters'])
                avgact = calc_avg(stimdata)
                avg_stimgrp.create_dataset('pop_vec', data=avgact)
                avg_stimgrp.create_dataset('clusters', data=clus)
                avg_stimgrp.create_dataset('windows', data=wins)
            
def calc_avg(stimdata):

    ntrial = len(stimdata.keys())
    popvec_run = np.zeros(stimdata['0']['pop_vec'].shape)
    for trial in stimdata.keys():
        popvec_run = popvec_run+np.array(stimdata[trial]['pop_vec'])

    avg = popvec_run / ntrial
    return avg

def bin_avg(binned_folder):
    '''
    Takes a folder containing binned files and computes 
    average ctivity for each binned file and places them
    in a new folder 
    '''

    path_to_binned = os.path.abspath(binned_folder)
    binned_data_files = glob.glob(os.path.join(path_to_binned, '*.binned'))

    if not binned_data_files:
        TOPOLOGY_LOG.error('NO BINNED DATA FILES')
        sys.exit(-1)
    avg_folder = os.path.join(path_to_binned, 'avgacty')
    if not os.path.exists(avg_folder):
        os.makedirs(avg_folder)

    for bdf in binned_data_files:
        bdf_fold, bdf_full_name = os.path.split(bdf)
        bdf_name, bdf_ext = os.path.splitext(bdf_full_name)
        avgactyf = os.path.join(avg_folder, bdf_name +'-avgacty.binned')
        compute_avg_acty_binned(bdf, avgactyf)

##########################################################
###### Cell Group Topological Computation Functions ######
##########################################################

def compute_barcode(pfile_stem, barcode_dict):
    '''
    Given a pfile, return a dictionary containing the persistence intervals
    for all generators of homology of all dimensions
    '''
    betti=0
    barcode_file = pfile_stem + '-simplex_{}.txt'.format(betti)
    while os.path.exists(barcode_file):
        
        with open(barcode_file, 'r') as bcf:
            barcode = []
            bcreader = csv.reader(bcf, delimiter=' ')
            for row in bcreader:
                introw = [int(s) for s in row]
                barcode.append(introw)
            barcode_dict[str(betti)] = barcode
        betti = betti+1
        barcode_file = pfile_stem + '-simplex_{}.txt'.format(betti)
    return barcode_dict

def compute_recursive(data_group, pfile_stem, h_stem, betti_persistence_perm_dict,
                      analysis_path, thresh):
    if 'pop_vec' in data_group.keys():
        TOPOLOGY_LOG.info('compute_recursive: calculating CI bettis')
        pfile = pfile_stem + '-simplex.txt'
        pfile = os.path.join(analysis_path, pfile)
        TOPOLOGY_LOG.info('pfile: %s' % pfile)
        betti_persistence_perm_dict['hstr'] = h_stem
        bettis = calc_bettis_from_binned_data(data_group, pfile, thresh)
        nbetti = len(bettis)
        barcode_dict = dict()
        barcode_dict = compute_barcode(os.path.join(analysis_path,pfile_stem), barcode_dict)
        betti_persistence_perm_dict['bettis'] = bettis
        betti_persistence_perm_dict['barcodes'] = barcode_dict
        return betti_persistence_perm_dict
    else:
        for perm, permkey in enumerate(data_group.keys()):
            new_data_group = data_group[permkey]
            new_pfile_stem = pfile_stem + '-{}'.format(permkey)
            new_h_stem = h_stem + '-{}'.format(permkey)
            new_bpp_dict = dict()
            bettis = compute_recursive(new_data_group,
                                                  new_pfile_stem, new_h_stem,
                                                  new_bpp_dict, analysis_path,
                                                  thresh)
            betti_persistence_perm_dict['{}'.format(permkey)] = bettis
        return betti_persistence_perm_dict

def calc_CI_bettis_hierarchical_binned_data(analysis_id, binned_data_file,
                                            block_path, thresh):
    '''
    Given a binned data file, compute the betti numbers of the Curto-Itskov
    Takes in a binned data file with arbitrary depth of permutations.
    Finds the bottom level permutation

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
    TOPOLOGY_LOG.info('Starting calc_CI_bettis_hierarchical_binned_data')
    TOPOLOGY_LOG.info('analysis_id: {}'.format(analysis_id))
    bdf_name, ext = os.path.splitext(os.path.basename(binned_data_file))
    analysis_path = os.path.join(block_path,
                                 'topology/{}/'.format(analysis_id))
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)

    analysis_files_prefix = analysis_id
    analysis_logfile_name = '{}-{}.log'.format(bdf_name, analysis_id)
    TOPOLOGY_LOG.info('bdf_name: {}'.format(bdf_name))
    TOPOLOGY_LOG.info('analysis_path: {}'.format(analysis_path))

    maxbetti = 50
    kwikfile = core.find_kwik(block_path)
    kwikname, ext = os.path.splitext(os.path.basename(kwikfile))

    TOPOLOGY_LOG.info('Beginning Curto+Itskov \
                        Topological Analysis of: {}'.format(kwikfile))
    TOPOLOGY_LOG.info('Theshold: {}'.format(thresh))

    with h5py.File(binned_data_file, 'r') as bdf:

        stims = bdf.keys()
        nstims = len(stims)
        bpd_withstim = dict()
        for stim in stims:
            TOPOLOGY_LOG.info('Calculating bettis for stim: {}'.format(stim))
            h_stem = stim + '-'
            stim_trials = bdf[stim]
            nreps = len(stim_trials)
            TOPOLOGY_LOG.info('Number of repetitions \
                                for stim {}: {}'.format(stim, str(nreps)))
            stim_bettis = np.zeros([nreps, maxbetti])

            betti_savefile = analysis_files_prefix \
                             + '-stim-{}'.format(stim) \
                             + '-betti.csv'
            betti_savefile = os.path.join(analysis_path, betti_savefile)
            TOPOLOGY_LOG.info('Betti savefile: {}'.format(betti_savefile))
            bps = analysis_files_prefix \
                  + '-stim-{}'.format(stim) \
                  + '-bettiPersistence.pkl'
            betti_persistence_savefile = os.path.join(analysis_path, bps)
            TOPOLOGY_LOG.info('Betti persistence \
                               savefile: {}'.format(betti_persistence_savefile))
            bpd = dict()
            pfile_stem = analysis_files_prefix \
                         + '-stim-{}'.format(stim) \
                         + '-rep-'
            bpd = compute_recursive(stim_trials,
                                    pfile_stem,
                                    h_stem,
                                    bpd,
                                    analysis_path,
                                    thresh)
            bpd_withstim[stim] = bpd
            with open(betti_persistence_savefile, 'w') as bpfile:
                pickle.dump(bpd, bpfile)
        bpdws_sfn = os.path.join(analysis_path, analysis_files_prefix+'-bettiResultsDict.pkl')
        with open(bpdws_sfn, 'w') as bpdwsfile:
                pickle.dump(bpd_withstim, bpdwsfile)
        TOPOLOGY_LOG.info('Completed All Stimuli')

def bptd_recursive(bpd, bpdf):

    if 'hstr' in bpd.keys():
        bettis = bpd['bettis']
        nfilt = len(bettis)
        betti_dict = dict()
        for filt, betti_nums in bettis:
            for dim, betti_num in enumerate(betti_nums):
                betti_dict[str(dim)] = betti_num
            betti_dict['filtration'] = filt
            betti_dict['hierarchy'] = bpd['hstr']
            filtdataframe = pd.DataFrame(data=betti_dict, index=[1])
            bpdf = bpdf.append(filtdataframe, ignore_index=True)
        return bpdf
    else:
        for indx, h_level in enumerate(bpd.keys()):
            bpdf = bptd_recursive(bpd[h_level], bpdf)
        return bpdf

def betti_pickle_to_dataframe(bpd):

    bpdf = pd.DataFrame(columns=['hierarchy', 'filtration', '0', '1', '2'])
    bpdf = bptd_recursive(bpd, bpdf)
    return bpdf

###################################################
###### Clique Topology Computation Functions ######
###################################################

def calc_fr_funcs(binned_dataset, windows, fs, i, j):

    #make the time vector
    tstart = windows[0, 0]/fs
    tend = windows[-1, -1]/fs
    t = (windows[:, 1] - windows[:, 0])/(2*fs)
    t = t - tstart # realign
    T = tstart-tend

    # Get the firing rate vectors for cells i and j
    fr_i_vec = binned_dataset[i, :]
    fr_j_vec = binned_dataset[j, :]

    # get Mean Firing Rate
    r_i = np.mean(fr_i_vec)
    r_j = np.mean(fr_j_vec)

    # interpolate
    fr_i = interp1d(t, fr_i_vec,
                    kind='nearest',
                    bounds_error=False,
                    fill_value="extrapolate")
    fr_j = interp1d(t, fr_j_vec,
                    kind='nearest',
                    bounds_error=False,
                    fill_value="extrapolate")

    return (fr_i, fr_j, T, r_i, r_j)

def calc_corr_interp(binned_dataset, windows, fs, i, j, tmax):

    #make the time vector
    tstart = windows[0, 0]/fs
    tend = windows[-1, -1]/fs
    t = (windows[:, 1] - windows[:, 0])/(2*fs)
    t = t - tstart # realign
    T = tstart-tend

    # Get the firing rate vectors for cells i and j
    fr_i_vec = binned_dataset[i, :]
    fr_j_vec = binned_dataset[j, :]

    # get Mean Firing Rate
    r_i = np.mean(fr_i_vec)
    r_j = np.mean(fr_j_vec)

    # interpolate
    fr_i = interp1d(t, fr_i_vec,
                    kind='nearest',
                    bounds_error=False,
                    fill_value="extrapolate")
    fr_j = interp1d(t, fr_j_vec,
                    kind='nearest',
                    bounds_error=False,
                    fill_value="extrapolate")

    nt = np.linspace(0, T, 1000)
    dt = nt[1]-nt[0]
    fr_i_sig = fr_i(nt)
    fr_j_sig = fr_j(nt)
    corr_ij = np.correlate(fr_i_sig, fr_j_sig)
    corr_ji = np.correlate(fr_j_sig, fr_i_sig)
    int_lim = np.round(tmax/dt)
    cij = np.sum(corr_ij[0:int_lim]*dt)
    cji = np.sum(corr_ji[0:int_lim]*dt)
    return max(cij, cji)/(tmax*r_i*r_j)

def calc_corr_raw(binned_dataset, windows, fs, i, j, tmax):

    dt = (windows[0, 1] - windows[0, 0]) / fs
    fr_i_vec = binned_dataset[i, :]
    fr_j_vec = binned_dataset[j, :]

    # get Mean Firing Rate
    r_i = np.mean(fr_i_vec)
    r_j = np.mean(fr_j_vec)

    N = len(fr_i_vec)
    corr_ij = np.correlate(fr_i_vec, fr_j_vec, mode='full')/float(N)
    corr_ji = np.correlate(fr_j_vec, fr_i_vec, mode='full')/float(N)
    int_lim = np.round(tmax/dt)
    if int_lim >= len(corr_ij):
        int_lim = len(corr_ij)
    cij = np.sum(corr_ij[0:int_lim])
    cji = np.sum(corr_ji[0:int_lim])
    C = max(cij, cji)/(int_lim*r_i*r_j)
    return C

def ccg(fr_i, fr_j, T, tau):
    '''
    Returns cross correlogram of Giusti et al. 2015
    '''
    ccg_ij, err = integrate.quad(lambda x: fr_i(x)*fr_j(x+tau), 0, T)/T
    return ccg_ij

def Rccg():

    Rccg_ij = lambda tau, t: fr_i(t)*fr_j(t+tau)/T
    Rccg_ji = lambda tau, t: fr_j(t)*fr_i(t+tau)/T
    return (Rccg_ij, Rccg_ji)

def compute_Cij(binned_dataset, windows, fs, i, j, tmax):

    (fr_i, fr_j, T, r_i, r_j) = calc_fr_funcs(binned_dataset, windows, fs, i, j)

    ccg_ij = lambda tau: ccg(fr_i, fr_j, T, tau)
    ccg_ji = lambda tau: ccg(fr_j, fr_i, T, tau)

    ccg_ij_val, err = integrate.quad(ccg_ij, 0, tmax)
    ccg_ji_val, err = integrate.quad(ccg_ji, 0, tmax)
    Cij_unnorm = max(ccg_ij_val, ccg_ji_val)

    Cij = Cij_unnorm/(tmax*r_i*r_j)
    return Cij

def compute_Cij_matrix(binned_dataset, windows, fs, nclus, tmax):

    Cij = np.zeros((nclus, nclus))
    bds = np.array(binned_dataset)
    for i in range(nclus):
        for j in range(i+1, nclus):
            Cij_val = calc_corr_raw(bds, windows, fs, i, j, tmax)
            Cij[i, j] = Cij_val
            Cij[j, i] = Cij_val
    return Cij

def build_perseus_input_corrmat(cij, nsteps, savefile):
    '''
    Formats Correlation Matrix information as input
    for perseus persistent homology software

    Parameters
    ------
    cij : ndarray
        correlation matrix
    savefile : str
        File in which to put the formatted cellgroup information

    Yields
    ------
    savefile : text File
        file suitable for running perseus on
    '''
    TOPOLOGY_LOG.info('Building perseus corrmat input')
    No, Mo = cij.shape
    if abs(No-Mo):
        TOPOLOGY_LOG.error('Not a square matrix! Aborting')
        return
    TOPOLOGY_LOG.info('Removing NaNs.')
    ctr = np.isnan(cij[0, :])
    cij = cij[~ctr]
    cij = cij[:, ~ctr]

    N, M = cij.shape
    TOPOLOGY_LOG.info('Shape before NaN removal: {}, {}   \
                        After: {}'.format(No, Mo, N))
    #if (1-np.diag(cij)).any():
    #   TOPOLOGY_LOG.warn('Diagonal entries of Cij not equal to 1. Correcting')
    #  cij = cij + np.diag(1-np.diag(cij))
    step_size = np.amax(cij)/float(nsteps)
    TOPOLOGY_LOG.info('Using persistence step_size: {}'.format(step_size))

    with open(savefile, 'w+') as pfile:
        #write num coords per vertex
        pfile.write('{}\n'.format(N))
        pfile.write('{} {} {} 5\n'.format(0, step_size, nsteps))
        for rowid in range(N):
            cij_row = cij[rowid, :]
            row_str = str(cij_row)
            row_str = row_str.replace('[', '')
            row_str = row_str.replace(']', '')
            row_str = row_str.replace(' ', '')
            row_str = row_str.replace(',', ' ')
            out_str = row_str+'\n'
            pfile.write(out_str)
    return savefile

def run_perseus_corrmat(pfile):
    '''
    Runs perseus persistent homology software on the data in pfile

    Parameters
    ------
    pfile : str
        file on which to compute homology

    Returns
    ------
    betti_file : str
        file containing resultant betti numbers

    '''
    TOPOLOGY_LOG.info('Running perseus-corrmat')
    of_string, ext = os.path.splitext(pfile)
    perseus_command = "/home/btheilma/bin/perseus"
    perseus_return_code = subprocess.call([perseus_command, 'corrmat', pfile,
                                           of_string])
    TOPOLOGY_LOG.info('pfile: {}'.format(pfile))
    betti_file = of_string+'_betti.txt'
    betti_file = os.path.join(os.path.split(pfile)[0], betti_file)
    return betti_file

def calc_clique_topology_bettis(cij, nsteps, pfile):

    TOPOLOGY_LOG.info('Calculating clique topology bettis')
    build_perseus_input_corrmat(cij, nsteps, pfile)
    betti_file = run_perseus_corrmat(pfile)
    bettis = []
    with open(betti_file, 'r') as bf:
        for bf_line in bf:
            if len(bf_line) < 2:
                continue
            betti_data = bf_line.split()
            nbetti = len(betti_data)-1
            filtration_time = int(betti_data[0])
            betti_numbers = map(int, betti_data[1:])
            bettis.append([filtration_time, betti_numbers])
    return bettis

def compute_cliquetop_recursive(data_group, pfile_stem,
                                betti_persistence_perm_dict,
                                analysis_path, nsteps):
    if 'Cij' in data_group.keys():
        pfile = pfile_stem + '-simplex.txt'
        pfile = os.path.join(analysis_path, pfile)
        bettis = calc_clique_topology_bettis(np.array(data_group['Cij']),
                                             nsteps, pfile)
        return bettis
    elif ('Cij' not in data_group.keys()) and ('pop_vec' in data_group.keys()):
        TOPOLOGY_LOG.error('Cij matrix not present.')
        return dict()
    else:
        for perm, permkey in enumerate(data_group.keys()):
            new_data_group = data_group[permkey]
            new_pfile_stem = pfile_stem + '-{}'.format(permkey)
            new_bpp_dict = dict()
            bettis = compute_cliquetop_recursive(new_data_group,
                                                 new_pfile_stem,
                                                 new_bpp_dict,
                                                 analysis_path,
                                                 nsteps)
            betti_persistence_perm_dict['{}'.format(permkey)] = bettis
        return betti_persistence_perm_dict

def calc_cliquetop_bettis_recursive(analysis_id, binned_data_file,
                                    block_path, nsteps):
    '''
    Given a binned data file, compute the betti numbers of the Clique complex.
    Takes in a binned data file with arbitrary depth of permutations.
    Finds the bottom level permutation

    Parameters
    ------
    analysis_id : str
        A string to identify this particular analysis run
    binned_data_file : str
        Path to the binned data file on which to compute topology
        MUST CONTAIN CIJ matrix
    block_path : str
        Path to the folder containing the data for the block
    thresh : float
        Threshold to use when identifying cell groups
    '''
    TOPOLOGY_LOG.info('Starting calc_CliqueTop_bettis_recursive')
    TOPOLOGY_LOG.info('analysis_id: {}'.format(analysis_id))
    bdf_name, ext = os.path.splitext(os.path.basename(binned_data_file))
    analysis_path_ex = 'topology/clique_top/{}/{}'.format(analysis_id, bdf_name)
    analysis_path = os.path.join(block_path, analysis_path_ex)
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)
    analysis_files_prefix = '{}-{}-CliqueTop-'.format(bdf_name, analysis_id)
    analysis_logfile_name = '{}-{}.log'.format(bdf_name, analysis_id)
    TOPOLOGY_LOG.info('bdf_name: {}'.format(bdf_name))
    TOPOLOGY_LOG.info('analysis_path: {}'.format(analysis_path))
    maxbetti = 50
    kwikfile = core.find_kwik(block_path)
    kwikname, ext = os.path.splitext(os.path.basename(kwikfile))

    TOPOLOGY_LOG.info('Beginning Clique Topology \
                        Analysis of: {}'.format(kwikfile))
    with h5py.File(binned_data_file, 'r') as bdf:

        stims = bdf.keys()
        nstims = len(stims)
        for stim in stims:
            TOPOLOGY_LOG.info('Calculating CliqueTop Bettis \
                                for stim: {}'.format(stim))
            stim_trials = bdf[stim]
            nreps = len(stim_trials)
            TOPOLOGY_LOG.info('Number of repetitions \
                               for stim {}: {}'.format(stim, str(nreps)))
            stim_bettis = np.zeros([nreps, maxbetti])

            betti_savefile = analysis_files_prefix \
                             + '-stim-{}'.format(stim) \
                             + '-betti.csv'
            betti_savefile = os.path.join(analysis_path, betti_savefile)
            TOPOLOGY_LOG.info('Betti savefile: {}'.format(betti_savefile))
            bps = analysis_files_prefix \
                  + '-stim-{}'.format(stim) \
                  + '-bettiPersistence.pkl'
            betti_persistence_savefile = os.path.join(analysis_path, bps)
            TOPOLOGY_LOG.info('Betti persistence savefile: \
                               {}'.format(betti_persistence_savefile))
            betti_persistence_dict = dict()
            pfile_stem = analysis_files_prefix \
                         + '-stim-{}'.format(stim) \
                         + '-rep-'
            bpd = compute_cliquetop_recursive(stim_trials,
                                              pfile_stem,
                                              betti_persistence_dict,
                                              analysis_path, nsteps)
            with open(betti_persistence_savefile, 'w') as bpfile:
                pickle.dump(bpd, bpfile)
        TOPOLOGY_LOG.info('Completed All Stimuli')

def compute_all_ci_topology(binned_folder, permuted_folder, shuffled_folder, analysis_id, block_path, thresh):

    if binned_folder:  
        binned_data_files = glob.glob(os.path.join(binned_folder, '*.binned'))
        for bdf in binned_data_files:
            TOPOLOGY_LOG.info('Computing topology for: %s' % bdf)
            calc_CI_bettis_hierarchical_binned_data(analysis_id, bdf,
                                                    block_path, thresh)
    if permuted_folder:
        permuted_data_files = glob.glob(os.path.join(permuted_folder, '*.binned'))
        for pdf in permuted_data_files:
            TOPOLOGY_LOG.info('Computing topology for: %s' % bdf)
            calc_CI_bettis_hierarchical_binned_data(analysis_id+'_real', pdf,
                                                    block_path, thresh)
    if shuffled_folder:
        spdfs = os.path.join(shuffled_folder, '*.binned')
        shuffled_permuted_data_files = glob.glob(spdfs)
        for spdf in shuffled_permuted_data_files:
            TOPOLOGY_LOG.info('Computing topology for: %s' % bdf)
            calc_CI_bettis_hierarchical_binned_data(analysis_id+'_shuffled',
                                                    spdf, block_path, thresh)

def dbLoadData(block_path):

    # Load Raw Data
    spikes = core.load_spikes(block_path)
    trials = events.load_trials(block_path)
    fs = core.load_fs(block_path)
    clusters = core.load_clusters(block_path)
    return (spikes, trials, clusters, fs)

def dag_bin(block_path, winsize, segment_info, ncellsperm, nperms, nshuffs, **kwargs):

    (spikes, trials, clusters, fs) = dbLoadData(block_path)
    bfdict = do_dag_bin(block_path, spikes, trials, clusters, fs, winsize, segment_info, ncellsperm, nperms, nshuffs, **kwargs)
    return bfdict
    
def do_dag_bin(block_path, spikes, trials, clusters, fs, winsize, segment_info, ncellsperm, nperms, nshuffs, cluster_group=['Good'], dtOverlap=0.0):

    block_path = os.path.abspath(block_path)
    # Create directories and filenames
    analysis_id = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    raw_binned_fname = analysis_id + '-{}.binned'.format(winsize)
    analysis_id_forward = analysis_id + '-{}'.format(winsize) 
    
    binned_folder = os.path.join(block_path, 'binned_data/{}/'.format(analysis_id))
    if not os.path.exists(binned_folder):
        os.makedirs(binned_folder)

    trialshuffle_folder = os.path.join(binned_folder, 'trialshuffle/')
    average_binned_folder = os.path.join(binned_folder, 'avgacty/')

    permuted_binned_folder = os.path.join(binned_folder, 'permuted_binned/')
    permuted_average_folder = os.path.join(average_binned_folder, 'permuted_binned/')
    permuted_shuffled_folder = os.path.join(permuted_binned_folder, 'shuffled_controls/')
    average_permuted_shuffled_folder = os.path.join(permuted_average_folder, 'shuffled_controls/')

    permuted_trialshuffle_folder = os.path.join(trialshuffle_folder, 'permuted_binned/')

    raw_binned_f = os.path.join(binned_folder, raw_binned_fname)

    #Cluster group
    #cluster_group = ['Good']

    # Bin the raw data
    TOPOLOGY_LOG.info('Binning data')
    build_population_embedding(spikes, trials, clusters, winsize, fs,
                               cluster_group, segment_info, raw_binned_f, dtOverlap)

    # Average binned raw data
    try:
        TOPOLOGY_LOG.info('Averaging activity')
        bin_avg(binned_folder)
    except:
        TOPOLOGY_LOG.error('Averaging Failed!')
        average_binned_folder = ''

    # Permute raw data
    try:
        TOPOLOGY_LOG.info('Permuting data') 
        make_permuted_binned_data_recursive(binned_folder, ncellsperm, nperms)
    except:
        TOPOLOGY_LOG.error('Permuting Failed!')
        permuted_binned_folder=''

    # Permute Averaged data
    try:
        TOPOLOGY_LOG.info('Permuting Averaged data') 
        make_permuted_binned_data_recursive(average_binned_folder, ncellsperm, nperms)
    except:
        TOPOLOGY_LOG.error('FAILED!')
        permuted_average_folder = ''
    
    # Shuffle Permuted data
    try:    
        TOPOLOGY_LOG.info('Shuffling permuted data') 
        make_shuffled_controls_recursive(permuted_binned_folder, nshuffs)
    except:
        TOPOLOGY_LOG.error('Shuffling FAILED!')
        permuted_shuffled_folder = ''
    
    # Shuffle Permuted Average Data 
    try:
        TOPOLOGY_LOG.info('Shuffling permuted average data')
        make_shuffled_controls_recursive(permuted_average_folder, nshuffs)
    except:
        TOPOLOGY_LOG.error('Shuffling Permuted AverageFAILED!')
        average_permuted_shuffled_folder = ''
    
    #Trial Shuffle Original Binned
    try:
        TOPOLOGY_LOG.info('Making Trial Shuffled Binned')
        make_trialshuffled(binned_folder)
    except:
        TOPOLOGY_LOG.error(' Trial Shuffled FAILED!')
        permuted_trialshuffle_folder = ''
    
    # Permute TrialShuffled
    try:    
        TOPOLOGY_LOG.info('Making Permuted Trial Shuffled')
        make_permuted_binned_data_recursive(trialshuffle_folder, ncellsperm, nperms)
    except:
        TOPOLOGY_LOG.error('Permuted Trial Shuffled FAILED!')
        permuted_trialshuffle_folder = ''
    
    bfdict={'permuted': permuted_binned_folder, 'avgpermuted': permuted_average_folder,
           'permutedshuff': permuted_shuffled_folder, 'avgpermshuff': average_permuted_shuffled_folder,
           'raw': binned_folder, 'analysis_id': analysis_id_forward, 'trialshuffled': trialshuffle_folder, 'trialshuffperm': permuted_trialshuffle_folder}
    return bfdict


def dag_topology(block_path, thresh, bfdict, simplexWinSize=0):
    
    permuted_binned_folder = bfdict['permuted']
    permuted_average_folder = bfdict['avgpermuted']
    permuted_shuffled_folder = bfdict['permutedshuff']
    average_permuted_shuffled_folder = bfdict['avgpermshuff']
    permTrialShuffFold = bfdict['trialshuffperm']
    analysis_id = bfdict['analysis_id']

    bpt = os.path.join(block_path, 'topology/')
    analysis_dict = dict()

    # Run topologies

    if permuted_binned_folder:
        tpid_permute = analysis_id + '-{}-permuted'.format(thresh)
        tpf_permute = os.path.join(bpt, tpid_permute)
        permuted_data_files = glob.glob(os.path.join(permuted_binned_folder, '*.binned'))
        for pdf in permuted_data_files:
            TOPOLOGY_LOG.info('Computing topology for: %s' % pdf)
            calc_CI_bettis_hierarchical_binned_data(tpid_permute, pdf,
                                                    block_path, thresh)

        p_results = glob.glob(os.path.join(tpf_permute, '*-bettiResultsDict.pkl'))[0]
        with open(p_results, 'r') as f:
            res = pickle.load(f)
            analysis_dict['permuted'] = res

    if permuted_shuffled_folder:
        tpid_permuteshuff = analysis_id + '-{}-permuted-shuffled'.format(thresh)
        tpf_permuteshuff = os.path.join(bpt, tpid_permuteshuff)
        spdfs = os.path.join(permuted_shuffled_folder, '*.binned')
        shuffled_permuted_data_files = glob.glob(spdfs)
        for spdf in shuffled_permuted_data_files:
            TOPOLOGY_LOG.info('Computing topology for: %s' % spdf)
            calc_CI_bettis_hierarchical_binned_data(tpid_permuteshuff,
                                                    spdf, block_path, thresh)
        pshuff_results = glob.glob(os.path.join(tpf_permuteshuff, '*-bettiResultsDict.pkl'))[0]    
        with open(pshuff_results, 'r') as f:
            res = pickle.load(f)
            analysis_dict['permuted-shuffled'] = res

    if permuted_average_folder:
        tpid_avgpermute = analysis_id + '-{}-permuted-average'.format(thresh)
        tpf_avgpermute = os.path.join(bpt, tpid_avgpermute)
        avgpermuted_data_files = glob.glob(os.path.join(permuted_average_folder, '*.binned'))
        for pdf in avgpermuted_data_files:
            TOPOLOGY_LOG.info('Computing topology for: %s' % pdf)
            calc_CI_bettis_hierarchical_binned_data(tpid_avgpermute, pdf,
                                                    block_path, thresh)
        pavg_results = glob.glob(os.path.join(tpf_avgpermute, '*-bettiResultsDict.pkl'))[0] 
        with open(pavg_results, 'r') as f:
            res = pickle.load(f)
            analysis_dict['average-permuted'] = res 

    if average_permuted_shuffled_folder:
        tpid_avgpermuteshuff = analysis_id + '-{}-permuted-average-shuffled'.format(thresh)
        tpf_avgpermuteshuff = os.path.join(bpt, tpid_avgpermuteshuff)
        shuffavgpermuted_data_files = glob.glob(os.path.join(average_permuted_shuffled_folder, '*.binned'))
        for pdf in shuffavgpermuted_data_files:
            TOPOLOGY_LOG.info('Computing topology for: %s' % pdf)
            calc_CI_bettis_hierarchical_binned_data(tpid_avgpermuteshuff, pdf,
                                                    block_path, thresh)
        apshuff_results = glob.glob(os.path.join(tpf_avgpermuteshuff, '*-bettiResultsDict.pkl'))[0]
        with open(apshuff_results, 'r') as f:
            res = pickle.load(f)
            analysis_dict['average-permuted-shuffled'] = res

    if permTrialShuffFold:
        tpid_permTrialShuff = analysis_id + '-{}-permuted-trialShuffled'.format(thresh)
        tpf_permTrialShuff = os.path.join(bpt, tpid_permTrialShuff)
        permTrialShuff_files = glob.glob(os.path.join(permTrialShuffFold, '*.binned'))
        for pdf in permTrialShuff_files:
            TOPOLOGY_LOG.info('Computing topology for: %s' % pdf)
            calc_CI_bettis_hierarchical_binned_data(tpid_permTrialShuff, pdf,
                                                    block_path, thresh)
        pTS_results = glob.glob(os.path.join(tpf_permTrialShuff, '*-bettiResultsDict.pkl'))[0]
        with open(pTS_results, 'r') as f:
            res = pickle.load(f)
            analysis_dict['permuted-trialShuffled'] = res

    if simplexWinSize:
        tpid_slidingSimplex = analysis_id + '-{}-slidingSimplex'.format(thresh)
        tpf_slidingSimplex = os.path.join(bpt, tpid_slidingSimplex)
        slidingSimplex_files = glob.glob(os.path.join(slidingSimplexFold, '*.binned'))
        for pdf in slidingSimplex_files:
            TOPOLOGY_LOG.info('Computing topology for: %s' % pdf)
            ssp.sspSlidingSimplex(tpid_slidingSimplex, pdf,
                                                    block_path, thresh, simplexWinSize)
        pSSP_results = glob.glob(os.path.join(tpf_slidingSimplex, '*-bettiResultsDict.pkl'))[0]
        with open(pSSP_results, 'r') as f:
            res = pickle.load(f)
            analysis_dict['slidingSimplex'] = res

    master_fname = analysis_id+'-{}-masterResults.pkl'.format(thresh)
    master_f = os.path.join(block_path, master_fname)
    with open(master_f, 'w') as f:
            pickle.dump(analysis_dict, f)

def mergeMasterResults(masterF, results, resultsName):
    with open(masterF, 'r') as f:
        currentRes = pickle.load(f)
    currentRes[resultsName] = results 
    with open(masterF, 'w') as f:
        pickle.dump(currentRes, f)


def loadRigidPandas(block_path):
    from ephys import rigid_pandas
    spikes = core.load_spikes(block_path)
    stims = rigid_pandas.load_acute_stims(block_path)
    fs = core.load_fs(block_path)
    stims['stim_duration'] = stims['stim_end'] - stims['stim_start']
    rigid_pandas.timestamp2time(stims, fs, 'stim_duration')
    stim_ids = stims['stim_name'].str.replace('_rep\d\d', '')
    stim_ids = stim_ids.str.replace('[a-i]001', '')
    for motif in 'abcdefgh':
        stim_ids = stim_ids.str.replace('[a-i]%s128'%(motif), motif)
    stims['stim_id'] = stim_ids
    rigid_pandas.count_events(stims, index='stim_id')
    spikes = spikes.join(rigid_pandas.align_events(spikes, stims, columns2copy=['stim_id', 'stim_presentation',
                                                   'stim_start', 'stim_duration', 'stim_end']))
    spikes['stim_aligned_time'] = (spikes['time_samples'].values.astype('int') -
                                   spikes['stim_start'].values)
    rigid_pandas.timestamp2time(spikes, fs, 'stim_aligned_time')
    return spikes

def rpGetStimID(rpFrame, trialsRow):
    
    return rpFrame[rpFrame['stim_start'] == trialsRow['time_samples']]['stim_id'].unique()[0]

def rpToTrials(rpFrame):
    
    trialFrame = pd.DataFrame(columns=['time_samples', 'stimulus', 'stimulus_end'])
    trialFrame['time_samples'] = rpFrame['stim_start'].unique()
    trialFrame['stimulus_end'] = rpFrame['stim_end'].unique()
    
    trialFrame['stimulus'] = trialFrame.apply(lambda row: rpGetStimID(rpFrame, row), axis=1)
    
    return trialFrame

def dag_bin_rigid_pandas(block_path, winsize, segment_info, ncellsperm, nperms, nshuffs, cluster_group=['Good'], dtOverlap=0.0):

    block_path = os.path.abspath(block_path)
    # Create directories and filenames
    analysis_id = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    raw_binned_fname = analysis_id + '-{}.binned'.format(winsize)
    analysis_id_forward = analysis_id + '-{}'.format(winsize) 
    
    binned_folder = os.path.join(block_path, 'binned_data/{}/'.format(analysis_id))
    if not os.path.exists(binned_folder):
        os.makedirs(binned_folder)

    trialshuffle_folder = os.path.join(binned_folder, 'trialshuffle/')
    average_binned_folder = os.path.join(binned_folder, 'avgacty/')

    permuted_binned_folder = os.path.join(binned_folder, 'permuted_binned/')
    permuted_average_folder = os.path.join(average_binned_folder, 'permuted_binned/')
    permuted_shuffled_folder = os.path.join(permuted_binned_folder, 'shuffled_controls/')
    average_permuted_shuffled_folder = os.path.join(permuted_average_folder, 'shuffled_controls/')

    permuted_trialshuffle_folder = os.path.join(trialshuffle_folder, 'permuted_binned/')

    raw_binned_f = os.path.join(binned_folder, raw_binned_fname)
    # Load Raw Data
    rpFrame = loadRigidPandas(block_path)
    spikes = rpFrame
    trials = rpToTrials(rpFrame)
    fs = core.load_fs(block_path)
    clusters = core.load_clusters(block_path)
    # Bin the raw data
    TOPOLOGY_LOG.info('Binning data')
    build_population_embedding(spikes, trials, clusters, winsize, fs,
                               cluster_group, segment_info, raw_binned_f, dtOverlap)

    # Average binned raw data
    TOPOLOGY_LOG.info('Averaging activity')
    bin_avg(binned_folder)

    # Permute raw data
    TOPOLOGY_LOG.info('Permuting data') 
    make_permuted_binned_data_recursive(binned_folder, ncellsperm, nperms)

    # Permute Averaged data
    TOPOLOGY_LOG.info('Permuting Averaged data') 
    make_permuted_binned_data_recursive(average_binned_folder, ncellsperm, nperms)

    # Shuffle Permuted data
    TOPOLOGY_LOG.info('Shuffling permuted data') 
    make_shuffled_controls_recursive(permuted_binned_folder, nshuffs)

    # Shuffle Permuted Average Data 
    TOPOLOGY_LOG.info('Shuffling permuted average data')
    make_shuffled_controls_recursive(permuted_average_folder, nshuffs)

    #Trial Shuffle Original Binned
    TOPOLOGY_LOG.info('Making Trial Shuffled Binned')
    make_trialshuffled(binned_folder)

    # Permute TrialShuffled
    TOPOLOGY_LOG.info('Making Permuted Trial Shuffled')
    make_permuted_binned_data_recursive(trialshuffle_folder, ncellsperm, nperms)

    
    bfdict={'permuted': permuted_binned_folder, 'avgpermuted': permuted_average_folder,
           'permutedshuff': permuted_shuffled_folder, 'avgpermshuff': average_permuted_shuffled_folder,
           'raw': binned_folder, 'analysis_id': analysis_id_forward, 'trialshuffled': trialshuffle_folder, 'trialshuffperm': permuted_trialshuffle_folder}
    return bfdict