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
from itertools import product

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

def calcCellGroups(data_mat, clusters, thresh):
    '''
    Given a matrix of firing rates, generate time sequence of cell groups

    Parameters
    ----------
    data_mat : numpy array
        Ncells x Nwindows of firing rates 
    clusters : numpy array 
        array of length NCells specifying cluster id for each row of data_mat 
    '''
    cell_groups = []
    Ncells, nwin = data_mat.shape
    mean_fr = np.mean(data_mat, axis=1, keepdims=True)
    mean_frs = np.tile(mean_fr, (1, nwin))
    above_thresh = np.greater(data_mat, thresh*mean_frs)
    for win in range(nwin):
        clus_in_group = clusters[above_thresh[:, win]]
        cell_groups.append([win, clus_in_group])
    return cell_groups 

def calcBettis(data_mat, clusters, pfile, thresh):
    '''
    Calculate bettis, matrix form
    '''
    cell_groups = calcCellGroups(data_mat, clusters, thresh)
    build_perseus_persistent_input(cell_groups, pfile)
    betti_file = run_perseus(pfile)
    bettis = []
    try:
        with open(betti_file, 'r') as bf:
            for bf_line in bf:
                if len(bf_line) < 2:
                    continue
                betti_data = bf_line.split()
                filtration_time = int(betti_data[0])
                betti_numbers = map(int, betti_data[1:])
                bettis.append([filtration_time, betti_numbers])
    except:
        bettis.append([-1, [-1]])
        TOPOLOGY_LOG.warn('Perseus returned invalid betti file')
    return bettis

#######################################
###### Current Binning Functions ######
#######################################

def calcCIBettisTensor(analysis_id, binned_data_file,
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
    TOPOLOGY_LOG.info('Starting calcCIBettisTensor')
    TOPOLOGY_LOG.info('analysis_id: {}'.format(analysis_id))
    bdf_name, ext = os.path.splitext(os.path.basename(binned_data_file))
    analysis_path = os.path.join(block_path,
                                 'topology/{}/'.format(analysis_id))
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)

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
            stim_trials = bdf[stim]
            ###  Prepare destination file paths
            betti_savefile = analysis_id \
                             + '-stim-{}'.format(stim) \
                             + '-betti.csv'
            betti_savefile = os.path.join(analysis_path, betti_savefile)
            TOPOLOGY_LOG.info('Betti savefile: {}'.format(betti_savefile))
            bps = analysis_id \
                  + '-stim-{}'.format(stim) \
                  + '-bettiPersistence.pkl'
            betti_persistence_savefile = os.path.join(analysis_path, bps)
            TOPOLOGY_LOG.info('Betti persistence \
                               savefile: {}'.format(betti_persistence_savefile))
            bpd = dict()
            pfile_stem = analysis_id \
                         + '-stim-{}'.format(stim)
            pfile_stem = os.path.join(analysis_path, pfile_stem)
            ### Compute Bettis
            bpd = do_compute_betti(stim_trials,
                                    pfile_stem,
                                    analysis_path,
                                    thresh)
            bpd_withstim[stim] = bpd
            with open(betti_persistence_savefile, 'w') as bpfile:
                pickle.dump(bpd, bpfile)
        bpdws_sfn = os.path.join(analysis_path, analysis_id+'-bettiResultsDict.pkl')
        with open(bpdws_sfn, 'w') as bpdwsfile:
                pickle.dump(bpd_withstim, bpdwsfile)
        TOPOLOGY_LOG.info('Completed All Stimuli')
        return bpdws_sfn

def do_compute_betti(stim_trials, pfile_stem, analysis_path, thresh):

    assert 'pop_tens' in stim_trials.keys(), 'No Data Tensor!!'
    data_tensor = np.array(stim_trials['pop_tens'])
    clusters = np.array(stim_trials['clusters'])
    levels = (data_tensor.shape)[2:] # First two axes are cells, windows. 
    assert len(levels) == 1, 'Cant handle more than one level yet' 
    bettidict = {}
    for trial in range(levels[0]):
        pfile = pfile_stem + '-rep%d-simplex.txt' % trial
        data_mat = data_tensor[:, :, trial]
        bettis = calcBettis(data_mat, clusters, pfile, thresh)
        bettidict[str(trial)] = {'bettis': bettis}
    return bettidict

def build_population_embedding_tensor(spikes, trials, clusters, win_size, fs,
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

            # Compute generic windows for this stimulus
            trial_len = (stim_trials['stimulus_end'] - stim_trials['time_samples']).unique()[0]
            gen_seg_start, gen_seg_end = get_segment([0, trial_len], fs, segment_info)
            gen_seg = [gen_seg_start, gen_seg_end]
            win_size_samples = int(np.round(win_size/1000. * fs))
            overlap_samples = int(np.round(dtOverlap/1000. * fs))
            gen_windows = create_subwindows(gen_seg, win_size_samples, overlap_samples)
            nwins = len(gen_windows)

            # Create Data set
            poptens_init = np.zeros((nclus, nwins, nreps))
            poptens_dset = stimgrp.create_dataset('pop_tens', data=poptens_init)
            stimgrp.create_dataset('clusters', data=clusters_list)
            stimgrp.create_dataset('windows', data=np.array(gen_windows))
            poptens_dset.attrs['fs'] = fs
            poptens_dset.attrs['win_size'] = win_size

            for rep in range(nreps):
                trial_start = stim_trials.iloc[rep]['time_samples']
                trial_start_t = float(trial_start)/fs
                for win_ind, win in enumerate(gen_windows):
                    win2 = [win[0] + trial_start, win[1]+trial_start]
                    spikes_in_win = get_spikes_in_window(spikes, win2)
                    clus_that_spiked = spikes_in_win['cluster'].unique()
                    if len(clus_that_spiked) > 0:
                        for clu in clus_that_spiked:
                            clu_msk = (spikes_in_win['cluster'] == clu)
                            pvclu_msk = (clusters_list == clu)
                            nsp_clu = float(len(spikes_in_win[clu_msk]))
                            win_s = win_size/1000.
                            poptens_dset[pvclu_msk, win_ind, rep] = nsp_clu/win_s

def build_permuted_data_tensor(data_tens, clusters, ncellsperm, nperms):
    ''' Builds a permuted data tensor 

    Parameters
    ----------
    data_tens : numpy array 
        NCells x nWin x nTrial array 
    clusters : numpy array 
        nCells x 1, giving cellID 
        for corresponding row in data_tens
    ncellsperm : int 
        number of cells in a permutation
    nperms : int 
        number of permutations 

    Returns
    -------
    ptens : numpy array 
        permuted data tensor
        nCellsPerm x nWin x nTrial x nPerms 
    clumapmat : numpy array 
        Cluster mapping- cluster id of each row in ptens 
        nCellsPerm x nPerms 
    '''
    nCells, nWin, nTrial = data_tens.shape
    ptens = np.zeros((ncellsperm, nWin, nTrial, nperms))
    clumapmat = np.zeros((ncellsperm, nperms))

    for perm in range(nperms):
        permt = np.random.permutation(nCells)
        if nCells >= ncellsperm:
            permt = permt[0:ncellsperm].tolist()
        else:
            permt = permt.tolist()
        ptens[:, :, :, perm] = data_tens[permt, :, :]
        clumapmat[:, perm] = clusters[permt]
    return (ptens, clumapmat)

##############################
###### Computation Dags ######
##############################

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

def do_dag_bin(block_path, spikes, trials, clusters, fs, winsize, segment_info,
               ncellsperm, nperms, nshuffs, cluster_group=['Good'], dtOverlap=0.0):

    block_path = os.path.abspath(block_path)
    # Create directories and filenames
    analysis_id = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    raw_binned_fname = analysis_id + '-{}-{}.binned'.format(winsize, dtOverlap)
    analysis_id_forward = analysis_id + '-{}-{}'.format(winsize, dtOverlap) 
    
    binned_folder = os.path.join(block_path, 'binned_data/{}/'.format(analysis_id))
    if not os.path.exists(binned_folder):
        os.makedirs(binned_folder)

    # Bin the raw data
    raw_binned_f = os.path.join(binned_folder, raw_binned_fname)
    TOPOLOGY_LOG.info('Binning data')
    build_population_embedding_tensor(spikes, trials, clusters, winsize, fs,
                                      cluster_group, segment_info, raw_binned_f, dtOverlap)

    bfdict = {'raw': binned_folder, 'analysis_id': analysis_id_forward}
    return bfdict 

def dag_topology(block_path, thresh, bfdict, simplexWinSize=0):

    aid = bfdict['analysis_id']
    bpt = os.path.join(block_path, 'topology/')
    analysis_dict = dict()
    if 'raw' in bfdict.keys():
        rawFolder = bfdict['raw']
        tpid_raw = aid +'-{}-raw'.format(thresh)
        tpf_raw = os.path.join(bpt, tpid_raw)
        rawDataFiles = glob.glob(os.path.join(rawFolder, '*.binned'))
        for rdf in rawDataFiles:
            TOPOLOGY_LOG.info('Computing topology for: %s' % rdf)
            resF = calcCIBettisTensor(tpid_raw, rdf, block_path, thresh)
        with open(resF, 'r') as f:
            res = pickle.load(f)
            analysis_dict['raw'] = res

    master_fname = aid+'-{}-masterResults.pkl'.format(thresh)
    master_f = os.path.join(block_path, master_fname)
    with open(master_f, 'w') as f:
            pickle.dump(analysis_dict, f)
    return master_f