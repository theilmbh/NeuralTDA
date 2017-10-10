###############################################################################
### topology2.py                                                            ###
### Computing Curto-Itskov topological features from neural population data ###
### Version 2.0: 30 November 2016                                           ###
### Brad Theilman                                                           ###
###############################################################################

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

################################
###### Module Definitions ######
################################

TOPOLOGY_LOG = logging.getLogger('NeuralTDA')

# Path to MCMC Sampler from Young et al. 2017
SCM_EXECUTABLE = '/home/brad/bin/mcmc_sampler'

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
    file_handler = logging.FileHandler(logging_file)
    file_handler.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    formatter.converter = time.gmtime

    # add formatter to ch and sh
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    # add ch and sh to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # Log initialization
    logger.info('Starting {}.'.format(func_name))

def get_spikes_in_window(spikes, window, rec):
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
    rec : int
        Recording ID of spikes that you'd like to extract

    Returns
    -------
    spikes_in_window : pandas DataFrame
        DataFrame with same layout as input spikes
        but containing only spikes within window
    '''
    mask = ((spikes['time_samples'] <= window[1]) &
            (spikes['time_samples'] >= window[0]) &
            (spikes['recording'] == rec))
    return spikes[mask]

def get_windows_for_spike(t, subwin_len, noverlap, segment):
    '''
    Determines bins into which a given spike should be placed

    Parameters
    ----------
    t : integer
        Spike time (samples)
    subwin_len : integer
        window length in samples
    noverlap : integer
        Number of samples of overlap for each bin
    segment : list
        Beginning and end samples of time period

    Returns
    -------
    wins : list
        list of bin IDs to which that spike belongs
    '''
    skip = subwin_len - noverlap
    dur = segment[1] - segment[0]
    A = (int(t) - int(segment[0])) / int(skip)
    J = int(int(subwin_len-1) / int(skip))
    max_k = int(np.floor(float(dur)/float(skip)))

    wins = []
    i0 = int(A)
    wins.append(i0)
    wins = i0 - np.array(range(J+1))
    wins = wins[wins >=0]
    wins = wins[wins < max_k]
    return wins

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

def build_perseus_input(cell_groups, savefile):
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
            out_str = str(grp_dim) + ' ' + vert_str + ' {}\n'.format(str(1))
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
    pfile_split = os.path.splitext(pfile)
    of_string = pfile_split[0]
    perseus_command = "perseus"
    perseus_return_code = subprocess.call([perseus_command, 'nmfsimtop', pfile,
                                           of_string])

    betti_file = of_string+'_betti.txt'
    #betti_file = os.path.join(os.path.split(pfile)[0], betti_file)

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
    segment_info : list
        'segstart' : time in ms of segment start relative to trial start
        'segend' : time in ms of segment end relative to trial end

    Returns
    ------
    segment : list
        bounds for the segment for which to compute topology, in samples.

    '''
    assert type(segment_info) == list
    seg_start = trial_bounds[0] \
                + np.floor(segment_info[0]*(fs/1000.))
    seg_end = trial_bounds[1] \
              + np.floor(segment_info[1]*(fs/1000.))
    return [seg_start, seg_end]

def calc_cell_groups(data_mat, clusters, thresh):
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
    nwin = data_mat.shape[1]
    mean_fr = np.mean(data_mat, axis=1, keepdims=True)
    mean_frs = np.tile(mean_fr, (1, nwin))
    above_thresh = np.greater(data_mat, thresh*mean_frs)
    for win in range(nwin):
        clus_in_group = clusters[above_thresh[:, win]]
        cell_groups.append([win, clus_in_group])
    return cell_groups

def calc_bettis(data_mat, clusters, pfile, thresh):
    '''
    Calculate betti numbers from binned firing rates.

    Parameters
    ----------
    data_mat : Ncells x Nbins array
        Firing rates
    clusters : list
        Cluster ids correspoding to the first dimension of data_mat
    pfile : str
        name of file to store intermediate computations
    thresh : float
        Multiple of average firing rate to consider a cell 'active'

    Returns
    -------
    bettis : list
        list of betti numbers.
        Each element is [<filtration_time>, <betti_vals_list>]
        Returns [-1, [-1]] on error
    '''
    cell_groups = calc_cell_groups(data_mat, clusters, thresh)
    build_perseus_persistent_input(cell_groups, pfile)
    betti_file = run_perseus(pfile)
    bettis = []
    f_time = []
    try:
        with open(betti_file, 'r') as bf:
            for bf_line in bf:
                if len(bf_line) < 2:
                    continue
                betti_data = bf_line.split()
                filtration_time = int(betti_data[0])
                betti_numbers = list(map(int, betti_data[1:]))
                bettis.append([filtration_time, betti_numbers])
    except:
        bettis.append([-1, [-1]])
    return bettis

def get_betti_savefile(aid, apath, stim):
    '''
    Formats filename to save betti csv file
    '''
    bs = aid + '-stim-{}'.format(stim) + '-betti.csv'
    bs = os.path.join(apath, bs)
    return bs

def get_bps(aid, apath, stim):
    '''
    Formats filename to save betti persistence save file
    '''
    bps = aid + '-stim-{}'.format(stim) + '-bettiPersistence.pkl'
    bps = os.path.join(apath, bps)
    return bps

def get_pfile_stem(aid, apath, stim):
    '''
    Formats the file stem for the pfile for perseus computations
    '''
    pfile_stem = aid + '-stim-{}'.format(stim)
    pfile_stem = os.path.join(apath, pfile_stem)
    return pfile_stem

def get_analysis_paths(aid, apath, stim):
    '''
    Computes all the required file paths for the betti computation
    '''
    ###  Prepare destination file paths
    bs = get_betti_savefile(aid, apath, stim)

    bps = get_bps(aid, apath, stim)

    pfs = get_pfile_stem(aid, apath, stim)
    return (bs, bps, pfs)

def get_pfile_name(pfile_stem, **kwargs):
    '''
    Formats a pfile name for perseus
    '''
    for key, val in kwargs.items():
        pfile_stem = pfile_stem + '-%s%d' % (str(key), int(val))
    pfile = pfile_stem +'-simplex.txt'
    return pfile

def lin2ind(shp, t):
    inds = []
    l = t
    for k in range(len(shp)-1):
        a = int(np.product(shp[(1+k):]))
        w = int(l)/a
        l = np.mod(t, a)
        inds.append(w)
    inds.append(l)
    return inds

############################################
###### Topology Computation Functions ######
############################################

def prep_paths(analysis_id, binned_data_file, block_path, shuffle, nperms):
    '''
    Formats analysis paths for betti computation
    '''
    bdf_name, ext = os.path.splitext(os.path.basename(binned_data_file))
    analysis_path = os.path.join(block_path,
                                 'topology/{}/'.format(analysis_id))
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)

    if shuffle:
        analysis_id = analysis_id +'-shuffle-'
    if nperms:
        analysis_id = analysis_id + '-permuted{}-'.format(nperms)

    return (analysis_id, analysis_path)

def calc_CI_bettis_tensor(analysis_id, binned_data_file,
                          block_path, thresh, shuffle=False, nperms=0,
                          ncellsperm=1, clusters=None):
    '''
    Given a binned data file, compute the betti numbers of the Curto-Itskov

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
    (analysis_id, analysis_path) = prep_paths(analysis_id, binned_data_file,
                                              block_path, shuffle, nperms)
    with h5py.File(binned_data_file, 'r') as bdf:
        stims = bdf.keys()
        bpd_withstim = dict()
        for stim in stims:
            binned_clusters = np.array(bdf[stim]['clusters'])
            stim_trials = bdf[stim]
            (bs, bps, pfs) = get_analysis_paths(analysis_id,
                                                analysis_path,
                                                stim)
            bpd = dict()
            ### Compute Bettis
            poptens = np.array(stim_trials['pop_tens'])
            clusters = np.array(stim_trials['clusters'])
            bpd = do_compute_betti(poptens, clusters, pfs, thresh,
                                   shuffle, nperms, ncellsperm)
            bpd_withstim[stim] = bpd
            with open(bps, 'wb') as bpfile:
                pickle.dump(bpd, bpfile)
        bpdws_sfn = os.path.join(analysis_path,
                                 analysis_id+'-bettiResultsDict.pkl')
        with open(bpdws_sfn, 'wb') as bpdwsfile:
            pickle.dump(bpd_withstim, bpdwsfile)
        return (bpdws_sfn, bpd_withstim)


def calc_CI_bettis_tensor_trialavg(analysis_id, binned_data_file,
                          block_path, thresh, shuffle=False, nperms=0,
                          ncellsperm=1, clusters=None):
    '''
    Given a binned data file, compute the betti numbers of the Curto-Itskov
    Average all trials for a given stim before computing bettis
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
    (analysis_id, analysis_path) = prep_paths(analysis_id, binned_data_file,
                                              block_path, shuffle, nperms)
    with h5py.File(binned_data_file, 'r') as bdf:
        stims = bdf.keys()
        bpd_withstim = dict()
        for stim in stims:
            binned_clusters = np.array(bdf[stim]['clusters'])
            stim_trials = bdf[stim]
            (bs, bps, pfs) = get_analysis_paths(analysis_id,
                                                analysis_path,
                                                stim)
            bpd = dict()
            ### Compute Bettis
            poptens = np.array(stim_trials['pop_tens'])
            clusters = np.array(stim_trials['clusters'])

            ## do trialaverage
            poptens = np.mean(poptens, axis=2)
            bpd = do_compute_betti(poptens[:, :, np.newaxis], clusters, pfs,
                                   thresh, shuffle, nperms, ncellsperm)
            bpd_withstim[stim] = bpd
            with open(bps, 'wb') as bpfile:
                pickle.dump(bpd, bpfile)
        bpdws_sfn = os.path.join(analysis_path,
                                 analysis_id+'-bettiResultsDict.pkl')
        with open(bpdws_sfn, 'wb') as bpdwsfile:
            pickle.dump(bpd_withstim, bpdwsfile)
        return (bpdws_sfn, bpd_withstim)

def do_compute_betti(poptens, clusters, pfile_stem, thresh,
                     shuffle, nperms, ncellsperm):

    '''
    Function to actually perform the betti number computation
    '''
    data_tensor = np.array(poptens)
    clusters = np.array(clusters)
    levels = (data_tensor.shape)[2:] # First two axes are cells, windows.
    assert len(levels) == 1, 'Cant handle more than one level yet'
    ntrials = levels[0]
    bettidict = {}
    for trial in range(ntrials):
        pfile = pfile_stem + '-rep%d-simplex.txt' % trial
        pfile = get_pfile_name(pfile_stem, rep=trial)
        data_mat = data_tensor[:, :, trial]
        if nperms:
            bettipermdict = {}
            (new_tensor, perm_cells) = get_perms(data_mat, nperms, ncellsperm)
            for perm in range(nperms):
                pfile = get_pfile_name(pfile_stem, rep=trial, perm=perm)
                nmat = new_tensor[:, :, perm]
                if shuffle:
                    nmat = get_shuffle(nmat)
                perm_clus = clusters[perm_cells[:, perm]]
                bettis = calc_bettis(nmat, perm_clus, pfile, thresh)
                bettipermdict[str(perm)] = {'bettis': bettis}
            bettidict[str(trial)] = bettipermdict
        else:
            if shuffle:
                data_mat = get_shuffle(data_mat)
                pfile = get_pfile_name(pfile_stem, rep=trial, shuffled=1)
            bettis = calc_bettis(data_mat, clusters, pfile, thresh)
            bettidict[str(trial)] = {'0': {'bettis': bettis}}
    return bettidict

def get_shuffle(data_mat):
    '''
    Shuffles a data matrix, each cell independently in time
    '''
    (cells, wins) = data_mat.shape
    for cell in range(cells):
        np.random.shuffle(data_mat[cell, :])
    return data_mat

def get_perms(data_mat, nperms, ncellsperm):
    '''
    Permutes the data matrix by building a data_tensor from random subsets
    of the population
    '''
    (cells, wins) = data_mat.shape
    if ncellsperm > cells:
        ncellsperm = cells
    new_tensor = np.zeros((ncellsperm, wins, nperms))
    perm_cells = np.zeros((ncellsperm, nperms)).astype(int)
    for perm in range(nperms):
        celllist = np.random.permutation(cells)[:ncellsperm]
        new_tensor[:, :, perm] = data_mat[celllist, :]
        perm_cells[:, perm] = celllist
    return (new_tensor, perm_cells)

###############################
###### Binning Functions ######
###############################

def build_binned_file_quick(spikes, trials, clusters, win_size, fs,
                            cluster_group, segment_info,
                            popvec_fname, dt_overlap=0.0):
    '''
    Embeds binned population activity into R^n
    resulting Tensor is Ncell x Nwin x NTrials

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

        # Extract clusters to bin
        if cluster_group != None:
            mask = np.ones(len(clusters.index)) < 0
            for grp in cluster_group:
                mask = np.logical_or(mask, clusters['quality'] == grp)
        clusters_to_use = clusters[mask]
        clusters_list = clusters_to_use['cluster'].unique()
        nclus = len(clusters_to_use.index)

        # Extract spikes
        spikes = spikes[spikes['cluster'].isin(clusters_list)]

        # Set binned file attributes
        popvec_f.attrs['win_size'] = win_size
        popvec_f.attrs['fs'] = fs
        popvec_f.attrs['nclus'] = nclus
        stims = trials['stimulus'].unique()

        for stim in stims:
            if str(stim) == 'nan':
                continue
            # Create Stimulus data group
            stimgrp = popvec_f.create_group(stim)

            # Extract stimulus trial information
            stim_trials = trials[trials['stimulus'] == stim]
            nreps = len(stim_trials.index)
            stim_recs = stim_trials['recording'].values
            trial_len = (stim_trials['stimulus_end'] \
                         - stim_trials['time_samples']).unique()[0]

            # Compute subwindow length and overlap in samples
            subwin_len = int(np.round(win_size/1000. * fs))
            noverlap = int(np.round(dt_overlap/1000. * fs))
            segment = get_segment([0, trial_len], fs, segment_info)

            # Bin the stimulus data into a population tensor
            poptens = build_activity_tensor_quick(stim_trials, spikes,
                                                  clusters_list, nclus,
                                                  win_size, subwin_len,
                                                  noverlap, segment)

            # Create the dataset and set attributes
            poptens_dset = stimgrp.create_dataset('pop_tens', data=poptens)
            stimgrp.create_dataset('clusters', data=clusters_list)
            poptens_dset.attrs['fs'] = fs
            poptens_dset.attrs['win_size'] = win_size

def build_activity_tensor_quick(stim_trials, spikes, clusters_list, nclus,
                                win_size, subwin_len, noverlap, segment):
    '''
    Bins population spike times into a population activity tensor
    The resultant tensor is Ncells x Nbins x Ntrials

    Parameters
    ----------
    stim_trials : pandas DataFrame
        Subset of trials dataframe from ephys_analysis containing
        all of the trial information from a single stimulus
    spikes : pandas DataFrame
        Subset of spikes dataframe from ephys_analysis containing
        all of the spike information for all the trials from a
        single stimulus
    clusters_list : list
        List of Cluster IDs to use in the binning
    nclus : int
        Number of clusters
    win_size : float
        window (bin) size in milliseconds
    subwin_len : int
        window (bin) length in samples
    noverlap : int
        bin overlap in samples
    segment : list
        return value of get_segment

    Returns
    ------
    poptens : numpy array
        Ncells x Nbin x Ntrials population activity tensor
    '''
    nreps = len(stim_trials.index)
    stim_recs = stim_trials['recording'].values
    skip = subwin_len - noverlap
    dur = segment[1] - segment[0]
    if dur <= 0:
        # segment does not match with trial length
        # return nothing
        print('Activity Tensor: Duration <= 0')
        return []
    nwins = int(np.round(float(dur)/float(skip)))
    poptens = np.zeros((nclus, nwins, nreps))
    for rep in range(nreps):
        trial_start = stim_trials.iloc[rep]['time_samples']
        trial_end = stim_trials.iloc[rep]['stimulus_end']
        samp_period = (trial_start + segment[0], trial_start + segment[1])
        rec = stim_recs[rep]
        stim_rec_spikes = get_spikes_in_window(spikes, samp_period, rec)
        sptimes = stim_rec_spikes['time_samples'].values
        clusters = stim_rec_spikes['cluster'].values
        for sp, clu in zip(sptimes, clusters):
            wins = get_windows_for_spike(sp, subwin_len, noverlap, samp_period)
            poptens[clusters_list==clu, wins, rep] += 1
    poptens /= (win_size/1000.0)
    return poptens

def scramble(a, axis=-1):
    b = np.random.random(a.shape)
    idx = np.argsort(b, axis=axis)
    shuffled = a[np.arange(a.shape[0])[:, None], idx]
    return shuffled

def build_shuffled_data_tensor(data_tens, nshuffs):
    '''
    Shuffles a data tensor
    '''
    ncells, nwin, ntrial = data_tens.shape
    shuff_tens = np.zeros((ncells, nwin, ntrial, nshuffs))
    for shuff in range(nshuffs):
        for trial in range(ntrial):
            shuff_tens[:, :, trial, shuff] = scramble(data_tens[:, :, trial])
    return shuff_tens

def build_permuted_data_tensor(data_tens, ncellsperm, nperms):
    '''
    Permutes a data tensor
    '''
    ncells, nwin, ntrial = data_tens.shape
    perm_tens = np.zeros((ncellsperm, nwin, ntrial, nperms))

    for trial in range(ntrial):
        perm_tens[:, :, trial, :] = get_perms(data_tens[:, :, trial],
                                              nperms, ncellsperm)[0]
    return perm_tens

def extract_population_tensors(binned_datafile, shuffle=False, clusters=None):
    '''
    Returns a dictionary containing all the population tensors for each stimulus

    Parameters
    ----------
    shuffle : bool
        If True, returns a shuffled population tensor
    clusters : list
        List of clusters to include in the returned population tensors.
        If None, all clusters in binned file are returned
    '''
    with h5py.File(binned_datafile, 'r') as bdf:
        stims = bdf.keys()
        print(stims)
        stim_tensors = dict()
        for ind, stim in enumerate(stims):
            binned_clusters = np.array(bdf[stim]['clusters'])
            poptens = np.array(bdf[stim]['pop_tens'])
            print('Stim: {}, Clusters:{}'.format(stim, str(clusters)))
            try:
                if clusters is not None:
                    poptens = poptens[np.in1d(binned_clusters, clusters), :, :]
                    print("Selecting Clusters: poptens:" 
                            + str(np.shape(poptens)))
                (ncell, nwin, ntrial) = np.shape(poptens)
            except (ValueError, IndexError):
                print('Poptens Error')
                continue
            if shuffle:
                poptens = build_shuffled_data_tensor(poptens, 1)
                poptens = poptens[:, :, :, 0]
            if  nwin == 0:
                continue
            stim_tensors[stim] = poptens
    return stim_tensors

def extract_population_tensor(binned_data_file, stim,
                              shuffle=False, clusters=None):
    '''
    Extracts the population tensor from a binned data file for
    a specific stim.  Shuffles it on command.
    Selects clusters given by 'clusters', all if none
    '''
    print('Extracting Population Activity Tensor...')
    with h5py.File(binned_data_file, 'r') as bdf:
        binned_clusters = np.array(bdf[stim]['clusters'])
        poptens = np.array(bdf[stim]['pop_tens'])
        print('Stim: {}, Clusters:{}'.format(stim, str(clusters)))
        try:
            if clusters is not None:
                poptens = poptens[np.in1d(binned_clusters, clusters), :, :]
                print("Selecting Clusters: poptens:" + str(np.shape(poptens)))
            (ncell, nwin, ntrial) = np.shape(poptens)
        except (ValueError, IndexError):
            print('Population Tensor Error')
            return []
        if shuffle:
            poptens = tp2.build_shuffled_data_tensor(poptens, 1)
            poptens = poptens[:, :, :, 0]
        if  nwin == 0:
            return []
    return poptens

##########################
###### SCM Controls ######
##########################

def num_trials(poptens):
    (ncells, nwin, ntrials) = np.shape(poptens)
    return ntrials

def num_win(poptens):
    (ncells, nwin, ntrials) = np.shape(poptens)
    return nwin

def num_cells(poptens):
    (ncells, nwin, ntrials) = np.shape(poptens)
    return ncells

def rejection_sampling(command, seed=0):
    # Call sampler with subprocess
    proc = subprocess.run(command, stdout=subprocess.PIPE)
    # Read output as a facet list
    facet_list = []
    for line in proc.stdout.decode().split("\n")[1:-1]:
        if line.find("#") == 0:
            yield facet_list
            facet_list = []
        else:
            facet_list.append([int(x) for x in line.strip().split()])
    yield facet_list

def prepare_scm_initial_condition(binmat, **kwargs):

    facets = sc.binarytomaxsimplex(binmat, rDup=True, **kwargs)
    with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as f:
        fname = f.name
        for facet in facets:
            fstr = str(facet)
            fstr = fstr.replace('(', '')
            fstr = fstr.replace(')', '')
            fstr = fstr.replace(',', '')
            f.write(fstr+'\n')
    return fname

def prepare_scm_command(facet_file, nsamps):

    command = [SCM_EXECUTABLE, facet_file, '-t', str(nsamps)]
    return command

def calc_scm_betti_distribution(poptens, thresh, trial, nsamples):
    '''
    Use the simplicial configuration model of Young et al. 2017
    to compute the null distribution of betti numbers for a specific trial
    in a population activity tensor
    '''
    popmat = poptens[:, :, trial]
    popmat_bin = sc.binnedtobinary(popmat, thresh)
    fname = prepare_scm_initial_condition(popmat_bin)
    cmd = prepare_scm_command(fname, nsamples)
    samples = rejection_sampling(cmd)
    sample_bettis = []
    for sample in tqdm.tqdm(samples):
        bettis=[]
        cgs = [[1, x] for x in sample]
        build_perseus_input(cgs, '/home/brad/betti_pfile.txt')
        betti_file = run_perseus('/home/brad/betti_pfile.txt')
        try:
            with open(betti_file, 'r') as bf:
                for bf_line in bf:
                    betti_numbers_arr = np.zeros(10)
                    if len(bf_line) < 2:
                        continue
                    betti_data = bf_line.split()
                    filtration_time = int(betti_data[0])
                    betti_numbers = np.array(list(map(int, betti_data[1:])))
                    betti_numbers_arr[0:len(betti_numbers)] = betti_numbers
                    bettis.append(np.array(betti_numbers_arr))
        except:
            bettis.append(-1*np.ones(10))
        sample_bettis.append(bettis)
    return np.array(sample_bettis)

def get_scg_at_time(binmat, t):
    '''
    Extract scg at a specific time
    time is given as specific index
    '''

    (ncells, nwins) = binmat.shape
    subpopmat = binmat[:, 0:t]
    scg = sc.binarytomaxsimplex(binmat, rDup=True)
    return scg

##############################
###### Betti Curve Funcs #####
##############################

def betti_dict_to_betti_curves(betti_dict, dims, twin, windt, dtovr):
    '''
    Interpolates Betti values using step functions to produce
    Betti curves

    Parameters
    ----------
    betti_dict : dict 
        Dictionary returned by betti computation
    dims : list 
        Dimensions to compute betti curves for 
    twin : list 
        list of interpolation time points
    windt : float 
        window size in milliseconds 
    dtovr : float 
        amount of window overlap in milliseconds
    '''

    stim_betticurves = {}
    for stim in betti_dict.keys():
        betticurve_save = np.empty((len(dims), len(twin), 0))

        trials = betti_dict[stim]
        for trial in trials.keys():
            perms = trials[trial]
            for perm in perms.keys():
                dat = perms[perm]['bettis']
                t = np.array([int(x[0]) for x in dat])
                t_milliseconds = t*((windt - dtovr)) + windt / 2.0
                t_vals = np.round((twin - windt/2) /(windt-dtovr))
                t_vals_milliseconds = twin
                b = [x[1] for x in dat]
                #t_vals = np.linspace(window[0], window[1], Ntimes)
                #t_vals = np.linspace(np.amin(t), np.amax(t), Ntimes)
                #t_vals_milliseconds = t_vals*((windt - dtovr)) + windt / 2.0
                b = [np.pad(np.array(x), (0, 10),
                            'constant', constant_values=0)
                     for x in b]
                betticurve_save_alldims = np.empty((0, len(twin)))
                for dim in dims:
                    b_val = np.array([x[dim] for x in b])
                    b_func = interp1d(t, b_val,
                                      kind='zero', bounds_error=False,
                                      fill_value=(b_val[0], b_val[-1]))
                    betti_curve_dim = b_func(t_vals)
                    betticurve_save_alldims = np.vstack((betticurve_save_alldims,
                                                        betti_curve_dim))
                betticurve_save = np.concatenate((betticurve_save,
                                                  betticurve_save_alldims[:, :,
                                                  np.newaxis]), axis=2)
        stim_betticurves[stim] = np.array(betticurve_save)
    return (stim_betticurves, t_vals, t_vals_milliseconds)

def compute_betti_curves(analysis_id, block_path, bdf,
                         thresh, nperms, ncellsperm, dims, twin,
                        windt, dtovr, shuffle=False):

    (resf, betti_dict) = calc_CI_bettis_tensor(analysis_id, bdf,
                              block_path, thresh, shuffle=shuffle,
                              nperms=nperms, ncellsperm=ncellsperm)

    return betti_dict_to_betti_curves(betti_dict, dims, twin, windt, dtovr)


def compute_trialaverage_betti_curves(analysis_id, block_path, bdf,
                         thresh, nperms, ncellsperm, dims, twin,
                         windt, dtovr, shuffle=False):

    (resf, betti_dict) = calc_CI_bettis_tensor_trialavg(analysis_id, bdf,
                              block_path, thresh, shuffle=shuffle,
                              nperms=nperms, ncellsperm=ncellsperm)

    return betti_dict_to_betti_curves(betti_dict, dims, twin, windt, dtovr)

##############################
###### Computation Dags ######
##############################

def db_load_data(block_path):

    # Load Raw Data
    spikes = core.load_spikes(block_path)
    trials = events.load_trials(block_path)
    fs = core.load_fs(block_path)
    clusters = core.load_clusters(block_path)
    return (spikes, trials, clusters, fs)

def bin_data(block_path, winsize, segment_info, **kwargs):
    '''
    Bins spiking data into population tensors.

    Parameters
    ----------
    block_path : str 
        Path to directory containing Kwik file of spike data 
    winsize : float 
        Width of the binning windows in MILLISECONDS
    segment_info : list 
        [a, b] gives a MILLISECONDS after stimulus start 
        and b MILLISECONDS before stimulus end 
    kwargs :
        cluster_group : list 
            Quality of clusters to include. e.g. ['Good', 'MUA']
            includes all Good and MUA clusters 
        dt_overlap : float 
            MILLISECONDS of overlap for each window 
        comment : str 
            A string to tag the resultant binned file with 
    '''
    (spikes, trials, clusters, fs) = db_load_data(block_path)
    bfdict = do_dag_bin_lazy(block_path, spikes, trials, clusters,
                             fs, winsize, segment_info, **kwargs)
    return bfdict 

def dag_bin(block_path, winsize, segment_info, **kwargs):

    (spikes, trials, clusters, fs) = db_load_data(block_path)
    bfdict = do_dag_bin_lazy(block_path, spikes, trials, clusters,
                             fs, winsize, segment_info, **kwargs)
    return bfdict

def do_dag_bin_lazy(block_path, spikes, trials, clusters, fs, winsize,
                    segment_info, cluster_group=['Good', 'MUA'], 
                    dt_overlap=0.0, comment=''):
    '''
    Take data structures from ephys_analysis and bin them into
    population tensors.
    This version checks to see if a binning with the same parameters already
    exists and returns the path to that file if so.

    Parameters
    ----------
    block_path : str
        Path to the directory containing the kwik file of the data to process.
    spikes : DataFrame
        Spike dataframe containing all the spikes for the dataset
    trials : DataFrame
        Trial DataFrame containing all the trials you'd like to bin.
    clusters : DataFrame
        cluster dataframe containing information for all the clusters
    fs : int
        sampling rate
    winsize :  float
        window size in milliseconds.
    segment_info : list
        List containing time in milliseconds relative to trial start for the
        beginning of the segment and time in ms relative to trial end for the
        end of the segment
    cluster_group : list
        List of strings of cluster groups to include in binning.
    dt_overlap : float
        time in milliseconds for windows to overlap
    comment : str
        string identifying anything special about this binning.

    Returns
    -------
    bfdict : dict
        dictionary containing paths to binned folders.

    '''
    #setup_logging('Dag Bin')
    block_path = os.path.abspath(block_path)
    # Create directories and filenames
    analysis_id = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    raw_binned_fname = analysis_id + '-{}-{}.binned'.format(winsize, dt_overlap)
    analysis_id_forward = analysis_id + '-{}-{}'.format(winsize, dt_overlap)
    bfdict = {'analysis_id': analysis_id_forward}

    #cg_string = '-'.join(cluster_group)
    seg_string = '-'.join(map(str, segment_info))
    if comment:
        seg_string = seg_string+ '-'+comment
    bin_string = 'binned_data/win-{}_dtovr-{}_seg-{}/'.format(winsize,
                                                              dt_overlap,
                                                              seg_string)
    binned_folder = os.path.join(block_path, bin_string)
    if not os.path.exists(binned_folder):
        os.makedirs(binned_folder)
    existing_binned = glob.glob(os.path.join(binned_folder, '*.binned'))

    if len(existing_binned) == 0:
        # not already binned!
        # Bin the raw data
        print('Data Not already binned')
        raw_binned_f = os.path.join(binned_folder, raw_binned_fname)
        build_binned_file_quick(spikes, trials, clusters, winsize, fs,
                              cluster_group, segment_info,
                              raw_binned_f, dt_overlap)
    else:
        raw_binned_f = existing_binned[0]

    bfdict['raw'] = binned_folder
    return bfdict

def dag_topology(block_path, thresh, bfdict, raw=True,
                 shuffle=False, shuffleperm=False, nperms=0, ncellsperm=1,
                 **kwargs):

    aid = bfdict['analysis_id']
    analysis_dict = dict()
    raw_folder = bfdict['raw']

    if 'raw' in bfdict.keys() and raw:
        tpid_raw = aid +'-{}-raw'.format(thresh)
        raw_data_files = glob.glob(os.path.join(raw_folder, '*.binned'))
        for rdf in raw_data_files:
            res_f = calc_CI_bettis_tensor(tpid_raw, rdf, block_path,
                                      thresh, **kwargs)[0]
        with open(res_f, 'r') as f:
            res = pickle.load(f)
            analysis_dict['raw'] = res

    if shuffle:
        tpid_raw = aid +'-{}-raw-shuffled'.format(thresh)
        raw_data_files = glob.glob(os.path.join(raw_folder, '*.binned'))
        for rdf in raw_data_files:
            res_f = calc_CI_bettis_tensor(tpid_raw, rdf, block_path, thresh,
                                      shuffle=True, **kwargs)[0]
        with open(res_f, 'r') as f:
            res = pickle.load(f)
            analysis_dict['rawshuffled'] = res

    if nperms:
        tpid_raw = aid +'-{}-permuted-{}-{}'.format(thresh, nperms, ncellsperm)
        raw_data_files = glob.glob(os.path.join(raw_folder, '*.binned'))
        for rdf in raw_data_files:
            res_f = calc_CI_bettis_tensor(tpid_raw, rdf, block_path, thresh,
                                      nperms=nperms, ncellsperm=ncellsperm,
                                      **kwargs)[0]
        with open(res_f, 'r') as f:
            res = pickle.load(f)
            analysis_dict['permuted'] = res

    if shuffleperm:
        tpid_raw = aid +'-{}-shuffled-permuted-{}-{}'.format(thresh, nperms,
                                                             ncellsperm)
        raw_data_files = glob.glob(os.path.join(raw_folder, '*.binned'))
        for rdf in raw_data_files:
            res_f = calc_CI_bettis_tensor(tpid_raw, rdf, block_path, thresh,
                                      shuffle=True, nperms=nperms,
                                      ncellsperm=ncellsperm, **kwargs)[0]
        with open(res_f, 'r') as f:
            res = pickle.load(f)
            analysis_dict['shuffledpermuted'] = res

    if 'alltrials' in bfdict.keys():
        at_folder = bfdict['alltrials']
        tpid_at = aid + '-{}-at'.format(thresh)
        atDataFiles = glob.glob(os.path.join(at_folder, '*.binned'))
        for atdf in atDataFiles:
            res_f = calc_CI_bettis_all_trials(tpid_at, atdf,
                                               block_path, thresh)[0]
        with open(res_f, 'r') as f:
            res = pickle.load(f)
            analysis_dict['at'] = res

    master_fname = aid
    for key, val in kwargs.iteritems():
        master_fname = master_fname+'-{}_{}'.format(key, val)
    master_fname = master_fname+'-{}-masterResults.pkl'.format(thresh)
    master_fname = os.path.join(block_path, master_fname)
    with open(master_fname, 'w') as master_f:
        pickle.dump(analysis_dict, master_f)
    return master_f
