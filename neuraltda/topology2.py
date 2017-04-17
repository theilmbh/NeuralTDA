################################################################################
### topology2.py                                                            ####
### Computing Curto-Itskov topological features from neural population data ####
### Version 2.0: 30 November 2016                                           ####
### Brad Theilman                                                           ####
################################################################################

import os
import subprocess
import time
import glob
import pickle
import logging
import datetime

import numpy as np
import h5py

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

    skip = subwin_len - noverlap
    dur = segment[1] - segment[0]
    A = (int(t) - int(segment[0])) / int(skip)
    J = int(subwin_len-1) / int(skip)
    max_k = int(np.floor(float(dur)/float(skip)))

    wins = []
    i0 = A
    wins.append(i0)
    wins = i0 - np.array(range(J+1))
    wins = wins[wins >=0]
    wins = wins[wins < max_k]
    return wins

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
    max_k = int(np.floor(float(dur)/float(skip)))
    starts = [segment[0] + k*skip for k in range(max_k)]
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

def old_get_segment(trial_bounds, fs, segment_info):
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
        'segend' : time in ms of segment end relative to trial end

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
        seg_end = trial_bounds[1] \
                  + np.floor(segment_info['segend']*(fs/1000.))
    return [seg_start, seg_end]

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
    Calculate bettis, matrix form
    '''
    cell_groups = calc_cell_groups(data_mat, clusters, thresh)
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

def get_betti_savefile(aid, apath, stim):
    bs = aid + '-stim-{}'.format(stim) + '-betti.csv'
    bs = os.path.join(apath, bs)
    return bs

def get_bps(aid, apath, stim):
    bps = aid + '-stim-{}'.format(stim) + '-bettiPersistence.pkl'
    bps = os.path.join(apath, bps)
    return bps

def get_pfile_stem(aid, apath, stim):
    pfile_stem = aid + '-stim-{}'.format(stim)
    pfile_stem = os.path.join(apath, pfile_stem)
    return pfile_stem

def get_analysis_paths(aid, apath, stim):
    ###  Prepare destination file paths
    bs = get_betti_savefile(aid, apath, stim)

    bps = get_bps(aid, apath, stim)

    pfs = get_pfile_stem(aid, apath, stim)
    return (bs, bps, pfs)

def get_pfile_name(pfile_stem, **kwargs):
    for key, val in kwargs.iteritems():
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

def calc_CI_bettis_tensor(analysis_id, binned_data_file,
                          block_path, thresh, shuffle=False, nperms=0,
                          ncellsperm=1, swl=None):
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
    TOPOLOGY_LOG.info('Starting calc_CI_bettis_tensor')
    TOPOLOGY_LOG.info('analysis_id: {}'.format(analysis_id))
    bdf_name, ext = os.path.splitext(os.path.basename(binned_data_file))
    analysis_path = os.path.join(block_path,
                                 'topology/{}/'.format(analysis_id))
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)

    TOPOLOGY_LOG.info('bdf_name: {}'.format(bdf_name))
    TOPOLOGY_LOG.info('analysis_path: {}'.format(analysis_path))
    TOPOLOGY_LOG.info('Theshold: {}'.format(thresh))

    if shuffle:
        analysis_id = analysis_id +'-shuffle-'
    if nperms:
        analysis_id = analysis_id + '-permuted{}-'.format(nperms)

    with h5py.File(binned_data_file, 'r') as bdf:
        stims = bdf.keys()
        bpd_withstim = dict()
        for stim in stims:
            TOPOLOGY_LOG.info('Calculating bettis for stim: {}'.format(stim))
            stim_trials = bdf[stim]
            (bs, bps, pfs) = get_analysis_paths(analysis_id,
                                                analysis_path,
                                                stim)
            TOPOLOGY_LOG.info('Betti savefile: {}'.format(bs))
            TOPOLOGY_LOG.info('Betti persistence \
                               savefile: {}'.format(bps))
            bpd = dict()
            ### Compute Bettis
            if swl:
                bpd = do_compute_betti_sliding_window(stim_trials,
                                                      pfs,
                                                      thresh, shuffle, nperms,
                                                      ncellsperm,
                                                      sliding_window_length=swl)
            else:
                bpd = do_compute_betti(stim_trials,
                                       pfs,
                                       thresh, shuffle, nperms, ncellsperm)
            bpd_withstim[stim] = bpd
            with open(bps, 'w') as bpfile:
                pickle.dump(bpd, bpfile)
        bpdws_sfn = os.path.join(analysis_path,
                                 analysis_id+'-bettiResultsDict.pkl')
        with open(bpdws_sfn, 'w') as bpdwsfile:
            pickle.dump(bpd_withstim, bpdwsfile)
        TOPOLOGY_LOG.info('Completed All Stimuli')
        return bpdws_sfn

def do_compute_betti(stim_trials, pfile_stem, thresh,
                     shuffle, nperms, ncellsperm):

    assert 'pop_tens' in stim_trials.keys(), 'No Data Tensor!!'
    data_tensor = np.array(stim_trials['pop_tens'])
    clusters = np.array(stim_trials['clusters'])
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
            new_tensor = get_perms(data_mat, nperms, ncellsperm)
            for perm in range(nperms):
                pfile = get_pfile_name(pfile_stem, rep=trial, perm=perm)
                nmat = new_tensor[:, :, perm]
                if shuffle:
                    nmat = get_shuffle(nmat)
                bettis = calc_bettis(nmat, clusters, pfile, thresh)
                bettipermdict[str(perm)] = {'bettis': bettis}
            bettidict[str(trial)] = bettipermdict
        else:
            if shuffle:
                data_mat = get_shuffle(data_mat)
                pfile = get_pfile_name(pfile_stem, rep=trial, shuffled=1)
            bettis = calc_bettis(data_mat, clusters, pfile, thresh)
            bettidict[str(trial)] = {'bettis': bettis}
    return bettidict

def do_compute_betti_sliding_window(stim_trials, pfile_stem,
                                    thresh, shuffle, nperms, ncellsperm,
                                    sliding_window_length=10):
    ''' sliding window length is given in # of bins. '''

    assert 'pop_tens' in stim_trials.keys(), 'No Data Tensor!!'
    data_tensor = np.array(stim_trials['pop_tens'])
    clusters = np.array(stim_trials['clusters'])
    (ncells, nwin, ntrials) = (data_tensor.shape)
    bettidict = {}

    # Compute number of sliding windows
    nslide = nwin - sliding_window_length
    for trial in range(ntrials):
        bettitrial = {}
        for slide in range(nslide):
            pfile = get_pfile_name(pfile_stem, rep=trial, slide=slide)
            data_mat = data_tensor[:, slide:slide+sliding_window_length, trial]
            if nperms:
                bettipermdict = {}
                new_tensor = get_perms(data_mat, nperms, ncellsperm)
                for perm in range(nperms):
                    pfile = get_pfile_name(pfile_stem, rep=trial,
                                           slide=slide, perm=perm)
                    nmat = new_tensor[:, :, perm]
                    if shuffle:
                        nmat = get_shuffle(nmat)
                    bettis = calc_bettis(data_mat, clusters, pfile, thresh)
                    bettipermdict[str(perm)] = {'bettis': bettis}
                bettitrial[str(slide)] = bettipermdict
            else:
                if shuffle:
                    data_mat = get_shuffle(data_mat)
                    pfile = get_pfile_name(pfile_stem, rep=trial,
                                           slide=slide, shuffled=1)
                bettis = calc_bettis(data_mat, clusters, pfile, thresh)
                bettitrial[str(slide)] = {'bettis': bettis}
        bettidict[str(trial)] = bettitrial
    return bettidict

def get_shuffle(data_mat):

    (cells, wins) = data_mat.shape
    for cell in range(cells):
        np.random.shuffle(data_mat[cell, :])
    return data_mat

def get_perms(data_mat, nperms, ncellsperm):

    (cells, wins) = data_mat.shape
    new_tensor = np.zeros((ncellsperm, wins, nperms))
    for perm in range(nperms):
        celllist = np.random.permutation(cells)[:ncellsperm]
        new_tensor[:, :, perm] = data_mat[celllist, :]
    return new_tensor

def do_compute_betti_shuffle(stim_trials, pfile_stem,
                             thresh, nshuffs=1):

    assert 'pop_tens' in stim_trials.keys(), 'No Data Tensor!!'
    data_tensor = np.array(stim_trials['pop_tens'])
    clusters = np.array(stim_trials['clusters'])
    levels = (data_tensor.shape)[2:] # First two axes are cells, windows.
    assert len(levels) == 1, 'Cant handle more than one level yet'
    bettidict = {}
    for trial in range(levels[0]):
        pfile = pfile_stem + '-rep%d-shuffle-simplex.txt' % trial
        data_mat = data_tensor[:, :, trial] # cell, window
        data_mat = (np.random.permute(data_mat.T)).T
        bettis = calc_bettis(data_mat, clusters, pfile, thresh)
        bettidict[str(trial)] = {'bettis': bettis}
    return bettidict

def do_compute_betti_multilevel(stim_trials, pfile_stem, thresh):

    assert 'pop_tens' in stim_trials.keys(), 'No Data Tensor!!'
    data_tensor = np.array(stim_trials['pop_tens'])
    clusters = np.array(stim_trials['clusters'])
    dts = data_tensor.shape
    levels = (dts)[2:] # First two axes are cells, windows.
    nlen = np.product(dts[2:])
    reshaped_data = np.reshape(data_tensor, (dts[0], dts[1], nlen))
    bettidict = {}
    for trial in range(nlen):
        ids = lin2ind(levels, trial) # ids is [trial, perm, perm, perm, ...]
        pfile = pfile_stem + ''.join(['-lev%d' % s for s in ids]) \
                + '-simplex.txt'
        data_mat = reshaped_data[:, :, trial]
        bettis = calc_bettis(data_mat, clusters, pfile, thresh)
        bettidict[str(trial)] = {'bettis': bettis, 'indices': ids}
    return bettidict

def compute_topology_all_trials(data_tensor, clusters, pfile_stem,
                                thresh, remove_duplicates=False):
    ''' Combine all trials into one trial
        Compute total topology
    '''
    (ncells, nwin, ntrials) = data_tensor.shape
    rs = nwin*ntrials
    pfile = pfile_stem + '-at-simplex.txt'
    data_mat = np.reshape(data_tensor, (ncells, rs))
    if remove_duplicates:
        lex_ind = np.lexsort(data_mat)
        data_mat = data_mat[:, lex_ind]
        diff = np.diff(data_mat, axis=1)
        ui = np.ones(len(data_mat.T), 'bool')
        ui[1:] = (diff != 0).any(axis=0)
        data_mat = data_mat[:, ui]
    # Note persistence is meaningless here...
    pbettis = calc_bettis(data_mat, clusters, pfile, thresh)
    bettis = pbettis[-1] # need last one
    bpd = {'bettis': bettis}
    return bpd

def calc_CI_bettis_all_trials(analysis_id, binned_data_file,
                              block_path, thresh):
    '''
    Given a binned data file, compute the betti numbers of the Curto-Itskov
    Collapses across all trials for a given stimulus

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
    TOPOLOGY_LOG.info('Starting calc_CI_bettis_tensor')
    TOPOLOGY_LOG.info('analysis_id: {}'.format(analysis_id))
    bdf_name, ext = os.path.splitext(os.path.basename(binned_data_file))
    analysis_path = os.path.join(block_path,
                                 'topology/{}/'.format(analysis_id))
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)

    TOPOLOGY_LOG.info('bdf_name: {}'.format(bdf_name))
    TOPOLOGY_LOG.info('analysis_path: {}'.format(analysis_path))
    TOPOLOGY_LOG.info('Theshold: {}'.format(thresh))

    with h5py.File(binned_data_file, 'r') as bdf:
        stims = bdf.keys()
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
                  + '-ATbettiPersistence.pkl'
            betti_persistence_savefile = os.path.join(analysis_path, bps)
            TOPOLOGY_LOG.info('Betti persistence \
                               savefile: {}'.format(betti_persistence_savefile))
            bpd = dict()
            pfile_stem = analysis_id \
                         + '-stim-{}'.format(stim)
            pfile_stem = os.path.join(analysis_path, pfile_stem)
            ### Compute Bettis
            data_tens = np.array(stim_trials['pop_tens'])
            clusters = np.array(stim_trials['clusters'])
            bpd = compute_topology_all_trials(data_tens,
                                   	         clusters,
                                                 pfile_stem,
                                                 thresh)
            bpd_withstim[stim] = bpd
            with open(betti_persistence_savefile, 'w') as bpfile:
                pickle.dump(bpd, bpfile)
        bpdws_sfn = os.path.join(analysis_path,
                                 analysis_id+'-AATrial-bettiResultsDict.pkl')
        with open(bpdws_sfn, 'w') as bpdwsfile:
            pickle.dump(bpd_withstim, bpdwsfile)
        TOPOLOGY_LOG.info('Completed All Stimuli')
        return bpdws_sfn

def concatenate_trials(data_tensor):

    (ncells, nwin, ntrials) = data_tensor.shape
    rs = nwin*ntrials
    data_mat = np.reshape(data_tensor, (ncells, rs))
    return data_mat

def compute_total_topology(analysis_id, binned_data_file,
                           block_path, thresh):
    '''
    Concatenates all trials across all stimuli and computes topology
    removes redundant cell group columns
    '''
    TOPOLOGY_LOG.info('Starting compute_total_topology')
    TOPOLOGY_LOG.info('analysis_id: {}'.format(analysis_id))
    bdf_name, ext = os.path.splitext(os.path.basename(binned_data_file))
    analysis_path = os.path.join(block_path,
                                 'topology/{}/'.format(analysis_id))
    if not os.path.exists(analysis_path):
        os.makedirs(analysis_path)

    TOPOLOGY_LOG.info('bdf_name: {}'.format(bdf_name))
    TOPOLOGY_LOG.info('analysis_path: {}'.format(analysis_path))
    TOPOLOGY_LOG.info('Theshold: {}'.format(thresh))

    bpd = dict()
    pfile_stem = analysis_id \
                 + '-TotalTopology'
    pfile_stem = os.path.join(analysis_path, pfile_stem)

    with h5py.File(binned_data_file, 'r') as bdf:
        stims = bdf.keys()
        nstims = len(stims)
        bpd_withstim = dict()
        stim_mat_list = []
        for stim in stims:
            TOPOLOGY_LOG.info('Calculating bettis for stim: {}'.format(stim))
            stim_trials = bdf[stim]
            ### Compute Bettis
            data_tens = np.array(stim_trials['pop_tens'])
            clusters = np.array(stim_trials['clusters'])
            stim_mat = concatenate_trials(data_tens)
            stim_mat_list.append(stim_mat)

        (ncell, nwin) = stim_mat_list[0].shape
        stimtens = np.zeros((ncell, nwin, nstims))
        for ind, stim_mat in enumerate(stim_mat_list):
            stimtens[:, :, ind] = stim_mat

        bpd = compute_topology_all_trials(stimtens, clusters, pfile_stem,
                                             thresh, remove_duplicates=True)

        bpdws_sfn = os.path.join(analysis_path,
                            analysis_id+'-TotalTopology-bettiResultsDict.pkl')
        with open(bpdws_sfn, 'w') as bpdwsfile:
            pickle.dump(bpd, bpdwsfile)
        TOPOLOGY_LOG.info('Completed All Stimuli')
        return bpdws_sfn

###############################
###### Binning Functions ######
###############################

def build_binned_file(spikes, trials, clusters, win_size, fs,
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
            if str(stim) == 'nan':
                continue
            stimgrp = popvec_f.create_group(stim)
            stim_trials = trials[trials['stimulus'] == stim]
            nreps = len(stim_trials.index)
            stim_recs = stim_trials['recording'].values
            trial_len = (stim_trials['stimulus_end'] \
                         - stim_trials['time_samples']).unique()[0]
            gen_windows = compute_gen_windows(trial_len, fs, segment_info,
                                              win_size, dt_overlap)
            # Create Data set
            poptens = build_activity_tensor(stim_trials, spikes, clusters_list,
                                            gen_windows, win_size)
            poptens_dset = stimgrp.create_dataset('pop_tens', data=poptens)
            stimgrp.create_dataset('clusters', data=clusters_list)
            stimgrp.create_dataset('windows', data=np.array(gen_windows))
            poptens_dset.attrs['fs'] = fs
            poptens_dset.attrs['win_size'] = win_size

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
            if str(stim) == 'nan':
                continue
            stimgrp = popvec_f.create_group(stim)
            stim_trials = trials[trials['stimulus'] == stim]
            nreps = len(stim_trials.index)
            stim_recs = stim_trials['recording'].values
            trial_len = (stim_trials['stimulus_end'] \
                         - stim_trials['time_samples']).unique()[0]

            # compute subwin len and noverlap in samples
            subwin_len = int(np.round(win_size/1000. * fs))
            noverlap = int(np.round(dt_overlap/1000. * fs))
            segment = get_segment([0, trial_len], fs, segment_info)
            # Create Data set
            poptens = build_activity_tensor_quick(stim_trials, spikes, clusters_list, nclus,
                                win_size, subwin_len, noverlap, segment)
            poptens_dset = stimgrp.create_dataset('pop_tens', data=poptens)
            stimgrp.create_dataset('clusters', data=clusters_list)
            #stimgrp.create_dataset('windows', data=np.array(gen_windows))
            poptens_dset.attrs['fs'] = fs
            poptens_dset.attrs['win_size'] = win_size

def compute_gen_windows(trial_len, fs, segment_info, win_size, dt_overlap):
        # Compute generic windows for this stimulus.
        #This assumes stimulus for all trials is same length
        # In order to avoid recomputing windows for each trial
        # trial_len is in samples 
        gen_seg_start, gen_seg_end = get_segment([0, trial_len],
                                                 fs,
                                                 segment_info)
        gen_seg = [gen_seg_start, gen_seg_end]
        win_size_samples = int(np.round(win_size/1000. * fs))
        overlap_samples = int(np.round(dt_overlap/1000. * fs))
        gen_windows = create_subwindows(gen_seg,
                                        win_size_samples,
                                        overlap_samples)
        return gen_windows

def build_activity_tensor(stim_trials, spikes, clusters_list,
                          gen_windows, win_size):

    nreps = len(stim_trials.index)
    stim_recs = stim_trials['recording'].values
    nwins = len(gen_windows)
    nclus = len(clusters_list)
    poptens = np.zeros((nclus, nwins, nreps))
    for rep in range(nreps):
        trial_start = stim_trials.iloc[rep]['time_samples']
        rec = stim_recs[rep]
        for win_ind, win in enumerate(gen_windows):
            win2 = win2 = [win[0] + trial_start, win[1] + trial_start]
            spikes_in_win = get_spikes_in_window(spikes, win2, rec)
            spiking_clusters = spikes_in_win['cluster'].unique()
            if len(spiking_clusters) > 0:
                for clu in spiking_clusters:
                    cluster_mask = (spikes_in_win['cluster'] == clu)
                    pvclu_msk = (clusters_list == clu)
                    nsp_clu = float(len(spikes_in_win[cluster_mask]))
                    nsp_clu2 = 1000.*nsp_clu/win_size
                    poptens[pvclu_msk, win_ind, rep] = nsp_clu2 
    return poptens

def build_activity_tensor_quick(stim_trials, spikes, clusters_list, nclus,
                                win_size, subwin_len, noverlap, segment):

    nreps = len(stim_trials.index)
    stim_recs = stim_trials['recording'].values
    
    skip = subwin_len - noverlap
    dur = segment[1] - segment[0]
    nwins = int(np.floor(float(dur)/float(skip)))
    poptens = np.zeros((nclus, nwins, nreps))
    for rep in range(nreps):
        trial_start = stim_trials.iloc[rep]['time_samples']
        trial_end = stim_trials.iloc[rep]['stimulus_end']
        samp_period = (trial_start + segment[0], trial_start + segment[1])
        rec = stim_recs[rep]
        stim_rec_spikes = spikes[spikes['recording'] == rec]
        sptimes = stim_rec_spikes['time_samples'].values
        clusters = stim_rec_spikes['cluster'].values
        for sp, clu in zip(sptimes, clusters):
            wins = get_windows_for_spike(sp, subwin_len, noverlap, samp_period)
            poptens[clusters_list==clu, wins, rep] += 1
    poptens /= (win_size/1000.0)
    return poptens 


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
    ncells, nwin, ntrial = data_tens.shape
    ptens = np.zeros((ncellsperm, nwin, ntrial, nperms))
    clumapmat = np.zeros((ncellsperm, nperms))

    for perm in range(nperms):
        permt = np.random.permutation(ncells)
        if ncells >= ncellsperm:
            permt = permt[0:ncellsperm].tolist()
        else:
            permt = permt.tolist()
        ptens[:, :, :, perm] = data_tens[permt, :, :]
        clumapmat[:, perm] = clusters[permt]
    return (ptens, clumapmat)

def build_permuted_binned_file(bf, pdf, ncp, nperms):
    '''
    Builds a permuted Binned data file

    Parameters
    ----------
    bf : str
        Path to original binned data file
    pdf : str
        path to new permuted data file
    ncp : int
        number of cells per permutation
    nperms : int
        number of permutations
    '''
    with h5py.File(bf, "r") as poptens_f:
        winsize = poptens_f.attrs['win_size']
        fs = poptens_f.attrs['fs']
        nclus = poptens_f.attrs['nclus']
        stims = poptens_f.keys()
        with h5py.File(pdf, "w") as perm_f:
            perm_f.attrs['win_size'] = winsize
            perm_f.attrs['fs'] = fs
            perm_f.attrs['nclus'] = nclus
            for stim in stims:
                perm_stimgrp = perm_f.create_group(stim)
                data_tens = np.array(poptens_f[stim]['pop_tens'])
                clusters = np.array(poptens_f[stim]['clusters'])
                windows = np.array(poptens_f[stim]['windows'])
                (ptens, clumapmat) = build_permuted_data_tensor(data_tens,
                                                                clusters,
                                                                ncp, nperms)
                perm_stimgrp.create_dataset('pop_tens', data=ptens)
                perm_stimgrp.create_dataset('windows', data=windows)
                perm_stimgrp.create_dataset('clusters', data=clusters)
                perm_stimgrp.create_dataset('clumapmat', data=clumapmat)

def permute_binned(binned_file, ncellsperm, nperms):
    ''' Takes a path to a binned file
        and produces a permuted version
    '''
    binned_file = os.path.abspath(binned_file)
    bdf_fold, bdf_full_name = os.path.split(binned_file)
    bdf_name, bdf_ext = os.path.splitext(bdf_full_name)
    permuted_binned_folder = os.path.join(bdf_fold, 'permuted_binned')
    if not os.path.exists(permuted_binned_folder):
        os.makedirs(permuted_binned_folder)
    pbd_name = bdf_name + '-permuted.binned'
    permuted_data_file = os.path.join(permuted_binned_folder, pbd_name)
    build_permuted_binned_file(binned_file, permuted_data_file,
                               ncellsperm, nperms)
    return permuted_binned_folder

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

def dag_bin(block_path, winsize, segment_info, **kwargs):

    (spikes, trials, clusters, fs) = db_load_data(block_path)
    bfdict = do_dag_bin_lazy(block_path, spikes, trials, clusters,
                             fs, winsize, segment_info, **kwargs)
    return bfdict

def do_dag_bin(block_path, spikes, trials, clusters, fs, winsize, segment_info,
               cluster_group=['Good'], dt_overlap=0.0):

    block_path = os.path.abspath(block_path)
    # Create directories and filenames
    analysis_id = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    raw_binned_fname = analysis_id + '-{}-{}.binned'.format(winsize, dt_overlap)
    analysis_id_forward = analysis_id + '-{}-{}'.format(winsize, dt_overlap)
    bfdict = {'analysis_id': analysis_id_forward}

    binned_folder = os.path.join(block_path,
                                 'binned_data/{}/'.format(analysis_id))
    if not os.path.exists(binned_folder):
        os.makedirs(binned_folder)

    # Bin the raw data
    raw_binned_f = os.path.join(binned_folder, raw_binned_fname)
    TOPOLOGY_LOG.info('Binning data')
    build_binned_file(spikes, trials, clusters, winsize, fs,
                                      cluster_group, segment_info, raw_binned_f,
                                      dt_overlap)
    bfdict['raw'] = binned_folder
    return bfdict

def do_dag_bin_lazy(block_path, spikes, trials, clusters, fs, winsize,
                    segment_info, cluster_group=['Good'], dt_overlap=0.0,
                    comment=''):
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

    block_path = os.path.abspath(block_path)
    # Create directories and filenames
    analysis_id = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    raw_binned_fname = analysis_id + '-{}-{}.binned'.format(winsize, dt_overlap)
    analysis_id_forward = analysis_id + '-{}-{}'.format(winsize, dt_overlap)
    bfdict = {'analysis_id': analysis_id_forward}

    cg_string = '-'.join(cluster_group)
    seg_string = '-'.join(map(str, segment_info))
    if comment:
        seg_string = seg_string+ '-'+comment
    bin_string = 'binned_data/win-{}_dtovr-{}_cg-{}_seg-{}/'.format(winsize,
                                                                    dt_overlap,
                                                                    cg_string,
                                                                    seg_string)
    binned_folder = os.path.join(block_path, bin_string)
    if not os.path.exists(binned_folder):
        os.makedirs(binned_folder)
    existing_binned = glob.glob(os.path.join(binned_folder, '*.binned'))

    if len(existing_binned) == 0:
        # not already binned!
        # Bin the raw data
        TOPOLOGY_LOG.info('Data not already binned.')
        raw_binned_f = os.path.join(binned_folder, raw_binned_fname)
        TOPOLOGY_LOG.info('Binning data')
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
            TOPOLOGY_LOG.info('Computing topology for: %s' % rdf)
            res_f = calc_CI_bettis_tensor(tpid_raw, rdf, block_path,
                                      thresh, **kwargs)
        with open(res_f, 'r') as f:
            res = pickle.load(f)
            analysis_dict['raw'] = res

    if shuffle:
        tpid_raw = aid +'-{}-raw-shuffled'.format(thresh)
        raw_data_files = glob.glob(os.path.join(raw_folder, '*.binned'))
        for rdf in raw_data_files:
            TOPOLOGY_LOG.info('Computing shuffled topology for: %s' % rdf)
            res_f = calc_CI_bettis_tensor(tpid_raw, rdf, block_path, thresh,
                                      shuffle=True, **kwargs)
        with open(res_f, 'r') as f:
            res = pickle.load(f)
            analysis_dict['rawshuffled'] = res

    if nperms:
        tpid_raw = aid +'-{}-permuted-{}-{}'.format(thresh, nperms, ncellsperm)
        raw_data_files = glob.glob(os.path.join(raw_folder, '*.binned'))
        for rdf in raw_data_files:
            TOPOLOGY_LOG.info('Computing topology for: %s' % rdf)
            res_f = calc_CI_bettis_tensor(tpid_raw, rdf, block_path, thresh,
                                      nperms=nperms, ncellsperm=ncellsperm,
                                      **kwargs)
        with open(res_f, 'r') as f:
            res = pickle.load(f)
            analysis_dict['permuted'] = res

    if shuffleperm:
        tpid_raw = aid +'-{}-shuffled-permuted-{}-{}'.format(thresh, nperms,
                                                             ncellsperm)
        raw_data_files = glob.glob(os.path.join(raw_folder, '*.binned'))
        for rdf in raw_data_files:
            TOPOLOGY_LOG.info('Computing topology for: %s' % rdf)
            res_f = calc_CI_bettis_tensor(tpid_raw, rdf, block_path, thresh,
                                      shuffle=True, nperms=nperms,
                                      ncellsperm=ncellsperm, **kwargs)
        with open(res_f, 'r') as f:
            res = pickle.load(f)
            analysis_dict['shuffledpermuted'] = res

    if 'alltrials' in bfdict.keys():
        at_folder = bfdict['alltrials']
        tpid_at = aid + '-{}-at'.format(thresh)
        atDataFiles = glob.glob(os.path.join(at_folder, '*.binned'))
        for atdf in atDataFiles:
            TOPOLOGY_LOG.info('Computing topology for %s' % atdf)
            res_f = calc_CI_bettis_all_trials(tpid_at, atdf,
                                               block_path, thresh)
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
