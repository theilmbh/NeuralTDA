import numpy as np
import pandas as pd
import os, sys
import subprocess
import pickle
import h5py
import time
import glob

from ephys import events, core

global alogf

def topology_log(logfile, log_str):
    with open(logfile, 'a+') as lf:
        log_line = str(time.time()) + ' ' + log_str + '\n'
        lf.write(log_line)

def get_spikes_in_window(spikes, window):
    '''
    Returns a spike DataFrame containing all spikes within a given window

    Parameters
    ------
    spikes : pandas DataFrame
        Contains the spike data (see core)
    window : tuple
        The lower and upper bounds of the time window for extracting spikes 

    Returns
    ------
    spikes_in_window : pandas DataFrame
        DataFrame with same layout as input spikes but containing only spikes 
        within window 
    '''
    mask = ((spikes['time_samples']<=window[1]) & 
            (spikes['time_samples']>=window[0]))
    return spikes[mask]


def mean_fr_decorator(mean_fr_func):

    def decorated(cluster_row, *args, **kwargs):
        print(cluster_row)
        try:
            int(cluster_row)
            mean_fr = mean_fr_func(cluster_row, *args, **kwargs)
            print(pd.DataFrame({'mean_fr': mean_fr}))
            return pd.DataFrame({'mean_fr': mean_fr})
        except ValueError:
            mean_fr = mean_fr_func(cluster_row['cluster'], *args, **kwargs)
            print(pd.DataFrame({'mean_fr': mean_fr}))
            return pd.DataFrame({'mean_fr': mean_fr})

    return decorated


def calc_mean_fr(cluster_row, spikes, window):
    '''
    Computes the mean firing rate of a unit within the given spikes DataFrame

    Parameters
    ------
    row : pandas dataframe row
        row of cluster dataframe containing clusterID to compute
    spikes : pandas dataframe 
        Contains the spike data.  
        Firing rate computed from this data over the window
    window : tuple
        time (samples) over which the spikes in 'spikes' occur.

    Returns
    ------
    mean_fr : float 
        Mean firing rate over the interval
    '''
    cluster = cluster_row['cluster']
    spikes = spikes[spikes['cluster']==cluster]
    spikes = get_spikes_in_window(spikes, window)
    nspikes = len(spikes.index)
    dt = window[1] - window[0]
    mean_fr = (1.0*nspikes) / dt
    retframe = pd.DataFrame({'mean_fr': [mean_fr]})
    return retframe.iloc[0]


def calc_mean_fr_int(cluster, spikes, window):
    ''' Does the same as above, but for ints.  This is bad.  Fix this
    '''
    spikes = spikes[spikes['cluster']==cluster]
    spikes = get_spikes_in_window(spikes, window)
    nspikes = len(spikes.index)
    dt = window[1] - window[0]
    mean_fr = (1.0*nspikes) / dt
    return mean_fr  


def create_subwindows(segment, subwin_len, n_subwin_starts):
    ''' Create list of subwindows for cell group identification 

    Parameters
    ------
    segment : list
        Beginning and end of the segment to subdivide into windows
    subwin_len : int
        number of samples to include in a subwindows
    n_subwin_starts : int
        number of shifts of the subwindows

    Returns
    ------
    subwindows : list 
        list of subwindows
    '''
    
    starts_dt = np.floor(subwin_len / n_subwin_starts)
    starts = np.floor(np.linspace(segment[0], segment[0]+subwin_len, n_subwin_starts))
    subwindows = []
    for start in starts:
        subwin_front = np.arange(start, segment[1], subwin_len)
        for front in subwin_front:
            subwin_end = front + subwin_len
            subwindows.append([front, subwin_end])

    return subwindows


def calc_population_vectors(spikes, clusters, windows, thresh):
    '''
    Builds population vectors according to Curto and Itskov 2008

    Parameters
    ------
    spikes : pandas DataFrame
        DataFrame containing spike data. 
        Must have 'fr_mean' column containing mean firing rate
    clusters : pandas DataFrame
        Dataframe containing cluster information
    windows : tuple
        The set of windows to compute population vectors for 
    thresh : float
        how many times above the mean the firing rate needs to be

    Returns 
    ------
    popvec_list : list
        population vector list.  
        Each element is a list containing the window and the population vector.
        The population vector is an array containing cluster ID and firing rate. 
    '''
    print('Building population vectors...')
    total_win = len(windows)
    popvec_list = []
    for win_num, win in enumerate(windows):
        if np.mod(win_num, 500)==0:
            print("Window {} of {}".format(str(win_num), str(total_win)))
            sys.stdout.flush()
        popvec = np.zeros([len(clusters.index), 3])
        for ind, cluster in enumerate(clusters['cluster'].values):
            fr = calc_mean_fr_int(cluster, spikes, win)
            popvec[ind, 1] = fr
            popvec[ind, 0] = cluster
            popvec[ind, 2] = fr>(1.0*thresh*clusters[
                                clusters['cluster']==cluster]['mean_fr']).values
        popvec_list.append([win, popvec])
    return popvec_list


DEFAULT_CG_PARAMS = {'cluster_group': None, 'subwin_len': 100, 'threshold': 6.0,
                     'n_subwin': 5}
def calc_cell_groups(spikes, segment, clusters, cg_params=DEFAULT_CG_PARAMS):
    '''
    Creates cell group dataframe according to Curto and Itskov 2008

    Parameters
    ------
    spikes : pandas dataframe
        Dataframe containing the spikes to analyze
    segment : tuple or list of floats
        time window for which to create cell groups 
    clusters : pandas DataFrame
        dataframe containing cluster information
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
    ------
    cell_groups : list
        list where each entry is a list containing a time window 
        and the ID's of the cells in that group
    '''

    print('Creating cellgroups...')
    cluster_group = cg_params['cluster_group']
    subwin_len    = cg_params['subwin_len']
    threshold     = cg_params['threshold']
    n_subwin      = cg_params['n_subwin']

    spikes = get_spikes_in_window(spikes, segment)
    if cluster_group != None:
        mask = np.ones(len(clusters.index)) < 0
        for grp in cluster_group:
            mask = np.logical_or(mask, clusters['quality'] == grp)
        clusters = clusters[mask]
        spikes = spikes[spikes['cluster'].isin(clusters['cluster'].values)]

    topology_subwindows = create_subwindows(segment, subwin_len, n_subwin)

    clusters['mean_fr'] = clusters.apply(lambda row: calc_mean_fr(row,
                                         spikes,segment)['mean_fr'], axis=1)
    # Build population vectors
    population_vector_list = calc_population_vectors(spikes, clusters, 
                                                     topology_subwindows, 
                                                     threshold)
    cell_groups = []
    for population_vector_win in population_vector_list:
        win          = population_vector_win[0]
        popvec       = population_vector_win[1]
        active_cells = popvec[(popvec[:, 2].astype(bool)), 0].astype(int)
        cell_groups.append([win, active_cells])
        
    return cell_groups

def build_perseus_input(cell_groups, savefile):
    ''' 
    Formats cell group information as input 
    for perseus persistent homology software

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
    print('Building Perseus input...')
    with open(savefile, 'w+') as pfile:
        #write num coords per vertex
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
            out_str  = str(grp_dim) + ' ' + vert_str + ' 1\n'
            #debug_print('Writing: %s' % out_str)
            pfile.write(out_str)

    return savefile

def build_perseus_persistent_input(cell_groups, savefile):
    '''
    Formats cell group information as input 
    for perseus persistent homology software, but assigns filtration
    levels for each cell group based on the time order of their appearance
    in the signal

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
    print('Building Perseus persistent input...')
    with open(savefile, 'w+') as pfile:
        #write num coords per vertex
        pfile.write('1\n')
        for ind, win_grp in enumerate(cell_groups):
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
            out_str  = str(grp_dim) + ' ' + vert_str + ' {}\n'.format(str(ind+1))
            #debug_print('Writing: %s' % out_str)
            pfile.write(out_str)

    return savefile

def run_perseus(pfile):
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
    global alogf
    print('Running Perseus...')
    of_string, ext = os.path.splitext(pfile)
    perseus_command = "/home/btheilma/bin/perseus" 
    perseus_args = "nmfsimtop {} {}".format(pfile, of_string)

    perseus_return_code = subprocess.call([perseus_command, 'nmfsimtop', pfile, 
                                           of_string])
    topology_log(alogf, 'perseus return code: {}'.format(perseus_return_code))
    betti_file = of_string+'_betti.txt'
    betti_file = os.path.join(os.path.split(pfile)[0], betti_file)
    return betti_file

def calc_bettis(spikes, segment, clusters, pfile, cg_params=DEFAULT_CG_PARAMS, persistence=False):
    ''' Calculate betti numbers for spike data in segment

    Parameters
    ------
    spikes : pandas DataFrame
        dataframe containing spike data
    segment : list
        time window of data to calculate betti numbers
    clusters : pandas Dataframe
        dataframe containing cluster data
    pfile : str
        file in which to save simplex data
    cg_params : dict
        Parameters for CG generation
    persistence : bool, optional 
        If true, will compute the time dependence of the bettis 
        by including filtration times in the cell groups

    Returns
    ------
    bettis : list
        betti numbers.  Each member is a list with the first member being 
        filtration time and the second being betti numbers
    '''
    print('In calc_bettis')
    topology_log(alogf, 'In calc_bettis...')
    cell_groups = calc_cell_groups(spikes, segment, clusters, cg_params)

    if persistence:
        build_perseus_persistent_input(cell_groups, pfile)
    else:
        build_perseus_input(cell_groups, pfile)

    betti_file = run_perseus(pfile)
    bettis = []
    with open(betti_file, 'r') as bf:
        for bf_line in bf:
            if len(bf_line)<2:
                continue
            betti_data      = bf_line.split()
            nbetti          = len(betti_data)-1
            filtration_time = int(betti_data[0])
            betti_numbers   = map(int, betti_data[1:])
            bettis.append([filtration_time, betti_numbers])
    return bettis


DEFAULT_SEGMENT_INFO = {'period': 1}
def get_segment(trial_bounds, fs, segment_info):
    '''
    Use segment info to determine segment to compute topology for 

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
        bounds for the segment to compute topology for, in samples 

    '''

    if segment_info['period'] == 1:
        return trial_bounds
    else:
        seg_start = trial_bounds[0] + np.floor(segment_info['segstart']*(fs/1000.))
        seg_end = trial_bounds[0] + np.floor(segment_info['segend']*(fs/1000.))
    return [seg_start, seg_end]


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

    maxbetti      = 10
    kwikfile      = core.find_kwik(block_path)
    kwikname, ext = os.path.splitext(os.path.basename(kwikfile))

    topology_log(alogf, '****** Beginning Curto+Itskov Topological Analysis of: {} ******'.format(kwikfile))

    spikes   = core.load_spikes(block_path)
    clusters = core.load_clusters(block_path)
    trials   = events.load_trials(block_path)
    fs       = core.load_fs(block_path)

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
        stim_trials = trials[trials['stimulus']==stim]
        nreps       = len(stim_trials.index)
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
            trial_end   = stim_trials.iloc[rep]['stimulus_end']

            cg_params                   = DEFAULT_CG_PARAMS
            cg_params['subwin_len']     = windt_samps
            cg_params['cluster_group']  = cluster_group
            cg_params['n_subwin']       = n_subwin
            cg_params['threshold']      = threshold

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
            trial_bettis                         = bettis[-1][1]
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
        stim_trials = trials[trials['stimulus']==stim]
        nreps       = len(stim_trials.index)
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
            trial_end   = stim_trials.iloc[rep]['stimulus_end']

            cg_params                   = DEFAULT_CG_PARAMS
            cg_params['subwin_len']     = windt_samps
            cg_params['cluster_group']  = cluster_group
            cg_params['n_subwin']       = n_subwin

            segment = get_segment([trial_start, trial_end], fs, segment_info)
            print('Trial bounds: {}  {}'.format(str(trial_start), 
                                                str(trial_end)))
            print('Segment bounds: {}  {}'.format(str(segment[0]), 
                                                  str(segment[1])))
            
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

def spike_time_subtracter(row, trial_start, trial_end, first_trial_start):
    spiketime = row['time_samples']
    if (spiketime >= trial_start) and (spiketime <= trial_end):
        return spiketime - (trial_start - first_trial_start)
    else:
        return spiketime 


def calc_CI_bettis_on_dataset_average_activity(block_path, cluster_group=None, windt_ms=50., n_subwin=5,
                           segment_info=DEFAULT_SEGMENT_INFO, persistence=False):
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
    maxbetti      = 10
    kwikfile      = core.find_kwik(block_path)
    kwikname, ext = os.path.splitext(os.path.basename(kwikfile))

    spikes   = core.load_spikes(block_path)
    clusters = core.load_clusters(block_path)
    trials   = events.load_trials(block_path)
    fs       = core.load_fs(block_path)

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

def build_population_embedding(spikes, trials, clusters, win_size, fs, cluster_group, segment_info, popvec_fname):
    '''
    Embeds binned population activity into R^n
    Still need TODO?

    Parameters
    ------
    spikes : pandas dataframe 
        Spike frame from ephys.core 

    win_size : float
        window size in ms
    '''
    global alogf 
    with h5py.File(popvec_fname, "w") as popvec_f:

        popvec_f.attrs['win_size'] = win_size
        popvec_f.attrs['fs'] = fs 
        #f.attrs['cluster_group'] = cluster_group

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
        popvec_dict = {}

        for stim in stims:
            stimgrp = popvec_f.create_group(stim)
            stim_trials = trials[trials['stimulus']==stim]
            nreps       = len(stim_trials.index)

            for rep in range(nreps):
                trialgrp = stimgrp.create_group(str(rep))

                trial_start = stim_trials.iloc[rep]['time_samples']
                trial_end   = stim_trials.iloc[rep]['stimulus_end']
                seg_start, seg_end = get_segment([trial_start, trial_end], fs, segment_info)
                topology_log(alogf, "segments: {} {}".format(seg_start, seg_end))
                win_size_samples = np.round(win_size/1000. * fs)

                windows = create_subwindows([seg_start, seg_end], win_size_samples, 1)
                nwins = len(windows)
                popvec_dset_init = np.zeros((nclus, nwins))
                popvec_dset = trialgrp.create_dataset('pop_vec', data=popvec_dset_init)
                popvec_clu_dset = trialgrp.create_dataset('clusters', data=clusters_list)
                popvec_win_dset = trialgrp.create_dataset('windows', data=np.array(windows))
                popvec_dset.attrs['fs'] = fs
                popvec_dset.attrs['win_size'] = win_size

                for win_ind, win in enumerate(windows):
                # compute population activity vectors
                    spikes_in_win = get_spikes_in_window(spikes, win)
                    clus_that_spiked = spikes_in_win['cluster'].unique()
                
                    # find spikes from each cluster
                    if len(clus_that_spiked) > 0:
                        for clu in clus_that_spiked:
                            popvec_dset[(clusters_list == clu), win_ind] = float(len(spikes_in_win[spikes_in_win['cluster']==clu]))/(win_size/1000.)

def prep_and_bin_data(block_path, bin_def_file, bin_id, nshuffs):

    global alogf
    spikes   = core.load_spikes(block_path)
    clusters = core.load_clusters(block_path)
    trials   = events.load_trials(block_path)
    fs       = core.load_fs(block_path)

    kwikfile      = core.find_kwik(block_path)

    binning_folder = do_bin_data(block_path, spikes, clusters, trials, fs, kwikfile, bin_def_file, bin_id)
    if nshuffs:
        print('Making Shuffled Controls')
        make_shuffled_controls(binning_folder, nshuffs)


def do_bin_data(block_path, spikes, clusters, trials, fs, kwikfile, bin_def_file, bin_id):
    '''
    Bins the data using build_population_embedding 
    Parameters are given in bin_def_file
    Each line of bin_def_file contains the parameters for each binning

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
        Identifyer for this particular binning run 
    '''

    # Try to make a folder to store the binnings
    global alogf
    binning_folder = os.path.join(block_path, 'binned_data/{}'.format(bin_id))
    if not os.path.exists(binning_folder):
        os.makedirs(binning_folder)
    kwikname, ext = os.path.splitext(os.path.basename(kwikfile))
    alogf = os.path.join(binning_folder, 'binning.log')

    with open(bin_def_file, 'r') as bdf:
        for bdf_line in bdf:
            binning_params = bdf_line.split(' ')
            binning_id = binning_params[0]
            win_size = float(binning_params[1])
            cluster_groups = binning_params[2]
            segment = int(binning_params[3])
            topology_log(alogf, 'seg specifier: {}'.format(segment))
            seg_start = float(binning_params[4])
            seg_end = float(binning_params[5])
            segment_info = {'period': segment, 'segstart':seg_start, 'segend': seg_end}
            cluster_group = cluster_groups.split(',')
            binning_path = os.path.join(binning_folder, '{}-{}.binned'.format(kwikname, binning_id))
            if os.path.exists(binning_path):
                print('Binning file {} already exists, skipping..'.format(binning_path))
                continue
            print('Binning data into {}'.format('{}.binned'.format(binning_id)))
            build_population_embedding(spikes, trials, clusters, win_size, fs, cluster_group, segment_info, binning_path)
            print('Done')
    return binning_folder

def calc_cell_groups_from_binned_data(binned_dataset, thresh):

    global alogf
    bds = np.array(binned_dataset['pop_vec'])
    clusters = np.array(binned_dataset['clusters'])
    [clus, nwin] = bds.shape
    topology_log(alogf, 'Number of Clusters: {}'.format(clus))

    mean_frs = np.mean(bds, 1)
    cell_groups = []
    for win in range(nwin):
        acty = bds[:, win]
        above_thresh = np.greater(acty, thresh*mean_frs)
        clus_in_group = clusters[above_thresh]
        cell_groups.append([win, clus_in_group])
    return cell_groups

def calc_bettis_from_binned_data(binned_dataset, pfile, thresh):
    cell_groups = calc_cell_groups_from_binned_data(binned_dataset, thresh)

    build_perseus_persistent_input(cell_groups, pfile)

    betti_file = run_perseus(pfile)
    bettis = []
    with open(betti_file, 'r') as bf:
        for bf_line in bf:
            if len(bf_line)<2:
                continue
            betti_data      = bf_line.split()
            nbetti          = len(betti_data)-1
            filtration_time = int(betti_data[0])
            betti_numbers   = map(int, betti_data[1:])
            bettis.append([filtration_time, betti_numbers])
    return bettis

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
                            permt = permt[0:n_cells_in_perm]
                            clusters_to_save = clusters[permt]
                            popvec_save = popvec[permt]
                            perm_permgrp = perm_trialgrp.create_group(str(perm_num))
                            perm_permgrp.create_dataset('pop_vec', data=popvec_save)
                            perm_permgrp.create_dataset('clusters', data=clusters_to_save)
                            perm_permgrp.create_dataset('windows', data=windows)

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



