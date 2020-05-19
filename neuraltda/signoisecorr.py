import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

def get_average_firing_rate(spikes, cluster_id, t_start, t_end, fs):
    '''
    Compute the average firing rate of the cluster <cluster_id> during the time interval [t_start, t_end]
    Average firing rate is (# of spikes in interval) / (duration of interval)
    '''
    cluster_spikes_in_interval = spikes[(spikes['cluster'] == cluster_id) &
                            (spikes['time_samples'] < t_end) & 
                            (spikes['time_samples'] > t_start)
    ]
    n_spikes = cluster_spikes_in_interval.size
    dt = (t_end - t_start) / fs
    return n_spikes / dt

def get_stimulus_average_firing_rate(spikes, trials, cluster_id, stimulus_id, fs):
    '''
    Return the average firing rate of a neuron over all trials from a given stimulus
    '''
    stim_trials = trials[trials['stimulus'] == stimulus_id]
    stim_starts = np.array(stim_trials['time_samples'])
    stim_ends = np.array(stim_trials['stimulus_end'])
    
    avg_fr = 0
    for (s_start, s_end) in zip(stim_starts, stim_ends):
        avg_fr += get_average_firing_rate(spikes , cluster_id, s_start, s_end, fs)
    return avg_fr / stim_trials.size

def get_stimulus_avg_sd_firing_rate(spikes, trials, cluster_id, stimulus_id, fs):
    '''
    Return the mean and standard deviation firing rate of a neuron over all trials from a given stimulus
    '''
    stim_trials = trials[trials['stimulus'] == stimulus_id]
    stim_starts = np.array(stim_trials['time_samples'])
    stim_ends = np.array(stim_trials['stimulus_end'])
    
    fr_list=[]
    for (s_start, s_end) in zip(stim_starts, stim_ends):
        fr_list.append(get_average_firing_rate(spikes , cluster_id, s_start, s_end, fs))
        
    return (np.mean(fr_list), np.std(fr_list))

    
def get_pair_stimulus_average_responses(spikes, trials, cluster_a, cluster_b, fs):
    '''
    Get the average responses from all stimuli for a pair of cells
    '''
    
    stimuli = sorted(trials['stimulus'].unique())
    ret = []
    for stim in stimuli:
        cluster_a_avg_response = get_stimulus_average_firing_rate(spikes, trials, cluster_a, stim, fs)
        cluster_b_avg_response = get_stimulus_average_firing_rate(spikes, trials, cluster_b, stim, fs)
        ret.append([cluster_a_avg_response, cluster_b_avg_response])
    return np.array(ret)


### Noise correlation funcs
def get_zscore_responses(spikes, trials, cluster_id, stimulus_id, fs):
    ''' Return an array of the z-scored responses from each trial for a given stim and cell'''
    
    # get mean and standard deviation response
    degenerate = False
    mu, sig = get_stimulus_avg_sd_firing_rate(spikes, trials, cluster_id, stimulus_id, fs)
    if (sig < 1e-12):
        print("Uh Oh... standard deviation very small")
        print("Cluster: {} Stimulus: {} mu: {} sigma: {}".format(cluster_id, stimulus_id, mu, sig))
        degenerate = True
    
    # for each trial compute z-score
    stim_trials = trials[trials['stimulus'] == stimulus_id]
    stim_starts = np.array(stim_trials['time_samples'])
    stim_ends = np.array(stim_trials['stimulus_end'])
    
    stim_clu_zscores = []
    for (s_start, s_end) in zip(stim_starts, stim_ends):
        if degenerate:
            stim_clu_zscores.append(0.0)
        else:
            trial_fr = get_average_firing_rate(spikes, cluster_id, s_start, s_end, fs)
            zscore = (trial_fr - mu)/sig
            stim_clu_zscores.append(zscore)
    return np.array(stim_clu_zscores)
    
def get_stim_zscores(spikes, trials, clusters, stimulus_id, fs):
    ''' Get an ncell x ntrial array of zscores for a given stim'''
    
    stimulus_zscores = []
    for cluster in clusters:
        stimulus_zscores.append(get_zscore_responses(spikes, trials, cluster, stimulus_id, fs))
    return np.vstack(stimulus_zscores)

def get_all_stim_zscores(spikes, trials, clusters, fs):
    ''' Return an ncell x ntrial x nstim array of zscores'''
    
    stimuli = sorted(trials['stimulus'].unique())
    all_stim_zscores = []
    for stim in stimuli:
        all_stim_zscores.append(get_stim_zscores(spikes, trials, clusters, stim, fs))
    return np.dstack(all_stim_zscores)
    
def get_pair_noise_correlation(stim_zscores, cluster_a, cluster_b):
    '''
    given an ncell x ntrial array of z-scores, compute the noise correlation between cells index a and index b
    cluster_a is the *index* of the cell, not its id
    '''

    z_a = stim_zscores[cluster_a, :]
    z_b = stim_zscores[cluster_b, :]
    return pearsonr(z_a, z_b)[0]

def get_pair_stim_avg_noise_correlation(all_stim_zscores, cluster_a, cluster_b):
    '''
    For a pair of cells given by *index* cluster_a, cluster_b, compute the noise correlation averaged over all stimuli
    
    '''
    
    nstim = all_stim_zscores.shape[2]
    corrs = []
    for stimnum in range(nstim):
        corrs.append(get_pair_noise_correlation(all_stim_zscores[:, :, stimnum], cluster_a, cluster_b))
    return np.mean(corrs)

def get_all_stim_avg_noise_correlation(all_stim_zscores, clusters):
    ''' Compute the stimulus averaged noise correlation from all pairs of cells
    all_stim_zscores is an ncell x ntrial x nstim array
    
    '''
    
    all_noise_corrs = []
    for pair in tqdm(combinations(range(len(clusters)),2)):
        all_noise_corrs.append(get_pair_stim_avg_noise_correlation(all_stim_zscores, pair[0], pair[1]))
    return np.array(all_noise_corrs)
    
###################################   
## Signal correlations
def get_signal_correlation(spikes, trials, cluster_a, cluster_b, fs):
    ''' Compute the signal correlation for a pair of cells
    
    spikes - spikes dataframe
    trials - trials dataframe containing all the trials for which you wish to compute signal correlations
                (e.g. all the trials from a particular class of stimuli)
    cluster_a - the *cluster id* of the first cell in the pair
    cluster_b - the *cluster id* of the second cell in the pair
    fs - sampling rate
    '''
    
    stim_responses = get_pair_stimulus_average_responses(spikes, trials, cluster_a, cluster_b, fs)
    stim_correlation = pearsonr(stim_responses[:, 0], stim_responses[:, 1])[0]
    return stim_correlation

def get_all_average_responses(spikes, trials, clusters, fs):
    ''' Return an ncell x nstim matrix of trial-averaged firing rate responses 
        spikes - spikes dataframe
        trials - trials dataframe containing all the trials for which you wish to compute signal correlations
                (e.g. all the trials from a particular class of stimuli)
        clusters - list of *cluster ids* for which you wish to compute average responses
                    the output array will have clusters indexed in this order
    '''
    
    stimuli = sorted(trials['stimulus'].unique())
    responses = np.zeros((len(clusters), len(stimuli)))
    for i in tqdm(range(len(clusters))):
        for j in range(len(stimuli)):
            responses[i, j] = get_stimulus_average_firing_rate(spikes, trials, clusters[i], stimuli[j], fs)
    return responses

from tqdm import tqdm
def get_all_signal_correlations(responses, clusters):
    ''' given an ncell x nstim matrix of trial averaged firing rate responses, 
    compute the signal correlations for every pair.
    
    responses - an ncell x nstim matrix of trial averaged firing rate responses
    clusters - array of *cluster ids* corresponding to the order of the rows in the response matrix
    '''
    all_pairs_signal_correlations = []
    for pair in tqdm(combinations(range(len(clusters)), 2)):
        response_a = responses[pair[0], :]
        response_b = responses[pair[1], :]
        pair_sig_corr = pearsonr(response_a, response_b)[0]
        all_pairs_signal_correlations.append(pair_sig_corr)
    return np.array(all_pairs_signal_correlations)