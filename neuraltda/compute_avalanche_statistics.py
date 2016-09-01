import numpy as np 
import h5py as h5 
import topology as top 

def calc_avalanche_frequencies(binned_data, nbins):
	'''
	Calculates the frequencies of the number of 
	spikes in each time binned_data
	'''

	with h5.File(binned_data, 'r') as f:

		stims = f.keys()
		stim_popvec_sums = []
		for stim_num, stim in enumerate(stims):
			stim_trials = f[stim]
			trials = stim_trials.keys()
			trial_popvec_sums = []
			for trial_num, trial in enumerate(trials):
				trial_data = stim_trials[trial]
				trial_popvec = trial_data['pop_vec']
				trial_popvec_summed = np.sum(trial_popvec, 0) #Sum along cluster axis 
				stim_popvec_sums.append(trial_popvec_summed)
		
		stim_popvec_sums = np.ndarray(stim_popvec_sums)

		avalanche_histograms, edges = np.histogramdd(stim_popvec_sums, bins=nbins)
		return (avalanche_histograms, edges)

