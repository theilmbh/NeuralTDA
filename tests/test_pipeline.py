import os
import glob
import logging

import numpy as np 
import pandas as pd 

import topology as top 
import topology_plots as topplt

TEST_PIPELINE_LOGGER = logging.getLogger('NeuralTDA')

def generate_test_rfs(n_cells, maxt, fs, dthetadt, t):

	N = maxt*fs 
	dt = 1.0/fs
	t = np.linspace(0, maxt, N)
	samps = np.round(t*fs)
	rf_centers = np.linspace(0, 2*np.pi, n_cells)
	theta = dthetadt*t 
	thetas = np.tile(theta, (n_cells, 1))
	rf_centers_all = np.transpose(np.tile(rf_centers, (N, 1)))
	dthetas = thetas - rf_centers_all
	return dthetas

def generate_test_trial(n_cells, dthetas, kappa, maxfr, fs, samps):

	dt = 1.0/fs
	l_p = np.cos(dthetas)
	non_norm_p = np.exp(kappa*l_p)/(np.pi*2*np.i0(kappa))
	probs = non_norm_p*(maxfr*dt)
	rsamp = np.random.uniform(size=probs.shape)
	spikes = 1*np.less(rsamp, probs)

	spikes_dataframe = pd.DataFrame(columns=['cluster',
											 'time_samples',
											 'recording'])
	for clu in range(n_cells):
		clu_spikes = np.int_(np.round(samps[spikes[clu, :]==1]))
		clu_id = len(clu_spikes)*[int(clu)]
		recs = len(clu_spikes)*[int(0)]
		spdf_add = pd.DataFrame(data={'cluster': clu_id,
									  'time_samples': clu_spikes,
									  'recording': recs})
		spikes_dataframe = spikes_dataframe.append(spdf_add, ignore_index=True)

	spikes_dataframe = spikes_dataframe.sort(columns='time_samples')
	return spikes_dataframe

def generate_test_dataset(n_cells, maxt, fs, dthetadt, kappa, maxfr, ntrials):

	N = maxt*fs 
	dt = 1.0/fs
	t = np.linspace(0, maxt, N)
	dthetas = generate_test_rfs(n_cells, maxt, fs, dthetadt, t)
	N_silence = N + 4*fs

	spikes_dataframe = pd.DataFrame(columns=['cluster',
											 'time_samples',
											 'recording'])
	trial_starts = N_silence*np.arange(ntrials)
	for trial, trial_start in enumerate(trial_starts):
		TEST_PIPELINE_LOGGER.info('Generating test trial {} of {}'.format(trial, ntrials))
		samps = np.round(t*fs) + trial_start
		trial_data = generate_test_trial(n_cells, dthetas, kappa,
										 maxfr, fs, samps)
		spikes_dataframe = spikes_dataframe.append(trial_data, 
												   ignore_index=True)
	clus_dataframe = pd.DataFrame({'cluster': range(n_cells),
								   'quality': n_cells*['Good']})
	trials_df = pd.DataFrame({'time_samples': trial_starts,
							  'stimulus': ntrials*['test_pipeline_stimulus'],
							  'stimulus_end':trial_starts+N})
	return (spikes_dataframe, clus_dataframe, trials_df)


def generate_singletrial_test_dataset(n_cells, maxt, fs, dthetadt, kappa, maxfr):

	N = maxt*fs 
	dt = 1.0/fs
	t = np.linspace(0, maxt, N)
	samps = np.round(t*fs)
	rf_centers = np.linspace(0, 2*np.pi, n_cells)
	theta = dthetadt*t 
	thetas = np.tile(theta, (n_cells, 1))
	rf_centers_all = np.transpose(np.tile(rf_centers, (N, 1)))
	dthetas = thetas - rf_centers_all

	l_p = np.cos(dthetas)
	non_norm_p = np.exp(kappa*l_p)/(np.pi*2*np.i0(kappa))
	probs = non_norm_p*(maxfr*dt)
	rsamp = np.random.uniform(size=probs.shape)
	spikes = 1*np.less(rsamp, probs)

	spikes_dataframe = pd.DataFrame(columns=['cluster',
											 'time_samples',
											 'recording'])
	for clu in range(n_cells):
		clu_spikes = np.int_(np.round(samps[spikes[clu, :]==1]))
		clu_id = len(clu_spikes)*[int(clu)]
		recs = len(clu_spikes)*[int(0)]
		spdf_add = pd.DataFrame(data={'cluster': clu_id,
									  'time_samples': clu_spikes,
									  'recording': recs})
		spikes_dataframe = spikes_dataframe.append(spdf_add, ignore_index=True)

	spikes_dataframe = spikes_dataframe.sort(columns='time_samples')

	clus_dataframe = pd.DataFrame({'cluster': range(n_cells),
								   'quality': n_cells*['Good']})

	trials_dataframe = pd.DataFrame({'time_samples': [0],
									 'stimulus': ['test_pipeline_stimulus'],
									 'stimulus_end':[N]})

	return (spikes_dataframe, clus_dataframe, trials_dataframe)

def generate_and_bin_test_data(block_path, kwikfile, bin_id, bin_def_file,
							   n_cells, maxt, fs, dthetadt, kappa, maxfr, ntrials):
	
	TEST_PIPELINE_LOGGER.info('Generating test dataset')
	spikes, clusters, trials = generate_test_dataset(n_cells, maxt, fs,
													 dthetadt, kappa,
													 maxfr, ntrials)
	TEST_PIPELINE_LOGGER.info('Binning test dataset')
	top.do_bin_data(block_path, spikes, clusters, trials,
					fs, kwikfile, bin_def_file, bin_id)

def test_pipeline(block_path, bin_id, analysis_id, bin_def_file, n_cells=60, maxt=6,
				  fs=24000.0, dthetadt=2*np.pi, kappa=2, maxfr=12, n_trials=20,
				  n_cells_in_perm=40, nperms=1, thresh=4.0):

	TEST_PIPELINE_LOGGER.info('*** Testing NeuralTDA Pipeline ***')
	kwikfile = 'B999_P00_S00.kwik'
	generate_and_bin_test_data(block_path, kwikfile, bin_id, bin_def_file,
							   n_cells, maxt, fs, dthetadt, kappa, maxfr, n_trials)

	binned_folder = os.path.join(block_path, 'binned_data/{}'.format(bin_id))
	TEST_PIPELINE_LOGGER.info('Permuting binned test dataset')
	top.make_permuted_binned_data_recursive(binned_folder,
											n_cells_in_perm,
											nperms)

	permuted_folder = os.path.join(binned_folder, 'permuted_binned/')
	shuffled_permuted_folder = os.path.join(permuted_folder,
											'shuffled_controls/')
	TEST_PIPELINE_LOGGER.info('Shuffling permuted binned test dataset')
	top.make_shuffled_controls_recursive(permuted_folder, 1)

	# compute topology for permuted data:
	top.compute_all_ci_topology(binned_folder, permuted_folder, shuffled_permuted_folder,
								analysis_id, block_path, thresh)

	TEST_PIPELINE_LOGGER.info('Test NeuralTDA Complete')






