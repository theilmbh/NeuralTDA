import os
import glob

import numpy as np 
import pandas as pd 

import topology as top 
import topology_plots as topplt

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

	spikes, clusters, trials = generate_test_dataset(n_cells, maxt, fs,
													 dthetadt, kappa,
													 maxfr, ntrials)

	top.do_bin_data(block_path, spikes, clusters, trials,
					fs, kwikfile, bin_def_file, bin_id)

def test_pipeline(block_path, bin_id, analysis_id, bin_def_file, n_cells=60, maxt=6,
				  fs=24000.0, dthetadt=2*np.pi, kappa=2, maxfr=12, n_trials=10,
				  n_cells_in_perm=40, nperms=1, thresh=4.0):

	kwikfile = 'B999_P00_S00.kwik'
	generate_and_bin_test_data(block_path, kwikfile, bin_id, bin_def_file,
							   n_cells, maxt, fs, dthetadt, kappa, maxfr, ntrials)

	binned_folder = os.path.join(block_path, 'binned_data/{}'.format(bin_id))
	top.make_permuted_binned_data_recursive(binned_folder,
											n_cells_in_perm,
											nperms)

	permuted_folder = os.path.join(binned_folder, 'permuted_binned/')
	shuffled_permuted_folder = os.path.join(permuted_folder,
											'shuffled_controls/')
	top.make_shuffled_controls_recursive(permuted_folder, 1)

	# compute topology for permuted data:
	binned_data_files = glob.glob(os.path.join(binned_folder, '*.binned'))
	permuted_data_files = glob.glob(os.path.join(permuted_folder, '*.binned'))
	spdfs = os.path.join(shuffled_permuted_folder, '*.binned')
	shuffled_permuted_data_files = glob.glob(spdfs)
	
	for bdf in binned_data_files:
		top.calc_CI_bettis_hierarchical_binned_data(analysis_id, bdf,
													block_path, thresh)

	for pdf in permuted_data_files:
		top.calc_CI_bettis_hierarchical_binned_data(analysis_id+'_real', pdf,
													block_path, thresh)

	for spdf in shuffled_permuted_data_files:
		top.calc_CI_bettis_hierarchical_binned_data(analysis_id+'_shuffled',
													spdf, block_path, thresh)

	maxbetti = 5
	figsize=(22,22)
	topplt.make_all_plots(block_path, analysis_id, maxbetti, maxt, figsize)






