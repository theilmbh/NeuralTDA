import numpy as np 
import pandas as pd 
import topology as top 
import os

def generate_test_dataset(n_cells, maxt, fs, dthetadt, kappa, maxfr):

	N = maxt*fs 
	dt = 1.0/fs
	t = np.linspace(0, maxt, N)
	samps = np.round(t*fs)
	print(samps)
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

	spikes_dataframe = pd.DataFrame(columns=['cluster', 'time_samples', 'recording'])
	for clu in range(n_cells):
		clu_spikes = np.int_(np.round(samps[spikes[clu, :]==1]))
		clu_id = len(clu_spikes)*[clu]
		recs = len(clu_spikes)*[0]
		spdf_add = pd.DataFrame(data={'cluster': clu_id, 'time_samples': clu_spikes, 'recording': recs})
		spikes_dataframe = spikes_dataframe.append(spdf_add, ignore_index=True)
	spikes_dataframe = spikes_dataframe.sort(columns='time_samples')
	clus_dataframe = pd.DataFrame({'cluster': range(n_cells), 'quality': n_cells*['Good']})

	trials_dataframe = pd.DataFrame({'time_samples': [0], 'stimulus': ['test_pipeline_stimulus'], 'stimulus_end':[N]})

	return (spikes_dataframe, clus_dataframe, trials_dataframe)

def generate_and_bin_test_data(block_path, kwikfile, bin_id, bin_def_file, n_cells, maxt, fs, dthetadt, kappa, maxfr):

	spikes, clusters, trials = generate_test_dataset(n_cells, maxt, fs, dthetadt, kappa, maxfr)
	print(spikes.head())

	top.do_bin_data(block_path, spikes, clusters, trials, fs, kwikfile, bin_def_file, bin_id)


def test_pipeline(block_path, kwikfile, bin_id, bin_def_file, n_cells, maxt, fs, dthetadt, kappa, maxfr, n_cells_in_perm, nperms):

	generate_and_bin_test_data(block_path, kwikfile, bin_id, bin_def_file, n_cells, maxt, fs, dthetadt, kappa, maxfr)

	binned_folder = os.path.join(block_path, 'binned_data/{}'.format(bin_id))
	top.make_permuted_binned_data_recursive(binned_folder, n_cells_in_perm, nperms)







