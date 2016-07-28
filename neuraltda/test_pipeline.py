import numpy as np 
import pandas as pd 

def generate_test_dataset(n_cells, maxt, fs, dthetadt, kappa, maxfr):

	N = maxt*fs 
	dt = 1.0/fs
	t = np.linspace(0, maxt, N)
	samps = t*fs

	rf_centers = np.linspace(0, 2*np.pi, n_cells)

	theta = dthetadt*t 
	thetas = np.tile(theta, (n_cells, 1))

	rf_centers_all = np.tile(rf_centers, (1, N))

	dthetas = thetas - rf_centers_all

	l_p = np.cos(dthetas)
	non_norm_p = np.exp(kappa*l_p)/(np.pi*2)
	norms = np.sum(non_norm_p, 1) #sum over time
	norms = np.tile(norms, (1, N))
	probs = np.divide(non_norm_p, norms)*(dt/maxfr)

	rsamp = np.random.uniform(probs.shape)
	spikes = 1*np.less(rsamp, probs)

	spikes_dataframe = pd.DataFrame(columns=['cluster', 'time_samples', 'recording'])
	for clu in range(n_cells):
		clu_spikes = samps[spikes[clu, :]]
		clu_id = len(clu_spikes)*['{}'.format(str(clu))]
		recs = len(clu_spikes)*[0]
		spdf_add = {'cluster': clu_id, 'time_samples': clu_spikes, 'recording': recs}
		spikes_dataframe.append(spdf_add)
	spikes_dataframe.sort(columns='time_samples', inplace=True)
	clus_dataframe = pd.DataFrame({'cluster': range(n_cells), 'cluster_group': n_cells*['Good']})

	return (spikes_dataframe, clus_dataframe)





