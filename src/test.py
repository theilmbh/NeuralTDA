import numpy as np 
import scipy as sp 

import topology 

import pandas as pd

def generate_ring_dataset(N_neurons, times, fr_fact):
	''' 
	Generates a test data set that has a ring structure
	''' 
	#phases of the tuning curves
	thetas = np.linspace(0, 2*np.pi, N_neurons)
	period = float((times[-1] - times[0]))/ (4*np.pi)

	norm_times = times / period
	nt_mat = np.tile(norm_times, (N_neurons, 1))
	phases = (nt_mat.transpose() - thetas).transpose()
	fr = np.cos(phases) *fr_fact
	fr[fr<0] = 0.0

	spikes = 1.0*np.less(np.random.uniform(size=fr.shape), fr)
	print(spikes.shape)
	spikes_frame = pd.DataFrame(columns=['cluster', 'time_samples', 'recording'])

	for neuron in range(N_neurons):
		spikes_from_neuron = times[np.squeeze(spikes[neuron, :]) > 0]
		#print(spikes_from_neuron.shape)
		#print(spikes[neuron, :] > 0)
		#print(spikes_from_neuron)
		clusterid = (neuron*np.ones(len(spikes_from_neuron))).astype(int)
		recording = np.zeros(len(spikes_from_neuron)).astype(int)
		toadd = pd.DataFrame(data={'cluster': clusterid, 'time_samples': spikes_from_neuron, 'recording': recording})
		#print(toadd)
		spikes_frame = spikes_frame.append(toadd)

	spikes_frame.to_pickle('ring_dataset.pkl')
	return spikes_frame 