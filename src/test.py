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

def test_ring_dataset(N_neurons, fs, max_fr):
	'''
	Runs a topology computation on a ring dataset 
	'''

	#Generate 1 second of samples:
	times = np.array(range(round(fs)))
	# Generate another second of samples 1 minute after the first
	times2 = times + 60.0*fs
	fr_fact = float(max_fr)/float(fs)

	#generate two trials
	spikes1 = generate_ring_dataset(N_neurons, times, fr_fact)
	spikes2 = generate_ring_dataset(N_neurons, times2, fr_fact)
	trials = pd.DataFrame({'time_samples': [times[0], times2[0]], 
						   'stimulus': ['test_ring_dataset', 'test_ring_dataset'])
						   'stimulus_end': [times[-1], times2[-1]]
						   })
	clusterIDs = range(N_neurons)
	qualities = ['Good' for i in range(N_neurons)]
	clusters = pd.DataFrame({'cluster': clusterIDs, 'quality': qualities})