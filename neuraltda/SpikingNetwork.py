import numpy as np 
import pandas as pd 
import brian2
import .topology2 as tp2
import matplotlib.pyplot as plt 

class SpikingNetwork:

    fs = 40000
    N = 10
    tau = 10*brian2.ms
    v0_max = 3.
    duration = 500*brian2.ms
    sigma = 0.2
    ntrial = 1
    Ninputspikes = 100
    
    def __init__(self, win_size, dt_overlap):
        self.win_size = win_size
        self.dt_overlap = dt_overlap
        brian2.start_scope()
        
        eqs = '''
        dv/dt = (v0-v)/tau+sigma*xi*tau**-0.5 : 1 (unless refractory)
        v0 : 1
        tau : second
        sigma : 1
        '''

        eqs = '''
        dv/dt = (v0-v + A)/tau : 1 (unless refractory)
        v0 : 1
        tau : second
        sigma : 1
        A : 1
        '''

        syn_eqs = '''
        A_post = A_syn : 1 (summed)
        dA_syn/dt = -A_syn/t_syn : 1 
        t_syn : second 
        w : 1
        '''

        self.G = brian2.NeuronGroup(self.N, eqs, threshold='v>1', reset='v=0', refractory=5*brian2.ms, method='euler')
        self.M = brian2.SpikeMonitor(self.G)

        self.G.v0 = 0.25
        self.G.tau = self.tau
        self.G.sigma=self.sigma
        # Comment these two lines out to see what happens without Synapses
        self.S = brian2.Synapses(self.G, self.G, syn_eqs, on_pre='''A_syn += w''', method='euler')
        self.S.connect(condition='i!=j', p=1)
        self.S.w = 'randn()'
        self.S.t_syn = 5*brian2.ms
        self.S.delay = 1*brian2.ms

        in_spiketimes = float(self.duration)*np.random.rand(self.Ninputspikes)
        in_spikeclusters = np.random.randint(0, self.N, size=self.Ninputspikes)
        self.input_group = brian2.SpikeGeneratorGroup(self.N, in_spikeclusters, in_spiketimes*brian2.second)
        self.input_connection = brian2.Synapses(self.input_group, self.G, 'w : 1', on_pre='v_post += w')
        self.input_connection.connect(i=np.arange(self.N), j=np.arange(self.N))
        self.input_connection.w = 1.0
        
        
        self.net = brian2.Network()
        self.net.add(self.G, self.M, self.S, self.input_connection, self.input_group)
        self.net.store()
        self.net.store('init')
        self.spikes = pd.DataFrame(columns=['cluster', 'time_samples', 'recording'])
        self.stim_trials = pd.DataFrame(columns=['time_samples', 'stimulus_end', 'recording'])

    def initialize(self):
        return np.array(self.S.w)
    
    def run(self, weights=None):
        self.spikes = pd.DataFrame(columns=['cluster', 'time_samples', 'recording'])
        self.stim_trials = pd.DataFrame(columns=['time_samples', 'stimulus_end', 'recording'])
        for trial in range(self.ntrial):
            self.net.restore('init')
            if weights is not None:
                self.S.w = weights
            self.net.store('init')
            self.net.run(self.duration)
            new_spikes = pd.DataFrame(data={'cluster': self.M.i, 'time_samples': np.int_(np.array(self.M.t)*self.fs), 'recording': len(self.M.t)*[trial]})
            new_spikes = new_spikes.sort(columns='time_samples')
            self.spikes = self.spikes.append(new_spikes, ignore_index=True)
            stim_trials_new = pd.DataFrame(data={'time_samples': [0], 'stimulus_end': [float(self.duration)], 'recording': [trial]})
            self.stim_trials = self.stim_trials.append(stim_trials_new, ignore_index=True)

        poptens = np.array(self.format_output())
        return poptens

    def format_output(self):

        clusters_list = range(self.N)
        subwin_len = int(np.round(self.win_size/1000. * self.fs))
        noverlap = int(np.round(self.dt_overlap/1000. * self.fs))
        trial_len = float(self.duration)*self.fs
        segment = tp2.get_segment([0, trial_len], self.fs, [0, 0])
        poptens = tp2.build_activity_tensor_quick(self.stim_trials, self.spikes, clusters_list, self.N,
                                              self.win_size, subwin_len, noverlap, segment)
        return poptens

    def visualize(self):
        plt.figure()
        plt.plot(self.M.t, self.M.i, '.k')