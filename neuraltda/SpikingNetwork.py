import numpy as np 
import pandas as pd 
import brian2
import topology2 as tp2

class SpikingNetwork:

    fs = 40000
    N = 25
    tau = 10*brian2.ms
    v0_max = 3.
    duration = 4000*brian2.ms
    sigma = 0.2
    
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

        self.G = brian2.NeuronGroup(self.N, eqs, threshold='v>1', reset='v=0', refractory=5*brian2.ms, method='euler')
        self.M = brian2.SpikeMonitor(self.G)

        self.G.v0 = 0.7
        self.G.tau = self.tau
        self.G.sigma=self.sigma
        # Comment these two lines out to see what happens without Synapses
        self.S = brian2.Synapses(self.G, self.G, 'w : 1', on_pre='v_post += w')
        self.S.connect(p=0.1)
        self.S.w = 'randn()'
        self.S.delay = 1*brian2.ms
        
        self.net = brian2.Network()
        self.net.add(self.G, self.M, self.S)
        
    def initialize(self):
        return np.array(self.S.w)
    
    def run(self, weights=None):
        if weights:
            self.S.w = weights
        self.net.run(self.duration)

        poptens = np.array(self.format_output())
        return poptens

    def format_output(self):

        spikes = pd.DataFrame(data={'cluster': self.M.i, 'time_samples': np.int_(np.array(self.M.t)*self.fs), 'recording': len(self.M.t)*[0]})
        spikes = spikes.sort(columns='time_samples')
        stim_trials = pd.DataFrame(data={'time_samples': [0], 'stimulus_end': [float(self.duration)], 'recording': [0]})
        clusters_list = range(self.N)
        subwin_len = int(np.round(self.win_size/1000. * self.fs))
        noverlap = int(np.round(self.dt_overlap/1000. * self.fs))
        trial_len = float(self.duration)*self.fs
        segment = tp2.get_segment([0, trial_len], self.fs, [0, 0])
        poptens = tp2.build_activity_tensor_quick(stim_trials, spikes, clusters_list, self.N,
                                              self.win_size, subwin_len, noverlap, segment)
        return poptens