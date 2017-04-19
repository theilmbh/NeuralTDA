
# coding: utf-8

# In[1]:

import neuraltda.topology2 as tp2
from ephys import core, events
import pandas as pd
from joblib import Parallel, delayed


# In[2]:

winSizes = [2.0, 5.0, 10.0, 15.0, 25.0, 50.0, 100.0]

povers = 0.5
njobs = 10


# In[3]:

blockPath='./'


# # Entire Trial

# In[ ]:

print('Binning Entire Trials...')
segmentInfo = [0, 0] 
Parallel(n_jobs=njobs)(delayed(tp2.dag_bin)(blockPath, winSize, segmentInfo, dt_overlap=winSize*povers, cluster_group=['Good', 'MUA']) for winSize in winSizes)


# # Correct / Incorrect Trials

# In[ ]:

print('Binning Correct/Incorrect Trials...')
segmentInfo = [0, 0]
spikes = core.load_spikes(blockPath)
trials = events.load_trials(blockPath)
fs = core.load_fs(blockPath)
clusters = core.load_clusters(blockPath)

correctTrials = trials[trials['correct']==True]
incorrectTrials = trials[trials['correct']==False]

Parallel(n_jobs=njobs)(delayed(tp2.do_dag_bin_lazy)(blockPath, spikes, correctTrials, clusters, fs, winSize, segmentInfo, cluster_group=['Good', 'MUA'], dt_overlap=povers*winSize, comment='correct') for winSize in winSizes)
Parallel(n_jobs=njobs)(delayed(tp2.do_dag_bin_lazy)(blockPath, spikes, incorrectTrials, clusters, fs, winSize, segmentInfo, cluster_group=['Good', 'MUA'], dt_overlap=povers*winSize, comment='incorrect') for winSize in winSizes)


# # Active listening

# In[ ]:

print('Binning Active Listening Trials...')
segmentInfo = [0, 0] 
spikes = core.load_spikes(blockPath)
trials = events.load_trials(blockPath)
fs = core.load_fs(blockPath)
clusters = core.load_clusters(blockPath)

activeTrials = trials[pd.notnull(trials['response'])]

Parallel(n_jobs=njobs)(delayed(tp2.do_dag_bin_lazy)(blockPath, spikes, activeTrials, clusters, fs, winSize, segmentInfo, cluster_group=['Good', 'MUA'], dt_overlap=povers*winSize, comment='ActiveListening') for winSize in winSizes)


# # Sample / Distractor

# In[ ]:

print('Binning Sample/Distractor Period for all trials...')
segmentInfo = [2500, 0] 
Parallel(n_jobs=njobs)(delayed(tp2.dag_bin)(blockPath, winSize, segmentInfo, dt_overlap=winSize*povers, cluster_group=['Good', 'MUA'], comment='SampleDistractor') for winSize in winSizes)


# # Target

# In[ ]:

print('Binning Target Period for All Trials...')
segmentInfo = [0, -2500.0] 
Parallel(n_jobs=njobs)(delayed(tp2.dag_bin)(blockPath, winSize, segmentInfo, dt_overlap=winSize*povers, cluster_group=['Good', 'MUA'], comment='Target') for winSize in winSizes)


# # Sample/Distractor Correct/Incorrect

# In[ ]:

print('Binning SD for Correct/Incorrect Trials...')
segmentInfo = [2500, 0]
spikes = core.load_spikes(blockPath)
trials = events.load_trials(blockPath)
fs = core.load_fs(blockPath)
clusters = core.load_clusters(blockPath)

correctTrials = trials[trials['correct']==True]
incorrectTrials = trials[trials['correct']==False]

Parallel(n_jobs=njobs)(delayed(tp2.do_dag_bin_lazy)(blockPath, spikes, correctTrials, clusters, fs, winSize, segmentInfo, cluster_group=['Good', 'MUA'], dt_overlap=povers*winSize, comment='correct') for winSize in winSizes)
Parallel(n_jobs=njobs)(delayed(tp2.do_dag_bin_lazy)(blockPath, spikes, incorrectTrials, clusters, fs, winSize, segmentInfo, cluster_group=['Good', 'MUA'], dt_overlap=povers*winSize, comment='incorrect') for winSize in winSizes)


# # Target Correct/Incorrect

# In[ ]:

print('Binning Target for Correct/Incorrect Trials...')
segmentInfo = [0, -2500.]
spikes = core.load_spikes(blockPath)
trials = events.load_trials(blockPath)
fs = core.load_fs(blockPath)
clusters = core.load_clusters(blockPath)

correctTrials = trials[trials['correct']==True]
incorrectTrials = trials[trials['correct']==False]

Parallel(n_jobs=njobs)(delayed(tp2.do_dag_bin_lazy)(blockPath, spikes, correctTrials, clusters, fs, winSize, segmentInfo, cluster_group=['Good', 'MUA'], dt_overlap=povers*winSize, comment='correct') for winSize in winSizes)
Parallel(n_jobs=njobs)(delayed(tp2.do_dag_bin_lazy)(blockPath, spikes, incorrectTrials, clusters, fs, winSize, segmentInfo, cluster_group=['Good', 'MUA'], dt_overlap=povers*winSize, comment='incorrect') for winSize in winSizes)

