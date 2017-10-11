####################################################################
##### NeuralTDA                                                #####
##### plotting.py : Routines for plotting topological measures #####
##### Brad Theilman 2017-10-11                                 #####
####################################################################

import numpy as np
import neuraltda.topology2 as tp2 

def plot_betti_curve(bc, t, stim, betti, ax):
    '''
    Plots a betti curve for a fixed stimulus and betti number
    on the given axis.

    Parameters
    ----------
    bc : dict 
        Mean and stderr betti curves.
        Output of 'compute_mean_stderr_betti_curves' in topolog2 
    t : list 
        Vector of interpolation times, in milliseconds. 
        Output of 'compute_betti_curves' in topology2 
    stim : str 
        Name of stimulus to plot 
    ax : matplotlib axes object 
        Plot on which to plot. 

    '''
    avg = bc[stim][0]
    stderr = bc[stim][1]

    y = avg[betti, :]
    s = stderr[betti, :]

    ax.plot(t/1000., y, linewidth=2)
    ax.fill_between(t/1000., y-s, y+s, alpha=0.5)
    ax.set_xticks(range(int(np.amax(t)/1000.) + 1))

