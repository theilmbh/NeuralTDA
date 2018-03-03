####################################################################
##### NeuralTDA                                                #####
##### plotting.py : Routines for plotting topological measures #####
##### Brad Theilman 2017-10-11                                 #####
####################################################################
import os
import numpy as np
import neuraltda.topology2 as tp2 

def plot_betti_curve(bc, t, stim, betti, ax, **kwargs):
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

    Returns
    -------
    lines : matplotlib lines
        the plot lines

    '''
    avg = bc[stim][0]
    stderr = bc[stim][1]

    y = avg[betti, :]
    s = stderr[betti, :]

    lines = ax.plot(t/1000., y, **kwargs)
    ax.fill_between(t/1000., y-s, y+s, alpha=0.5, **kwargs)
    ax.set_xticks(range(int(np.amax(t)/1000.) + 1))

def plot_normalized_betti_curve(bc, t, stim, betti, ax, **kwargs):
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

    Returns
    -------
    lines : matplotlib lines
        the plot lines

    '''
    avg = bc[stim][0]
    stderr = bc[stim][1]

    y = avg[betti, :]
    s = stderr[betti, :]
    ymax = np.amax(y)
    if ymax < 1e-6:
        ymax=1

    lines = ax.plot(t/1000., y/ymax, **kwargs)
    ax.fill_between(t/1000., (y-s)/ymax, (y+s)/ymax, alpha=0.5, **kwargs)
    ax.set_xticks(range(int(np.amax(t)/1000.) + 1))
    return lines
    return lines

def save_fig(fig, fig_save_dir, figfname):

    figsave = os.path.join(fig_save_dir, figfname + '.pdf')
    fig.savefig(figsave, orientation='landscape')
