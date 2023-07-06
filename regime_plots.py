"""
Trevor Amestoy
July 2023
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors

from rodionov import rodionov_regimes
 
def plot_single_regime_changes(Z, shift_indices, rsi_values,
                               savefig=True, l=None, p=None):
    """Generates a plot of regime means and RSI values from Rodionov algo.

    Args:
        Z (pd.DataFrame): Standardized data; must have datetime index.
        shift_indices (list): list of indices with regime shifts
        rsi_values (array): list of RSI values for every period in the array.
        l (int): The L-parameter used in Rodionov; to be added to plot title
        p (float): The p parameter in Rodionov; to be added to plot title.
    """
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8,5), dpi=200, 
                                sharex=True, gridspec_kw={'height_ratios':[1,0.25]})

    cmap = mpl.colormaps['cool']
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    # Plot standardized Z values
    ax1.bar(Z.index, Z.values.flatten(), np.ones(len(Z)), label ='Standardized Flow', 
            color = 'cornflowerblue', alpha=0.75)

    # Add total mean
    ax1.hlines(np.mean(Z.values), xmin=Z.index[0], xmax=Z.index[-1], 
            color = 'black', ls = ':', label=f'Total Mean: {np.mean(Z.values):.{3}f}')

    if shift_indices:
        # Add initial regime mean
        r_mean = np.mean(Z.values[0:shift_indices[0]])
        ax1.hlines(r_mean, xmin=Z.index[0], xmax=Z.index[shift_indices[0]], 
                    color = 'orange', label=f'R1 Mean {r_mean:.{3}f}', lw=2.5, alpha = 0.9)

        for i in range(len(shift_indices)):
            r_start = shift_indices[i]
            if r_start == shift_indices[-1]:
                r_end = -1
            else:
                r_end = shift_indices[i+1]
            
            r_mean = np.mean(Z.values[r_start:r_end])
            ax1.hlines(r_mean, xmin=Z.index[r_start], xmax=Z.index[r_end], 
                    color = 'black', #cmap(norm(r_mean)), 
                    label=f'R{i+1} Mean {r_mean:.{3}f}', 
                    linewidth=2.5, alpha = 0.9)

    # Plot RSI values
    ax2.grid(which='major')
    ax2.bar(Z.index, rsi_values, alpha = 0.9, color='black')
    ax2.set_ylabel('RSI', fontsize=12)
    ax2.yaxis.set_label_coords(-0.08, 0.5)
    ax2.set_xlabel('Year', fontsize=12)


    ax1.set_ylabel('Standardized Flow', fontsize=12)
    ax1.yaxis.set_label_coords(-0.08, 0.5)
    if l:
        ax1.set_title(f'Regime shifts for l={l} and  p={p}')
    
    plt.tight_layout()
    if savefig:
        plt.savefig(f'regime_shifts_l{l}_p{p}.png', dpi=200)
    plt.show()
    return
 
 
 
def plot_regime_changes_with_alt_params(Z, l_min= 5, l_max=40,
                                        p=0.05, savefig=True):
    """Runs Rodionov algo for many different regime lengths (l) 
    and plots set of found regimes changes

    Args:
        Z (pd.DataFrame or pd.Series): Standardized array to be passed to Rodionov
        l_min (int, optional): Min l-param value to test. Defaults to 5.
        l_max (int, optional): Max l-param value to test. Defaults to 40.
        p (float, optional): Significance probability for t-test. Defaults to 0.05.
        savefig (bool, optional): Exports to cwd if True. Defaults to True.
    """
    
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(8,5), dpi=200, 
                               sharex=True, gridspec_kw={'height_ratios':[1,0.25,0.25]})
    cmap = mpl.colormaps['cool']
    norm = mcolors.Normalize(vmin=-1, vmax=1)

    # Plot standardized Z values
    ax1.bar(Z.index, Z.values.flatten(), np.ones(len(Z)), label ='Standardized Flow', 
            color = 'cornflowerblue', alpha=0.75)
    
    # Add total mean
    ax1.hlines(np.mean(Z.values), xmin=Z.index[0], xmax=Z.index[-1], 
            color = 'black', ls = ':', label=f'Total Mean: {np.mean(Z.values):.{3}f}')

    # Loop through different Rodionov L parameters 
    shift_counter = np.zeros(len(Z))
    test_lengths= np.arange(l_min, l_max, 1)
    for l_length in test_lengths:
        shift_indices, rsi_values = rodionov_regimes(Z.values.flatten(),
                                                        l_length, p = p)
        # If regimes were found, plot them
        if shift_indices:
            # Add initial regime mean
            r_mean = np.mean(Z.values[0:shift_indices[0]])
            ax1.hlines(r_mean, xmin=Z.index[0], xmax=Z.index[shift_indices[0]], 
                    color = 'orange', label=f'R1 Mean {r_mean:.{3}f}', lw=2.5, alpha = 0.2)

            # Plot different regimes as hlines
            for i in range(len(shift_indices)):
                r_start = shift_indices[i]
                shift_counter[r_start] += 1
                if r_start == shift_indices[-1]:
                    r_end = -1
                else:
                    r_end = shift_indices[i+1]
                
                r_mean = np.mean(Z.values[r_start:r_end])
                ax1.hlines(r_mean, xmin=Z.index[r_start], xmax=Z.index[r_end], 
                        color = 'black', #cmap(norm(r_mean)), 
                        label=f'R{i+1} Mean {r_mean:.{3}f}', 
                        linewidth=2.5, alpha = 0.2)

        # Plot RSI values
        ax2.grid(which='major')
        ax2.bar(Z.index, rsi_values, alpha = 0.2, color='black')
        ax2.set_ylabel('RSI', fontsize=12)
        ax2.yaxis.set_label_coords(-0.08, 0.5)
        
        # Plot Shift counts
        ax3.grid(which='major')
        ax3.bar(Z.index, shift_counter/len(test_lengths), color='maroon')
        ax3.set_ylabel('Found\nFreq.', fontsize=12)
        ax3.yaxis.set_label_coords(-0.08, 0.5)
        ax3.set_xlabel('Year', fontsize=12)
        ax3.set_ylim([0,1.0])

    ax1.set_ylabel('Standardized Flow', fontsize=12)
    ax1.yaxis.set_label_coords(-0.08, 0.5)
    ax1.set_title(f'Regime shifts detected across {len(test_lengths)} searches\nl=[{l_min},...,{l_max}], p={p}')
    plt.tight_layout()

    if savefig:
        plt.savefig('./regime_shifts_alt_params.png', dpi=200)
    plt.show()
    return