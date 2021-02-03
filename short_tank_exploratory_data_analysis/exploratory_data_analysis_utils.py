# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import os,sys
import pandas as pd
import numpy as np
import math
import h5py
import itertools
from functools import reduce

from progressbar import *

from sklearn.metrics import roc_curve, auc
from sklearn.utils.extmath import stable_cumsum

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from seaborn import heatmap

class ExploratoryDataset():
    def __init__(self, data_path):
        data_file = h5py.File(data_path, "r")

        hdf5_hit_pmt    = data_file["hit_pmt"]
        hdf5_hit_time   = data_file["hit_time"]
        hdf5_hit_charge = data_file["hit_charge"]

        self.hit_pmt    = np.memmap(data_path, mode="r", shape=hdf5_hit_pmt.shape,
                                    offset=hdf5_hit_pmt.id.get_offset(), dtype=hdf5_hit_pmt.dtype)

        self.hit_time   = np.memmap(data_path, mode="r", shape=hdf5_hit_time.shape,
                                            offset=hdf5_hit_time.id.get_offset(), dtype=hdf5_hit_time.dtype)

        self.hit_charge = np.memmap(data_path, mode="r", shape=hdf5_hit_charge.shape,
                                            offset=hdf5_hit_charge.id.get_offset(), dtype=hdf5_hit_charge.dtype)

        angles     = np.array(data_file['angles'])
        energies   = np.array(data_file['energies'])
        positions  = np.array(data_file['positions'])
        labels     = np.array(data_file['labels'])
        root_files = np.array(data_file['root_files'])

        # Initialize event geometry data
        self.barrel_map_array_idxs = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 15, 16, 17, 12, 13, 14, 18]
        self.pmts_per_mpmt = 19

        event_hits_index = np.append(data_file["event_hits_index"], hdf5_hit_pmt.shape[0]).astype(np.int64)

        mpmt_positions_file = '/data/WatChMaL/data/IWCDshort_mPMT_image_positions.npz'
        mpmt_positions   = np.load(mpmt_positions_file)['mpmt_image_positions']

        data_size = np.max(mpmt_positions, axis=0) + 1
        n_channels = self.pmts_per_mpmt
        self.data_size = np.insert(data_size, 0, n_channels)


def plot_compare_dists(dists, 
                       numerator_dist_idxs, 
                       denominator_dist_idxs,
                       labels, axes=None, colors=None, 
                       bins=20,
                       title=None, ratio_range=None, 
                       xlabel=None, 
                       xscale='linear', yscale='linear',
                       xrange=None,
                       linestyle=None, 
                       normalized=True,
                       histtype=u'step',
                       verbose=False, loc='best'):
    '''
    Plot distributions and plot their ratio.
    Args:
        dists                   ... list of 1d arrays
        numerator_dist_idxs     ... list of indices of distributions to use as numerator 
                                    in the ratio plot
        denominator_dist_idxs   ... list of indices of distributions to use as denominator
                                    in the ratio plot
        labels                  ... list of labels for each distribution
        axes                    ... optional, list of two matplotlib.pyplot.axes on which 
                                    to place the plots
        colors                  ... list of colors to use for each distribution
        bins                    ... number of bins to use in histogram
        title                   ... plot title
        ratio_range             ... range of distribution range to plot
        xlabel                  ... x-axis label
        linestyle               ... list of linestyles to use for each distribution
    author: Calum Macdonald
    June 2020
    '''
    ret = False


    __, plot_bins, __ = plt.hist(dists,
                                label=labels,
                                histtype=histtype,
                                bins=bins ,color=colors ,alpha=0.8)
    plt.close()    

    if axes is None:
        fig, axes = plt.subplots(2,1,figsize=(12,12))
        ret = True
    axes = axes.flatten()

    # Create hist plot
    ax = axes[0]

    if normalized:
        hist_weights = [np.ones(len(dists[i]))*1/len(dists[i]) for i in range(len(dists))]
    else:
        hist_weights = None
    
    if (xscale == 'log'):
        #print(plot_bins)
        plot_bins = np.logspace(np.log10(plot_bins[0]), np.log10(plot_bins[-1]), bins + 1)
        #print(plot_bins)
    
    # Plot main histogram
    ns, plot_bins, patches = ax.hist(dists, 
                            weights=hist_weights, 
                            label=labels,
                            histtype=histtype,
                            bins=plot_bins , color=colors ,alpha=0.8)

    if verbose:
        print("Bins: ", plot_bins)
    
    if linestyle is not None:
        for i,patch_list in enumerate(patches):
            for patch in patch_list:
                patch.set_linestyle(linestyle[i])
    
    if xrange is not None: 
        ax.set_xlim(xrange)

    ax.legend(loc=loc)

    if title is not None:
        if normalized:
            title = title + ' (Normalized)'
        ax.set_title(title)
    
    # Plot Ratio histogram
    ax2 = axes[1]
    for i, idx in enumerate(numerator_dist_idxs):
        lines = ax2.plot(plot_bins[:-1],     
                 ns[idx] / ns[denominator_dist_idxs[i]], 
                 alpha=0.8,label='{} to {}'.format(labels[idx],labels[denominator_dist_idxs[i]]),
                 )
        lines[0].set_color(patches[idx][0].get_edgecolor())
        lines[0].set_drawstyle('steps')
    
    if ratio_range is not None: 
        ax2.set_ylim(ratio_range)
    
    if xrange is not None: 
        ax2.set_xlim(xrange)

    ax2.legend()
    ax2.set_title('Ratio of Distributions')
    lines = ax2.plot(plot_bins[:-1],np.ones(len(plot_bins)-1),color='k',alpha=0.5)
    lines[0].set_linestyle('-.')

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)   

    ax2.set_xscale(xscale)  

    if xlabel is not None: 
        ax.set_xlabel(xlabel)
        ax2.set_xlabel(xlabel)
    
    ylabel = 'Count'

    if yscale == 'log':
        ylabel = ylabel + ' (log scale)'
    if normalized:
        ylabel = ylabel + ' (Normalized)'
    
    ax.set_ylabel(ylabel)

    if ret: return fig
    

def plot_computed_dists(dists, 
                       numerator_dist_idxs, denominator_dist_idxs,
                       labels, axes=None, colors=None, linestyle=None, 
                       bins=20,
                       title=None,
                       ratio_range=None, 
                       xlabel=None, xrange=None, 
                       xscale='linear', yscale='linear',
                       normalized=True,
                       histtype=u'step',
                       loc='best', verbose=False):
    '''
    Plot distributions and plot their ratio.
    Args:
        dists                   ... list of 1d arrays
        numerator_dist_idxs     ... list of indices of distributions to use as numerator 
                                    in the ratio plot
        denominator_dist_idxs   ... list of indices of distributions to use as denominator
                                    in the ratio plot
        labels                  ... list of labels for each distribution
        axes                    ... optional, list of two matplotlib.pyplot.axes on which 
                                    to place the plots
        colors                  ... list of colors to use for each distribution
        bins                    ... number of bins to use in histogram
        title                   ... plot title
        ratio_range             ... range of distribution range to plot
        xlabel                  ... x-axis label
        linestyle               ... list of linestyles to use for each distribution
    author: Calum Macdonald
    June 2020
    '''
    ret = False
    if axes is None:
        fig, axes = plt.subplots(2,1,figsize=(12,12))
        ret = True
    axes = axes.flatten()
    ax = axes[0]

    if normalized:
        hist_weights = [np.ones(len(dists[i]))*1/len(dists[i]) for i in range(len(dists))]
    else:
        hist_weights = None
    
    # Plot main histogram
    ns = []
    patches = []
    for idx in range(len(dists)):
        n_list, plot_bins, patche_list = ax.hist(
                                    x=bins[idx][:-1],
                                    bins=bins[idx], 
                                    weights=dists[idx],
                                    histtype=histtype, 
                                    label=labels[idx], 
                                    color=colors[idx],
                                    alpha=0.8,
                                    density=normalized)
        ns.append(n_list)
        patches.append(patche_list)

    if verbose:
        print("Bins: ", bins)
    
    if linestyle is not None:
        for i,patch_list in enumerate(patches):
            for patch in patch_list:
                patch.set_linestyle(linestyle[i])
    
    if xrange is not None: 
        ax.set_xlim(xrange)

    ax.legend(loc=loc)
    if title is not None: 
        ax.set_title(title)
    
    # Plot Ratio histogram

    
    ax2 = axes[1]
    for i, idx in enumerate(numerator_dist_idxs):
        lines = ax2.plot(bins[idx][:-1],     
                 ns[idx] / ns[denominator_dist_idxs[i]], 
                 alpha=0.8,label='{} to {}'.format(labels[idx],labels[denominator_dist_idxs[i]]),
                 )
        lines[0].set_color(patches[idx][0].get_edgecolor())
        lines[0].set_drawstyle('steps')
    
    if ratio_range is not None: 
        ax2.set_ylim(ratio_range)
    
    if xrange is not None: 
        ax2.set_xlim(xrange)

    ax2.legend()
    ax2.set_title('Ratio of Distributions')
    lines = ax2.plot(plot_bins[:-1],np.ones(len(plot_bins)-1),color='k',alpha=0.5)
    lines[0].set_linestyle('-.')

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)   

    ax2.set_xscale(xscale)


    ylabel = 'Count'

    if yscale == 'log':
        ylabel = ylabel + ' (log scale)'
    if normalized:
        ylabel = ylabel + ' (Normalized)'

    ax.set_ylabel(ylabel)

    if xlabel is not None: 
        ax.set_xlabel(xlabel)
        ax2.set_xlabel(xlabel)
    
    """
    if ret: return fig
    """
    

def parametrized_ray_point(x,y,z,theta,phi,t):
    '''
    parametrized_ray_point(x,y,z,theta,phi,t)
    Purpose: Find the point of a line departing (x,y,z) in direction specified by (theta,phi) and parametrized by t at
    given value of t. 
    Args: x, y, z           ... origin co-ordinates of line
          theta, phi        ... polar and azimuthal angles of departure
          t                 ... parameter giving desired point
    author: Calum Macdonald
    May 2020
    '''
    return x + np.sin(theta)*np.cos(phi)*t,y + np.sin(theta)*np.sin(phi)*t, z + np.cos(theta)*t


def distance_to_wall(position, angle):
    """
    distance_to_wall(position, angle)
    
    Purpose : Calculate distance from event origin to IWCD wall along particle trajectory.
    
    Args: position            ... array of [x, y, z] co-ordinates of event origin
          angle               ... array of [theta, phi] angles of departure
    author: Calum Macdonald
    May 2020
    """
    x = position[0]
    y = position[2]
    z = position[1]
    theta = angle[0]
    phi = angle[1]
    no_radial=False
    sols = []

    cylinder_half_height = 300 # Old value 521
    cylinder_radius = 400 # Old value 371

    #Solve for intersections of parametrized path with the cylinder and caps, keep only positive parameter solns
    try:
        shared_expression = np.sqrt(-np.sin(theta)**2*(-2*(cylinder_radius**2)+(x**2 + y**2)
                                 + (y**2 - x**2)*np.cos(2*phi)-2*x*y*np.sin(2*phi)))/(np.sin(theta)*np.sqrt(2))
    except:
        no_radial=True
    if not no_radial:
        try:
            radial_parameter_sol_1 = -1/np.sin(theta)*(x*np.cos(phi)+y*np.sin(phi) + shared_expression)

            if radial_parameter_sol_1 > 0: 
                sols.append(radial_parameter_sol_1)
        except:
            pass
        try:
            radial_parameter_sol_2 = 1/np.sin(theta)*(-x*np.cos(phi)-y*np.sin(phi) + shared_expression)

            if radial_parameter_sol_2 > 0: 
                sols.append(radial_parameter_sol_2)
        except:
            pass
    try:
        # Check for cap solutions
        cap_parameter_sol_top    =  (cylinder_half_height - z)/np.cos(theta)
        cap_parameter_sol_bottom = -(cylinder_half_height + z)/np.cos(theta)

        if cap_parameter_sol_top > 0: 
            sols.append(cap_parameter_sol_top)
        if cap_parameter_sol_bottom > 0: 
            sols.append(cap_parameter_sol_bottom)
    except:
        pass
    sols = np.sort(sols)
    x_int,y_int,z_int = parametrized_ray_point(x,y,z,theta,phi,sols[0])
    return np.sqrt((x-x_int)**2+(y-y_int)**2+(z-z_int)**2)

def event_hit_type(position, angle):
    """
    distance_to_wall(position, angle)
    
    Purpose : Calculate distance from event origin to IWCD wall along particle trajectory.
    
    Args: position            ... array of [x, y, z] co-ordinates of event origin
          angle               ... array of [theta, phi] angles of departure
    author: Calum Macdonald
    May 2020
    """
    hit_type_dict = {'endcap': 0, 'barrel': 1}

    x = position[0]
    y = position[2]
    z = position[1]
    theta = angle[0]
    phi = angle[1]
    no_radial=False

    sols = []
    hit_types = []

    cylinder_half_height = 300 # Old value 521
    cylinder_radius = 400 # Old value 371

    # Solve for intersections of parametrized path with the cylinder and caps, keep only positive parameter solns
    try:
        # Compute shared expression for barrel solutions
        shared_expression = np.sqrt(-np.sin(theta)**2*(-2*(cylinder_radius**2)+(x**2 + y**2)
                                 + (y**2 - x**2)*np.cos(2*phi)-2*x*y*np.sin(2*phi)))/(np.sin(theta)*np.sqrt(2))
    except:
        no_radial=True
    if not no_radial:
        # Check for barrel solutions
        try:
            radial_parameter_sol_1 = -1/np.sin(theta)*(x*np.cos(phi)+y*np.sin(phi) + shared_expression)

            if radial_parameter_sol_1 > 0: 
                sols.append(radial_parameter_sol_1)
                hit_types.append(hit_type_dict['barrel'])
        except:
            pass
        try:
            radial_parameter_sol_2 = 1/np.sin(theta)*(-x*np.cos(phi)-y*np.sin(phi) + shared_expression)

            if radial_parameter_sol_2 > 0: 
                sols.append(radial_parameter_sol_2)
                hit_types.append(hit_type_dict['barrel'])
        except:
            pass
    try:
        # Check for endcap solutions
        cap_parameter_sol_top    =  (cylinder_half_height - z)/np.cos(theta)
        cap_parameter_sol_bottom = -(cylinder_half_height + z)/np.cos(theta)

        if cap_parameter_sol_top > 0: 
            sols.append(cap_parameter_sol_top)
            hit_types.append(hit_type_dict['endcap'])
        if cap_parameter_sol_bottom > 0: 
            sols.append(cap_parameter_sol_bottom)
            hit_types.append(hit_type_dict['endcap'])
    except:
        pass

    sols = np.array(sols)
    hit_types = np.array(hit_types)

    sols_idxs = np.argsort(sols)

    return hit_types[sols_idxs][0]

def deprecated_distance_to_wall(position, angle):
    """
    distance_to_wall(position, angle)
    
    Purpose : Calculate distance from event origin to IWCD wall along particle trajectory.
    
    Args: position            ... array of [x, y, z] co-ordinates of event origin
          angle               ... array of [theta, phi] angles of departure
    author: Calum Macdonald
    May 2020
    """
    x = position[0]
    y = position[2]
    z = position[1]
    theta = angle[0]
    phi = angle[1]
    no_radial=False
    sols = []

    cylinder_half_height = 521
    cylinder_radius = 371

    #Solve for intersections of parametrized path with the cylinder and caps, keep only positive parameter solns
    try:
        shared_expression = np.sqrt(-np.sin(theta)**2*(-2*(cylinder_radius**2)+(x**2 + y**2)
                                 + (y**2 - x**2)*np.cos(2*phi)-2*x*y*np.sin(2*phi)))/(np.sin(theta)*np.sqrt(2))
    except:
        no_radial=True
    if not no_radial:
        try:
            radial_parameter_sol_1 = -1/np.sin(theta)*(x*np.cos(phi)+y*np.sin(phi) + shared_expression)

            if radial_parameter_sol_1 > 0: 
                sols.append(radial_parameter_sol_1)
        except:
            pass
        try:
            radial_parameter_sol_2 = 1/np.sin(theta)*(-x*np.cos(phi)-y*np.sin(phi) + shared_expression)

            if radial_parameter_sol_2 > 0: 
                sols.append(radial_parameter_sol_2)
        except:
            pass
    try:
        # Check for cap solutions
        cap_parameter_sol_top    =  (cylinder_half_height - z)/np.cos(theta)
        cap_parameter_sol_bottom = -(cylinder_half_height + z)/np.cos(theta)

        if cap_parameter_sol_top > 0: 
            sols.append(cap_parameter_sol_top)
        if cap_parameter_sol_bottom > 0: 
            sols.append(cap_parameter_sol_bottom)
    except:
        pass
    sols = np.sort(sols)
    x_int,y_int,z_int = parametrized_ray_point(x,y,z,theta,phi,sols[0])
    return np.sqrt((x-x_int)**2+(y-y_int)**2+(z-z_int)**2)

def deprecated_event_hit_type(position, angle):
    """
    distance_to_wall(position, angle)
    
    Purpose : Calculate distance from event origin to IWCD wall along particle trajectory.
    
    Args: position            ... array of [x, y, z] co-ordinates of event origin
          angle               ... array of [theta, phi] angles of departure
    author: Calum Macdonald
    May 2020
    """
    hit_type_dict = {'endcap': 0, 'barrel': 1}

    x = position[0]
    y = position[2]
    z = position[1]
    theta = angle[0]
    phi = angle[1]
    no_radial=False

    sols = []
    hit_types = []

    cylinder_half_height = 521
    cylinder_radius = 371

    # Solve for intersections of parametrized path with the cylinder and caps, keep only positive parameter solns
    try:
        # Compute shared expression for barrel solutions
        shared_expression = np.sqrt(-np.sin(theta)**2*(-2*(cylinder_radius**2)+(x**2 + y**2)
                                 + (y**2 - x**2)*np.cos(2*phi)-2*x*y*np.sin(2*phi)))/(np.sin(theta)*np.sqrt(2))
    except:
        no_radial=True
    if not no_radial:
        # Check for barrel solutions
        try:
            radial_parameter_sol_1 = -1/np.sin(theta)*(x*np.cos(phi)+y*np.sin(phi) + shared_expression)

            if radial_parameter_sol_1 > 0: 
                sols.append(radial_parameter_sol_1)
                hit_types.append(hit_type_dict['barrel'])
        except:
            pass
        try:
            radial_parameter_sol_2 = 1/np.sin(theta)*(-x*np.cos(phi)-y*np.sin(phi) + shared_expression)

            if radial_parameter_sol_2 > 0: 
                sols.append(radial_parameter_sol_2)
                hit_types.append(hit_type_dict['barrel'])
        except:
            pass
    try:
        # Check for endcap solutions
        cap_parameter_sol_top    =  (cylinder_half_height - z)/np.cos(theta)
        cap_parameter_sol_bottom = -(cylinder_half_height + z)/np.cos(theta)

        if cap_parameter_sol_top > 0: 
            sols.append(cap_parameter_sol_top)
            hit_types.append(hit_type_dict['endcap'])
        if cap_parameter_sol_bottom > 0: 
            sols.append(cap_parameter_sol_bottom)
            hit_types.append(hit_type_dict['endcap'])
    except:
        pass

    sols = np.array(sols)
    hit_types = np.array(hit_types)

    sols_idxs = np.argsort(sols)

    return hit_types[sols_idxs][0]

def plot_2d_ratio(dist_1_x,dist_1_y,dist_2_x, dist_2_y,bins=(150,150),fig=None,ax=None,
                  title=None, xlabel=None, ylabel=None, ratio_range=None):
    '''
    Plots the 2d ratio between the 2d histograms of two distributions.
    Args:
        dist_1_x:               ... 1d array of x-values of distribution 1 of length n
        dist_1_y:               ... 1d array of y-values of distribution 1 of length n
        dist_2_x:               ... 1d array of x-values of distribution 2 of length n
        dist_2_y:               ... 1d array of y-values of distribution 2 of length n
        bins:                   ... tuple of integer numbers of bins in x and y 
    author: Calum Macdonald
    May 2020
    '''
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(8,8))
    bin_range = [[np.min([np.min(dist_1_x),np.min(dist_2_x)]),np.max([np.max(dist_1_x),np.max(dist_2_x)])],
             [np.min([np.min(dist_1_y),np.min(dist_2_y)]),np.max([np.max(dist_1_y),np.max(dist_2_y)])]]
    ns_1, xedges, yedges = np.histogram2d(dist_1_x,dist_1_y,bins=bins,density=True,range=bin_range)
    ns_2,_,_ = np.histogram2d(dist_2_x,dist_2_y,bins=bins,density=True,range=bin_range)
    ratio = ns_1/ns_2
    ratio = np.where((ns_2==0) & (ns_1==0),1,ratio)
    ratio = np.where((ns_2==0) & (ns_1!=0),10,ratio)
    pc = ax.pcolormesh(xedges, yedges, np.swapaxes(ratio,0,1),vmin=ratio_range[0],vmax=ratio_range[1],cmap="RdBu_r")
    fig.colorbar(pc, ax=ax)
    if title is not None: ax.set_title(title)
    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    return fig


a = np.ones((100,100))

[i for i in itertools.product(range(3),range(3))]

def binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    '''
        SOURCE: Scikit.metrics internal usage tool
    '''
    """Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    pos_label : int or str, default=None
        The label of the positive class
    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.
    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    # classes.dtype.kind in ('O', 'U', 'S') is required to avoid
    # triggering a FutureWarning by calling np.array_equal(a, b)
    # when elements in the two arrays are not comparable.
    classes = np.unique(y_true)
    if (pos_label is None and (
            classes.dtype.kind in ('O', 'U', 'S') or
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1])))):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError("y_true takes value in {{{classes_repr}}} and "
                         "pos_label is not specified: either make y_true "
                         "take value in {{0, 1}} or {{-1, 1}} or "
                         "pass pos_label explicitly.".format(
                             classes_repr=classes_repr))
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]



def plot_binned_performance(softmaxes, labels, binning_features, binning_label,efficiency, bins, index_dict, 
                            label_0, label_1, metric='purity',fixed='rejection',ax=None,marker='o',color='k',title_note=''):
    '''
    Plots the purity as a function of a physical parameter in the dataset, at a fixed signal efficiency (true positive rate).
    Args:
        softmaxes                      ... 2d array with first dimension n_samples
        labels                         ... 1d array of true event labels
        binning_features               ... 1d array of features for generating bins, eg. energy
        binning_label                  ... name of binning feature, to be used in title and xlabel
        efficiency                     ... signal efficiency per bin, to be fixed
        bins                           ... either an integer (number of evenly spaced bins) or list of n_bins+1 edges
        index_dict                     ... dictionary of particle string keys and values corresponding to labels in 'labels'
        label_0                        ... string, positive particle label, must be key of index_dict
        label_1                        ... string, negative particle label, must be key of index_dict
        metric                         ... string, metric to plot ('purity' for signal purity, 'rejection' for rejection fraction, 'efficiency' for signal efficiency)
        ax                             ... axis on which to plot
        color                          ... marker color
        marker                         ... marker type
    author: Calum Macdonald
    June 2020
    '''
    legend_label_dict = {'gamma':'\u03B3','e':'e-','mu':'\u03BC -'}
    label_size = 14

    assert binning_features.shape[0] == softmaxes.shape[0], 'Error: binning_features must have same length as softmaxes'

    #bin by whatever feature
    if isinstance(bins, int):
        _,bins = np.histogram(binning_features, bins=bins)
    bins = bins[0:-1]
    bin_assignments = np.digitize(binning_features, bins)
    bin_data = []
    for bin_idx in range(len(bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        this_bin_idxs = np.where(bin_assignments==bin_num)[0]
        bin_data.append({'softmaxes':softmaxes[this_bin_idxs], 'labels' : labels[this_bin_idxs], 'n' : this_bin_idxs.shape[0]})

    #compute efficiency, thresholds, purity per bin
    bin_metrics = []
    for bin_idx, data in enumerate(bin_data):
        (softmaxes_0,softmaxes_1),(labels_0,labels_1) = separate_particles([data['softmaxes'],data['labels']],data['labels'],index_dict,desired_labels=[label_0,label_1])
        fps, tps, thresholds = binary_clf_curve(np.concatenate((labels_0,labels_1)),np.concatenate((softmaxes_0,softmaxes_1))[:,index_dict[label_0]], 
                                                pos_label=index_dict[label_0])
        fns = tps[-1] - tps
        tns = fps[-1] - fps
        efficiencies = tps/(tps + fns)
        operating_point_idx = (np.abs(efficiencies - efficiency)).argmin()
        if metric == 'purity': performance = tps[operating_point_idx]/(tps[operating_point_idx] + fps[operating_point_idx])
        else: performance = tns / (tns + fps)
        bin_metrics.append((efficiencies[operating_point_idx], performance[operating_point_idx], np.sqrt(tns[operating_point_idx])/(tns[operating_point_idx] + fps[operating_point_idx])))
    bin_metrics = np.array(bin_metrics)

    # plt.bar(bins,bin_metrics[:,1],align='edge',width=(np.max(binning_features)-np.min(binning_features))/len(bins))
    bin_centers = [(bins[i+1] - bins[i])/2 + bins[i] for i in range(0,len(bins)-1)]
    bin_centers.append((np.max(binning_features) - bins[-1])/2 + bins[-1])

    metric_name = '{}-{} Signal Purity'.format(label_0,label_1) if metric== 'purity' else '{} Rejection Fraction'.format(legend_label_dict[label_1])
    title = '{} \n vs {} At Bin {} Signal Efficiency {}{}'.format(metric_name, binning_label, legend_label_dict[label_0], efficiency,title_note)
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        plt.errorbar(bin_centers,bin_metrics[:,1],yerr=bin_metrics[:,2],fmt=marker,color=color,ecolor='k',elinewidth=0.5,capsize=4,capthick=1,alpha=0.5, linewidth=2)
        plt.ylabel('{} Signal Purity'.format(legend_label_dict[label_0]) if metric == 'purity' else '{} Rejection Fraction'.format(legend_label_dict[label_1]), fontsize=label_size)
        plt.xlabel(binning_label, fontsize=label_size)
        plt.title(title)

    else:
        ax.errorbar(bin_centers,bin_metrics[:,1],yerr=bin_metrics[:,2],fmt=marker,color=color,ecolor='k',elinewidth=0.5,capsize=4,capthick=1,alpha=0.5, linewidth=2)
        ax.set_ylabel('{} Signal Purity'.format(legend_label_dict[label_0]) if metric == 'purity' else '{} Rejection Fraction'.format(legend_label_dict[label_1]), fontsize=label_size)
        ax.set_xlabel(binning_label, fontsize=label_size)
        ax.set_title(title)
    # return bin_metrics[:,2]

def plot_fitqun_binned_performance(scores, labels, true_momentum, reconstructed_momentum, fpr_fixed_point, index_dict, recons_mom_bin_size=50, true_mom_bins=20, 
                            ax=None,marker='o',color='k',title_note='',metric='efficiency',yrange=None):

    label_size = 14

    #remove gamma events
    scores, labels, true_momentum, reconstructed_momentum = separate_particles([scores, labels, true_momentum, reconstructed_momentum],labels,index_dict,desired_labels=['e','mu'])
    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    true_momentum = np.concatenate(true_momentum)
    reconstructed_momentum = np.concatenate(reconstructed_momentum)

    #bin by reconstructed momentum
    bins = [0. + recons_mom_bin_size * i for i in range(math.ceil(np.max(reconstructed_momentum)/recons_mom_bin_size))]   
    bins = bins[0:-1]
    recons_mom_bin_assignments = np.digitize(reconstructed_momentum, bins)
    recons_mom_bin_idxs_list = [[]]*len(bins)
    for bin_idx in range(len(bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        recons_mom_bin_idxs_list[bin_idx] = np.where(recons_mom_bin_assignments==bin_num)[0]

    #compute threshold giving fixed fpr per reconstructed energy bin
    thresholds_per_event = np.ones_like(labels, dtype=float)
    for bin_idx, bin_idxs in enumerate(recons_mom_bin_idxs_list):        
        if bin_idxs.shape[0] > 0:
            fps, tps, thresholds = binary_clf_curve(labels[bin_idxs],scores[bin_idxs], 
                                                    pos_label=index_dict['e'])
            fns = tps[-1] - tps
            tns = fps[-1] - fps
            fprs = fps/(fps + tns)
            operating_point_idx = (np.abs(fprs - fpr_fixed_point)).argmin()
            thresholds_per_event[bin_idxs] = thresholds[operating_point_idx]

    #bin by true momentum
    ns,bins = np.histogram(true_momentum, bins=true_mom_bins, range=(200., np.max(true_momentum)) if metric=='mu fpr' else (0,1000))
    bins = bins[0:-1]
    true_mom_bin_assignments = np.digitize(true_momentum, bins)
    true_mom_bin_idxs_list = [[]]*len(bins)
    for bin_idx in range(len(bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        true_mom_bin_idxs_list[bin_idx]=np.where(true_mom_bin_assignments==bin_num)[0]

    #find metrics for each true momentum bin
    bin_metrics=[]
    for bin_idxs in true_mom_bin_idxs_list:
        pred_pos_idxs = np.where(scores[bin_idxs] - thresholds_per_event[bin_idxs] > 0)[0]
        pred_neg_idxs = np.where(scores[bin_idxs] - thresholds_per_event[bin_idxs] < 0)[0]
        fp = np.where(labels[bin_idxs[pred_pos_idxs]] == index_dict['mu'] )[0].shape[0]
        tp = np.where(labels[bin_idxs[pred_pos_idxs]] == index_dict['e'] )[0].shape[0]
        fn = np.where(labels[bin_idxs[pred_neg_idxs]] == index_dict['e'] )[0].shape[0]
        tn = np.where(labels[bin_idxs[pred_neg_idxs]] == index_dict['mu'] )[0].shape[0]
        if metric=='efficiency':
            bin_metrics.append(tp/(tp+fn))
        else:
            bin_metrics.append(fp/(fp + tn))

    #plot metrics
    bin_centers = [(bins[i+1] - bins[i])/2 + bins[i] for i in range(0,len(bins)-1)]
    bin_centers.append((np.max(true_momentum) - bins[-1])/2 + bins[-1] if metric=='mu fpr' else (1000 - bins[-1])/2 + bins[-1])
    metric_name = 'e- Signal Efficiency' if metric== 'efficiency' else '\u03BC- Mis-ID Rate'
    title = '{} \n vs True Momentum At Reconstructed Momentum Bin \u03BC- Mis-ID Rate of {}%{}'.format(metric_name, fpr_fixed_point*100, title_note)
    if ax is None:
        fig = plt.figure(figsize=(12,6))
        plt.errorbar(bin_centers,bin_metrics,yerr=np.zeros_like(bin_metrics),fmt=marker,color=color,ecolor='k',elinewidth=0.5,capsize=4,capthick=1,alpha=0.5, linewidth=2)
        plt.ylabel(metric_name)
        plt.xlabel("True Momentum (MeV/c)", fontsize=label_size)
        if yrange is not None: plt.ylim(yrange)
        plt.title(title)
    else:
        ax.errorbar(bin_centers[:50],bin_metrics[:50],yerr=np.zeros_like(bin_metrics[:50]),fmt=marker,color=color,ecolor='k',elinewidth=0.5,capsize=4,capthick=1,alpha=0.5, linewidth=2)
        nax = ax.twinx()
        # nax.bar(bin_centers,ns,fill=False,width=bins[3]-bins[2])
        ax.set_ylabel(metric_name)
        ax.set_xlabel("True Momentum (MeV/c)", fontsize=label_size)
        if yrange is not None: ax.set_ylim(yrange) 
        ax.set_title(title)

    return true_momentum, thresholds_per_event


def plot_response(softmaxes, labels, particle_names, index_dict,linestyle=None,bins=None,fig=None,axes=None,legend_locs=None,fitqun=False,xlim=None,label_size=14):
    '''
    Plots classifier softmax outputs for each particle type.
    Args:
        softmaxes                    ... 2d array with first dimension n_samples
        labels                       ... 1d array of particle labels to use in every output plot, or list of 4 lists of particle names to use in each respectively
        particle_names               ... list of string names of particle types to plot. All must be keys in 'index_dict' 
        index_dict                   ... dictionary of particle labels, with string particle name keys and values corresponsing to 
                                         values taken by 'labels'
        bins                         ... optional, number of bins for histogram
        fig, axes                    ... optional, figure and axes on which to do plotting (use to build into bigger grid)
        legend_locs                  ... list of 4 strings for positioning the legends
    author: Calum Macdonald
    June 2020
    '''
    
    legend_size=label_size
    legend_label_dict = {'gamma':'\u03B3','e':'e-','mu':'\u03BC -'}

    if axes is None:
        fig,axes = plt.subplots(1,4,figsize=(15,5)) if not fitqun else plt.subplots(1,1,figsize=(7,7))
    label_dict = {value:key for key, value in index_dict.items()}

    softmaxes_list = separate_particles([softmaxes], labels, index_dict, [name for name in index_dict.keys()])[0]
    
    for i, softmaxes in enumerate(softmaxes_list):
        p_name = particle_names[i]

    if isinstance(particle_names[0],str):
        particle_names = [particle_names for _ in range(4)]
    if fitqun:
        ax = axes
        density = False
        for i in [index_dict[particle_name] for particle_name in particle_names[1]]:
            _,bins,_ = ax.hist(softmaxes_list[i][:,1],
                        label=legend_label_dict[label_dict[i]],range=xlim,
                        alpha=0.7,histtype=u'step',bins=bins,density=density,
                        linestyle=linestyle[i],linewidth=2)    
            ax.legend(loc=legend_locs[0] if legend_locs is not None else 'best', fontsize=legend_size)
            ax.set_xlabel('e-muon nLL Difference')
            ax.set_ylabel('Normalized Density' if density else 'N Events', fontsize=label_size)
            # ax.set_yscale('log')
    else:
        for output_idx,ax in enumerate(axes[:-1]):
            for i in [index_dict[particle_name] for particle_name in particle_names[output_idx]]:
                ax.hist(softmaxes_list[i][:,output_idx],
                        label=f"{legend_label_dict[label_dict[i]]} Events",
                        alpha=0.7,histtype=u'step',bins=bins,density=True,
                        linestyle=linestyle[i],linewidth=2)            
            ax.legend(loc=legend_locs[output_idx] if legend_locs is not None else 'best', fontsize=legend_size)
            ax.set_xlabel('P({})'.format(legend_label_dict[label_dict[output_idx]]), fontsize=label_size)
            ax.set_ylabel('Normalized Density', fontsize=label_size)
            ax.set_yscale('log')
        ax = axes[-1]
        for i in [index_dict[particle_name] for particle_name in particle_names[-1]]:
                ax.hist(softmaxes_list[i][:,0] + softmaxes_list[i][:,1],
                        label=legend_label_dict[particle_names[-1][i]],
                        alpha=0.7,histtype=u'step',bins=bins,density=True,
                        linestyle=linestyle[i],linewidth=2)         
        ax.legend(loc=legend_locs[-1] if legend_locs is not None else 'best', fontsize=legend_size)
        ax.set_xlabel('P({}) + P({})'.format(legend_label_dict['gamma'],legend_label_dict['e']), fontsize=label_size)
        ax.set_ylabel('Normalized Density', fontsize=label_size)
        ax.set_yscale('log')
    plt.tight_layout()
    return fig

def rms(arr):
    '''
    Returns RMS value of the array.
    Args:
        arr                         ... 1d array of numbers
    author: Calum Macdonald
    June 2020
    '''
    return math.sqrt(reduce(lambda a, x: a + x * x, arr, 0) / len(arr))

def plot_binned_response(softmaxes, labels, binning_features, binning_label,efficiency, bins, p_bins, index_dict,log_scales=[]):
    '''
    Plot softmax response, binned in a feature of the event.
    Args:
        softmaxes                   ... 2d array of softmax output, shape (nsamples, 3)
        labels                      ... 1d array of labels, length n_samples
        binning_features            ... 1d array of feature to use in binning, length n_samples
        binning_label               ... string, name of binning feature to use in title and x-axis label
        efficiency                  ... bin signal efficiency to fix
        bins                        ... number of bins to use in feature histogram
        p_bins                      ... number of bins to use in probability density histogram
        index_dict                  ... dictionary of particle labels, must have 'gamma','mu','e' keys pointing to values taken by 'labels'
        log_scales                  ... indices of axes.flatten() to which to apply log color scaling
    author: Calum Macdonald
    June 2020
    '''
    legend_label_dict = {0:'\u03B3',1:'e-',2:'\u03BC-'}

    label_size = 18
    fig, axes = plt.subplots(3,4,figsize=(12*4,12*3))

    log_axes = axes.flatten()[log_scales]

    #bin by whatever feature
    if isinstance(bins, int):
        _,bins = np.histogram(binning_features, bins=bins)
    b_bin_centers = [bins[i] + (bins[i+1]-bins[i])/2 for i in range(bins.shape[0]-1)]
    binning_edges=bins
    bins = bins[0:-1]
    bin_assignments = np.digitize(binning_features, bins)
    bin_data = []
    for bin_idx in range(len(bins)):
        bin_num = bin_idx + 1 #these are one-indexed for some reason
        this_bin_idxs = np.where(bin_assignments==bin_num)[0]
        bin_data.append({'softmaxes':softmaxes[this_bin_idxs], 'labels' : labels[this_bin_idxs]})
    
    edges = None
    for output_idx in range(3):
        for particle_idx in range(3):
            ax = axes[particle_idx][output_idx]
            data = np.ones((len(bins), len(p_bins) if not isinstance(p_bins, int) else p_bins))
            means = []
            stddevs = []
            for bin_idx, bin in enumerate(bin_data):
                relevant_softmaxes = separate_particles([bin['softmaxes']],bin['labels'], index_dict)[0][particle_idx]

                if edges is None: ns, edges = np.histogram(relevant_softmaxes[:,output_idx], bins=p_bins,density=True,range=(0.,1.))
                else: ns, _ = np.histogram(relevant_softmaxes[:,output_idx], bins=edges,density=True)

                data[bin_idx, :] = ns
                p_bin_centers = [edges[i] + (edges[i+1]-edges[i])/2 for i in range(edges.shape[0]-1)]
                means.append(np.mean(relevant_softmaxes[:,output_idx]))
                stddevs.append(np.std(relevant_softmaxes[:,output_idx]))                

            if ax in log_axes:
                min_value = np.unique(data)[1]
                data = np.where(data==0, min_value, data)
            mesh = ax.pcolormesh(binning_edges, edges, np.swapaxes(data,0,1),norm=colors.LogNorm() if ax in log_axes else None)
            fig.colorbar(mesh,ax=ax)
            ax.errorbar(b_bin_centers, means, yerr = stddevs, fmt='k+', ecolor='k', elinewidth=0.5, capsize=4, capthick=1.5)
            ax.set_xlabel(binning_label,fontsize=label_size)
            ax.set_ylabel('P({})'.format(legend_label_dict[output_idx]),fontsize=label_size)
            ax.set_ylim([0,1])
            ax.set_title('P({}) Density For {} Events vs {}'.format(legend_label_dict[output_idx],legend_label_dict[particle_idx],binning_label),fontsize=label_size)

    for particle_idx in range(3):
            ax = axes[particle_idx][-1]
            data = np.ones((len(bins), len(p_bins) if not isinstance(p_bins, int) else p_bins))
            means = []
            stddevs = []
            for bin_idx, bin in enumerate(bin_data):
                relevant_softmaxes = separate_particles([bin['softmaxes']],bin['labels'], index_dict)[0][particle_idx]
                ns, _ = np.histogram(relevant_softmaxes[:,0] + relevant_softmaxes[:,1], bins=edges,density=True)
                data[bin_idx, :] = ns
                p_bin_centers = [edges[i] + (edges[i+1]-edges[i])/2 for i in range(edges.shape[0]-1)]
                means.append(np.mean(relevant_softmaxes[:,0] + relevant_softmaxes[:,1]))
                stddevs.append(np.std(relevant_softmaxes[:,0] + relevant_softmaxes[:,1]))
            if ax in log_axes:
                min_value = np.unique(data)[1]
                data = np.where(data==0, min_value, data)
            mesh = ax.pcolormesh(binning_edges,edges, np.swapaxes(data,0,1),norm=colors.LogNorm() if ax in log_axes else None)
            fig.colorbar(mesh,ax=ax)
            ax.set_ylim([0,1])
            ax.errorbar(b_bin_centers, means, yerr = stddevs, fmt='k+', ecolor='k', elinewidth=0.5, capsize=4, capthick=1.5)
            ax.set_xlabel(binning_label,fontsize=label_size)
            ax.set_ylabel('P({}) + P({})'.format(legend_label_dict[0], legend_label_dict[1]),fontsize=label_size)
            ax.set_title('P({}) + P({}) Density For {} Events vs {}'.format(legend_label_dict[0],legend_label_dict[1],legend_label_dict[particle_idx],binning_label),fontsize=label_size)


def separate_particles(input_array_list,labels,index_dict,desired_labels=['gamma','e','mu']):
    '''
    Separates all arrays in a list by indices where 'labels' takes a certain value, corresponding to a particle type.
    Assumes that the arrays have the same event order as labels. Returns list of tuples, each tuple contains section of each
    array corresponsing to a desired label.
    Args:
        input_array_list            ... list of arrays to be separated, must have same length and same length as 'labels'
        labels                      ... list of labels, taking any of the three values in index_dict.values()
        index_dict                  ... dictionary of particle labels, must have 'gamma','mu','e' keys pointing to values taken by 'labels', 
                                        unless desired_labels is passed
        desired_labels              ... optional list specifying which labels are desired and in what order.
    author: Calum Macdonald
    June 2020
    '''
    idxs_list = [np.where(labels==index_dict[label])[0] for label in desired_labels]

    separated_arrays = []
    for array in input_array_list:
        separated_arrays.append(tuple([array[idxs] for idxs in idxs_list]))

    return separated_arrays

def collapse_test_output(softmaxes, labels, index_dict,predictions=None,ignore_type=None):
    '''
    Collapse gamma class into electron class to allow more equal comparison to FiTQun.
    Args:
        softmaxes                  ... 2d array of dimension (n,3) corresponding to softmax output
        labels                     ... 1d array of event labels, of length n, taking values in the set of values of 'index_dict'
        index_dict                 ... Dictionary with keys 'gamma','e','mu' pointing to the corresponding integer
                                       label taken by 'labels'
        predictions                ... 1d array of event type predictions, of length n, taking values in the 
                                       set of values of 'index_dict'   
        ignore_type                ... single string, name of event type to exclude                     
    '''
    if ignore_type is not None:
        keep_indices = np.where(labels!=index_dict[ignore_type])[0]
        softmaxes = softmaxes[keep_indices]
        labels = labels[keep_indices]
        if predictions is not None: predictions = predictions[keep_indices]

    new_labels = np.ones((softmaxes.shape[0]))*index_dict['e']
    new_softmaxes = np.zeros((labels.shape[0], 3))
    if predictions is not None:
        new_predictions = np.ones_like(predictions) * index_dict['e']
    for idx, label in enumerate(labels):
            if index_dict["mu"] == label: new_labels[idx] = index_dict["mu"]
            new_softmaxes[idx,:] = [0,softmaxes[idx][0] + softmaxes[idx][1], softmaxes[idx][2]]
            if predictions is not None:
                if predictions[idx]==index_dict['mu']: new_predictions[idx] = index_dict['mu']

    if predictions is not None: return new_softmaxes, new_labels, new_predictions
    return new_softmaxes, new_labels
