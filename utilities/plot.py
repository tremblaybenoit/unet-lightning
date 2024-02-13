# General (operations):
import os
import sys
# General (computations):
import numpy as np
import scipy as sp
# For file searches:
import glob
from astropy.io import fits
import gc
# Plotting
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import imageio
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import sunpy.visualization.colormaps as cm
# import copy


def colorbar_minmax(img, img_coord=None, img_shape=None):

    # Default values
    if img_coord is None:
        img_coord = (0, 0)
    if img_shape is None:
        img_shape = img.shape

    # Compute min/max
    min_img = np.nanmin(img[img_coord[0]:img_coord[0] + img_shape[0], img_coord[1]:img_coord[1] + img_shape[1]])
    max_img = np.nanmax(img[img_coord[0]:img_coord[0] + img_shape[0], img_coord[1]:img_coord[1] + img_shape[1]])

    # Adjust min/max to be symetrical if diverging
    if min_img*max_img < 0:
        return -np.nanmax([np.abs(min_img), np.abs(max_img)]), np.nanmax([np.abs(min_img), np.abs(max_img)])
    return min_img, max_img

def make_list(var, ndim=0):
    if np.ndim(var) == ndim: var = [var]
    if np.ndim(var) == ndim+1: var = [var]
    return var

def make_tuple(var, ndim=0):
    if np.ndim(var) == ndim: var = (var)
    if np.ndim(var) == ndim+1: var = (var)
    return var

def plot_maps(img, img_alpha=None, 
              img_coord=None, img_shape=None, img_pixel=None, 
              img_ticks=None, img_labels=None, img_title=None, 
              img_filename=None, img_show=False,  
              cb_label=None, cb_minmax=None, cb_cmap=None, 
              plt_coord=None, plt_color=None,
              plt_linew=None, plt_lines=None, plt_symbl=None,
              vec_color=None):
    
    # Make lists/tuples
    img = make_list(img, ndim=2)
    if img_alpha is not None: img_alpha = make_list(img_alpha)
    if img_coord is not None: img_coord = make_list(img_coord, ndim=2)
    if img_shape is not None: img_shape = make_list(img_shape, ndim=2)
    if img_pixel is not None: img_pixel = make_list(img_pixel, ndim=2)
    if img_ticks is not None: img_ticks = make_list(img_ticks, ndim=2)
    if img_labels is not None: img_labels = make_list(img_labels, ndim=2)
    if img_title is not None: img_title = make_list(img_title)
    if img_filename is not None: img_filename = make_list(img_filename)
    if img_show is not None: img_show = make_list(img_show)
    if cb_label is not None: cb_label = make_list(cb_label)
    if cb_minmax is not None: cb_minmax = make_list(cb_minmax)
    if cb_cmap is not None: cb_cmap = make_list(cb_cmap)
    if plt_coord is not None: 
        if np.ndim(plt_coord) == 1: 
            plt_coord = make_list(plt_coord, ndim=1)
        elif np.ndim(plt_coord) == 3:
            plt_coord = make_list(plt_coord, ndim=3)
    if plt_color is not None: plt_color = make_list(plt_color)
    if plt_linew is not None: plt_linew = make_list(plt_linew)
    if plt_lines is not None: plt_lines = make_list(plt_lines)
    if plt_symbl is not None: plt_symbl = make_list(plt_symbl)
    if vec_color is not None: vec_color = make_list(vec_color)
        
    # Dimensions
    nrows, ncols, _, _ = np.shape(img)

    # Set up default lists
    if img_alpha is None: img_alpha = np.ones((nrows, ncols)).tolist() 
    if img_coord is None: img_coord = [[(0, 0), ] * ncols, ] * nrows 
    if img_shape is None: img_shape = [[img[row][col].shape for col in range(ncols)] for row in range(nrows)]
    if img_pixel is None:
        img_pixel = [[(1, 1), ] * ncols, ] * nrows
        if img_labels is None: img_labels = [[('y-axis (pixels)', 'x-axis (pixels)'), ] * ncols, ] * nrows
    img_ratio = [[img_shape[row][col][0]/img_shape[row][col][1] for col in range(ncols)] for row in range(nrows)]
    # figsize = (np.sum(img_shape[row][col][0]/img_shape[row][col][1])) # TODO: Add figures of different "sizes" on a single row
    if cb_minmax is None: cb_minmax = [[colorbar_minmax(img[row][col], img_coord=img_coord[row][col], img_shape=img_shape[row][col]) for col in range(ncols)] for row in range(nrows)]
    if plt_coord is not None: 
        # Color of lines
        if plt_linew is None: plt_linew = np.ones((nrows, ncols)).tolist()  
        # Style of lines
        if plt_lines is None: plt_lines = [['-', ] * ncols, ] * nrows
        # Symbol of lines
        if plt_symbl is None: plt_symbl = [['', ] * ncols, ] * nrows

    # Layout
    font_size = 12
    figsize_x = ncols * 5.4075 
    figsize_y = nrows * 4.5  
    fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=False)

    # Rows
    for row in tqdm(range(nrows)):
        # Cols
        for col in tqdm(range(ncols)):

            # Adjust subplot layout
            # TODO: Fix ratio (only works for square images)
            spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05/ncols + float(col) / ncols,
                                    right=float(col + 1) / ncols - 0.08 / ncols,
                                    bottom=0.10 / nrows + float((nrows - 1 - row)) / nrows,
                                    top=float(nrows - row) / nrows - 0.10 / nrows, wspace=None, hspace=None)
            ax = fig.add_subplot(spec[:, :])
            
            # Extract subpatch
            img_plot = img[row][col][img_coord[row][col][0]:img_coord[row][col][0] + img_shape[row][col][0], 
                                     img_coord[row][col][1]:img_coord[row][col][1] + img_shape[row][col][1]]
            # Spatial extent
            extent = np.array([img_pixel[row][col][1]*img_coord[row][col][1], 
                               img_pixel[row][col][1]*(img_coord[row][col][1] + img_shape[row][col][1]), 
                               img_pixel[row][col][0]*img_coord[row][col][0], 
                               img_pixel[row][col][0]*(img_coord[row][col][0] + img_shape[row][col][0])])
            
            # Colormap
            if cb_cmap is None:
                # Divergent colormap (positive and negative)
                if cb_minmax[row][col][0] * cb_minmax[row][col][1] < 0:
                    cmap = 'RdBu_r'
                # Linear colormap
                else:
                    cmap = 'GnBu_r'
            else:
                cmap = cb_cmap[row][col]

            # Plot image
            I = ax.imshow(img_plot, extent=extent, cmap=cmap, vmin=cb_minmax[row][col][0], vmax=cb_minmax[row][col][1], 
                          aspect=1, interpolation='none', alpha=img_alpha[row][col], origin='lower')
            
            # Plot vectors

            # Plot lines
            if plt_coord is not None:

                # Color of lines
                if plt_color is None: 
                    if cmap == 'RdBu_r':
                        color = 'black'
                    elif cmap == 'GnBu_r':
                        color = 'red'
                    else:
                        color = 'black'
                else:
                    color = plt_color[row][col]

                # Plot lines
                for loop in range(len(plt_coord[row][col])):
                    x, y = zip(*plt_coord[row][col][loop])
                    ax.plot(x, y, color=color, 
                            linewidth=plt_linew[row][col], linestyle=plt_lines[row][col], marker=plt_symbl[row][col])
            
            # Title
            if img_title is not None: ax.set_title(img_title[row][col], fontsize=font_size, y=1.005, wrap=True)
            # x/y-axis layout
            ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size, 
                                           left=True, right=True)
            ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size, 
                                           bottom=True, top=True)
            if img_labels is None:
                ax.set_ylabel('y-axis', fontsize=font_size, labelpad=5.0)
                ax.set_xlabel('x-axis', fontsize=font_size, labelpad=3.0)
            else: 
                ax.set_ylabel(img_labels[row][col][0], fontsize=font_size, labelpad=5.0)
                ax.set_xlabel(img_labels[row][col][1], fontsize=font_size, labelpad=3.0)
            if img_ticks is not None:
                ax.get_yaxis().set_major_locator(plt.MultipleLocator(img_ticks[row][col][0]))
                ax.get_xaxis().set_major_locator(plt.MultipleLocator(img_ticks[row][col][1]))

            # Colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0)
            cb = colorbar(I, extend='neither', cax=cax)
            cb.ax.tick_params(axis='y', direction='out', labelsize=font_size, width=1, length=2.5)
            if cb_label is not None: cb.set_label(cb_label[row][col], labelpad=15.0, rotation=270, size=font_size)

    # Show and Save
    plt.draw()
    if img_filename is not None: plt.savefig(img_filename, format='png', dpi=300)
    if img_show is False: plt.close('all')



