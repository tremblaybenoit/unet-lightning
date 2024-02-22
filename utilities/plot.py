# General (computations):
import numpy as np
import scipy as sp
# For file searches:
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

def plot_maps(img, img_alpha=None, img_norm=None, 
              img_coord=None, img_shape=None, img_pixel=None, 
              img_ticks=None, img_labels=None, img_title=None, 
              img_labelspad=(5, 3), img_tickw=1, img_tickl=2.5, 
              img_tickdir='out', img_titlepad=1.005, 
              cb_label=None, cb_minmax=None, cb_cmap=None, 
              cb_pad=0, cb_tickw=1, cb_tickl=2.5, cb_font=None,
              cb_dir='out', cb_rot=270, cb_labelpad=15.5, cb_side='right', 
              plt_coord=None, plt_color=None,
              plt_linew=None, plt_lines=None, plt_symbl=None,
              vec=None, vec_coord=None, vec_step=None, 
              vec_scale=None, vec_width=None, vec_hwidth=None, 
              vec_hlength=None, vec_haxislength=None, vec_color=None,
              vec_qkey=None, vec_label=None,
              fig_filename=None, fig_show=False, fig_format='png', fig_dpi=300,
              fig_transparent=False, fig_lx=4.0, fig_ly=4.0, fig_lcb=5, fig_font=12,
              fig_left=0.8, fig_right=0.8, fig_bottom=0.48, fig_top=0.32, 
              fig_wspace=0.0, fig_hspace=0.0):
    
    # Make lists/tuples
    img = make_list(img, ndim=2)
    if img_alpha is not None: img_alpha = make_list(img_alpha)
    if img_norm is not None: img_norm = make_list(img_norm)
    if img_coord is not None: img_coord = make_list(img_coord, ndim=2)
    if img_shape is not None: img_shape = make_list(img_shape, ndim=2)
    if img_pixel is not None: img_pixel = make_list(img_pixel, ndim=2)
    if img_ticks is not None: img_ticks = make_list(img_ticks, ndim=2)
    if img_labels is not None: img_labels = make_list(img_labels, ndim=2)
    if img_title is not None: img_title = make_list(img_title)
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
    if vec is not None: vec = make_list(vec, ndim=2)
    if vec_coord is not None: vec_coord = make_list(vec_coord, ndim=2)
    if vec_step is not None: vec_step = make_list(vec_step)
    if vec_scale is not None: vec_scale = make_list(vec_scale)
    if vec_width is not None: vec_width = make_list(vec_width)
    if vec_hwidth is not None: vec_hwidth = make_list(vec_hwidth)
    if vec_hlength is not None: vec_hlength = make_list(vec_hlength)
    if vec_haxislength is not None: vec_haxislength = make_list(vec_haxislength)
    if vec_color is not None: vec_color = make_list(vec_color)
    if vec_qkey is not None: vec_qkey = make_list(vec_qkey)
    if vec_label is not None: vec_label = make_list(vec_label)
        
    # Dimensions
    nrows, ncols, _, _ = np.shape(img)

    # Set up default lists and values
    if img_alpha is None: img_alpha = np.ones((nrows, ncols)).tolist() 
    if img_norm is None: img_norm = [['linear', ] * ncols, ] * nrows 
    if img_coord is None: img_coord = [[(0, 0), ] * ncols, ] * nrows 
    if img_shape is None: img_shape = [[img[row][col].shape for col in range(ncols)] for row in range(nrows)]
    if img_pixel is None:
        img_pixel = [[(1, 1), ] * ncols, ] * nrows
        if img_labels is None: img_labels = [[('y-axis (pixels)', 'x-axis (pixels)'), ] * ncols, ] * nrows
    if cb_minmax is None: cb_minmax = [[colorbar_minmax(img[row][col], img_coord=img_coord[row][col], img_shape=img_shape[row][col]) for col in range(ncols)] for row in range(nrows)]
    if cb_font is None: cb_font = fig_font
    if plt_coord is not None: 
        # Color of lines
        if plt_linew is None: plt_linew = np.ones((nrows, ncols)).tolist()  
        # Style of lines
        if plt_lines is None: plt_lines = [['-', ] * ncols, ] * nrows
        # Symbol of lines
        if plt_symbl is None: plt_symbl = [['', ] * ncols, ] * nrows
    else:
        plt_coord = [[None, ] * ncols, ] * nrows
    
    # Layout
    font_size = fig_font
    fig_sizex = 0. 
    fig_sizey = 0.
    fig_specx = np.zeros((nrows, ncols+1)).tolist() 
    fig_specy = np.zeros((nrows+1, ncols)).tolist()
    img_ratio = [[img_shape[row][col][0]/img_shape[row][col][1] for col in range(ncols)] for row in range(nrows)] 
    for row in range(nrows):
        for col in range(ncols):
            if img_ratio[row][col] > 1:
                fig_specy[row+1][col] = fig_specy[row][col] + fig_bottom + fig_ly*img_ratio[row][col] + fig_top
                fig_specx[row][col+1] = fig_specx[row][col] + fig_left + fig_lx + fig_right
            else:
                fig_specy[row+1][col] = fig_specy[row][col] + fig_bottom + fig_ly + fig_top
                fig_specx[row][col+1] = fig_specx[row][col] + fig_left + fig_lx/img_ratio[row][col] + fig_right
            if row == 0 and col == ncols-1: fig_sizex = fig_specx[row][col+1]
            if row == nrows-1 and col == 0: fig_sizey = fig_specy[row+1][col]
    fig = plt.figure(figsize=(fig_sizex, fig_sizey), constrained_layout=False)

    # Rows
    for row in tqdm(range(nrows)):
        # Cols
        for col in tqdm(range(ncols)):

            # Adjust subplot layout
            spec = fig.add_gridspec(nrows=1, ncols=1, left=fig_left/fig_sizex + fig_specx[row][col]/fig_sizex,
                                    right= -fig_right/fig_sizex + fig_specx[row][col+1]/fig_sizex,
                                    bottom=fig_bottom/fig_sizey + fig_specy[(nrows-1)-row][col]/fig_sizey,
                                    top= -fig_top/fig_sizey + fig_specy[(nrows-1)-row+1][col]/fig_sizey, 
                                    wspace=fig_wspace, hspace=fig_hspace)
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
                          aspect=1, interpolation='none', alpha=img_alpha[row][col], origin='lower', norm=img_norm[row][col])
            
            # Plot vectors
            #    q = ax.quiver(x_plot, y_plot, vx_plot[::step[i][j], ::step[i][j]], vy_plot[::step[i][j], ::step[i][j]],
            #                  units='xy', scale=scale[i][j], width=width[i][j], headwidth=headwidth[i][j],
            #                  headlength=headlength[i][j],
            #                  headaxislength=headaxislength[i][j], pivot='tail', scale_units='xy')
            #    qk_label = str(np.around(qk_length[i][j], decimals=2))
            #    qk = ax.quiverkey(q, 0.05 / ncols + float(j) / ncols + 0.60 / ncols, 0.015 / nrows + i / nrows,
            #                      qk_length[i][j] / 1000., qk_label + r' km s$^{-1}$', labelpos='E',
            #                      coordinates='figure', fontproperties={'size': str(font_size)}, labelsep=0.05)

            # Plot lines
            if plt_coord[row][col] is not None:
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
            if img_title is not None: ax.set_title(img_title[row][col], fontsize=font_size, y=img_titlepad, wrap=True)
            # x/y-axis layout
            ax.get_yaxis().set_tick_params(which='both', direction=img_tickdir, width=img_tickw, length=img_tickl, labelsize=font_size, 
                                           left=True, right=True)
            ax.get_xaxis().set_tick_params(which='both', direction=img_tickdir, width=img_tickw, length=img_tickl, labelsize=font_size, 
                                           bottom=True, top=True)
            if img_labels is None:
                ax.set_ylabel('y-axis', fontsize=font_size, labelpad=img_labelspad[0])
                ax.set_xlabel('x-axis', fontsize=font_size, labelpad=img_labelspad[0])
            else: 
                ax.set_ylabel(img_labels[row][col][0], fontsize=font_size, labelpad=img_labelspad[0])
                ax.set_xlabel(img_labels[row][col][1], fontsize=font_size, labelpad=img_labelspad[0])
            if img_ticks is not None:
                ax.get_yaxis().set_major_locator(plt.MultipleLocator(img_ticks[row][col][0]))
                ax.get_xaxis().set_major_locator(plt.MultipleLocator(img_ticks[row][col][1]))

            # Colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(cb_side, size="{0}%".format(fig_lcb*fig_lx*ncols/fig_sizex), pad=cb_pad)
            cb = colorbar(I, extend='neither', cax=cax)
            cb.ax.tick_params(axis='y', direction=cb_dir, labelsize=cb_font, width=cb_tickw, length=cb_tickl)
            if cb_label is not None: cb.set_label(cb_label[row][col], labelpad=cb_labelpad, rotation=cb_rot, size=cb_font)

    # Show and Save
    if fig_filename is not None: plt.savefig(fig_filename, format=fig_format, dpi=fig_dpi, transparent=fig_transparent)
    if fig_show is False: 
        plt.close('all')
    else:
        plt.draw()


