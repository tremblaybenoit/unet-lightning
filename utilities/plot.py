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
import mpl_scatter_density # adds projection='scatter_density'


# Absolute errors
def absolute_errors_scalar(v1, v2):
    return np.sqrt((v1 - v2) ** 2)

# Relative errors
def relative_errors_scalar(v1, v2):
    return np.sqrt((v1 - v2) ** 2) / np.sqrt(v1 ** 2 + 1.e-12)

# Absolute errors
def absolute_errors_vector(v1_x, v1_y, v2_x, v2_y):
    return np.sqrt((v1_x - v2_x) ** 2 + (v1_y - v2_y) ** 2)

# Relative errors
def relative_errors_vector(v1_x, v1_y, v2_x, v2_y):
    return np.sqrt((v1_x - v2_x) ** 2 + (v1_y - v2_y) ** 2) / np.sqrt(v1_x ** 2 + v1_y ** 2)

# Cosine similatiry
def cosine_similarity_vector(v1_x, v1_y, v2_x, v2_y):
    return (v2_x * v1_x + v2_y * v1_y) / (np.sqrt(v2_x ** 2 + v2_y ** 2) * np.sqrt(v1_x ** 2 + v1_y ** 2))

def img_minmax(img, img_coord=None, img_shape=None):

    # Default values
    if img_coord is None:
        img_coord = (0, 0)
    if img_shape is None:
        img_shape = img.shape

    # Compute min/max
    min_img = np.nanmin(img[img_coord[0]:img_coord[0] + img_shape[0], img_coord[1]:img_coord[1] + img_shape[1]])
    max_img = np.nanmax(img[img_coord[0]:img_coord[0] + img_shape[0], img_coord[1]:img_coord[1] + img_shape[1]])

    return min_img, max_img

def colorbar_minmax(img, img_coord=None, img_shape=None):

    # Default values
    if img_coord is None:
        img_coord = (0, 0)
    if img_shape is None:
        img_shape = img.shape

    # Compute min/max
    min_img, max_img = img_minmax(img[img_coord[0]:img_coord[0] + img_shape[0], img_coord[1]:img_coord[1] + img_shape[1]])

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

def plot_scatter(reference, inference, 
                 img_coord=None, img_shape=None, 
                 scat_alpha=None, scat_ticks=None, scat_labels=None, scat_title=None, 
                 scat_labelspad=(5, 3), scat_tickw=1, scat_tickl=2.5, 
                 scat_tickdir='out', scat_titlepad=1.005, scat_proj=None, 
                 scat_marker='.', scat_markersize=0.9, scat_color=None, 
                 scat_grid=True, scat_gridlinew=0.5, 
                 ref_label='Reference (1:1)', ref_color=None, ref_linew=0.5, ref_lines='--',
                 fit=None, fit_color=None, fit_linew=0.25, fit_lines='-',
                 legend_loc=None, legend_font=10,
                 cb_label=None, cb_minmax=None, cb_cmap=None, 
                 cb_pad=0, cb_tickw=1, cb_tickl=2.5, cb_font=None,
                 cb_dir='out', cb_rot=270, cb_labelpad=16, cb_side='right', 
                 fig_filename=None, fig_show=False, fig_format='png', fig_dpi=300,
                 fig_transparent=False, fig_lx=4.0, fig_ly=4.0, fig_lcb=5, fig_font=12,
                 fig_left=0.8, fig_right=0.8, fig_bottom=0.48, fig_top=0.32, 
                 fig_wspace=0.0, fig_hspace=0.0):
    
    # Make lists/tuples
    reference = make_list(reference, ndim=2)
    inference = make_list(inference, ndim=2)
    if img_coord is not None: img_coord = make_list(img_coord, ndim=1)
    if img_shape is not None: img_shape = make_list(img_shape, ndim=1)
    if scat_alpha is not None: scat_alpha = make_list(scat_alpha)
    if scat_ticks is not None: scat_ticks = make_list(scat_ticks, ndim=1)
    if scat_labels is not None: scat_labels = make_list(scat_labels, ndim=1)
    if scat_proj is not None: scat_proj = make_list(scat_proj)
    if scat_title is not None: scat_title = make_list(scat_title)
    if fit is not None: fit = make_list(fit)
    if legend_loc is not None: legend_loc = make_list(legend_loc)
    if cb_label is not None: cb_label = make_list(cb_label)
    if cb_minmax is not None: cb_minmax = make_list(cb_minmax)
    if cb_cmap is not None: cb_cmap = make_list(cb_cmap)
        
    # Dimensions
    nrows, ncols, _, _ = np.shape(reference)

    # Colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']
    # "Viridis-like" colormap with white background
    if cb_cmap is None:
        white_viridis = LinearSegmentedColormap.from_list('white_viridis', 
                                                          [(0, '#ffffff'), 
                                                           (1e-20, '#440053'),
                                                           (0.2, '#404388'),
                                                           (0.4, '#2a788e'),
                                                           (0.6, '#21a784'),
                                                           (0.8, '#78d151'),
                                                           (1, '#fde624'),
                                                           ], N=256)

    # Set up default lists and values
    if img_coord is None: img_coord = [[(0, 0), ] * ncols, ] * nrows 
    if img_shape is None: img_shape = [[reference[row][col].shape for col in range(ncols)] for row in range(nrows)]
    if scat_alpha is None: scat_alpha = [[1.0, ] * ncols, ] * nrows
    if scat_ticks is None: scat_ticks = [[(1, 1), ] * ncols, ] * nrows 
    if scat_proj is None: scat_proj = [[None, ] * ncols, ] * nrows
    if scat_labels is None: scat_labels = [[('Inference', 'Reference'), ] * ncols, ] * nrows
    if scat_color is None: scat_color = colors[0]
    if ref_color is None: ref_color = 'black'
    if fit_color is None: fit_color = colors[1]
    if fit is None: fit = [[True, ] * ncols, ] * nrows
    if legend_loc is None: legend_loc = [['best', ] * ncols, ] * nrows
    if cb_font is None: cb_font = fig_font
    
    # Layout
    font_size = fig_font
    fig_sizex = 0. 
    fig_sizey = 0.
    fig_specx = np.zeros((nrows, ncols+1)).tolist() 
    fig_specy = np.zeros((nrows+1, ncols)).tolist()
    for row in range(nrows):
        for col in range(ncols):
            fig_specy[row+1][col] = fig_specy[row][col] + fig_bottom + fig_ly + fig_top
            fig_specx[row][col+1] = fig_specx[row][col] + fig_left + fig_lx + fig_right
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
            
            # Metrics
            if np.ndim(reference[row][col]) == 2:
                x_val = reference[row][col].copy().flatten()
                y_val = inference[row][col].copy().flatten()
            elif np.ndim(reference[row][col]) == 1:
                x_val = reference[row][col].copy().flatten()
                y_val = inference[row][col].copy().flatten()
            x_min, x_max = np.nanmin(x_val), np.nanmax(x_val)
            y_min, y_max = np.nanmin(y_val), np.nanmax(y_val)
            xy_range = [np.nanmin([x_min, y_min]), np.nanmax([x_max, y_max])]
            
            # Plot scatter
            if scat_proj[row][col] == 'scatter_density':
                ax = fig.add_subplot(spec[:, :], projection=scat_proj[row][col])
                I = ax.scatter_density(x_val, y_val, cmap=white_viridis)
            else:
                ax = fig.add_subplot(spec[:, :])
                I = ax.scatter(x_val, y_val, c=scat_color, alpha=scat_alpha[row][col], marker=scat_marker, s=scat_markersize)
            # Reference
            ax.set_aspect(1)
            ax.plot(xy_range, xy_range, label=ref_label, color=ref_color, linewidth=ref_linew, linestyle=ref_lines)
            # Fit
            if fit[row][col] is True:
                # Fit
                slope, origin = np.polyfit(x_val.flatten(), y_val.flatten(), 1)
                if origin >= 0:
                    fit_label = r'y = {0:.3f}x + {1:.3f}'.format(slope, origin)
                else:
                    fit_label = r'y = {0:.3f}x - {1:.3f}'.format(slope, np.abs(origin))
                ax.plot(xy_range, [xy_range[0] * slope + origin, xy_range[1] * slope + origin],
                        label=fit_label, color=fit_color, linewidth=fit_linew, linestyle=fit_lines)

            # Grid
            ax.grid(scat_grid, linewidth=scat_gridlinew)
            ax.set_xlim(xy_range)
            ax.set_ylim(xy_range)
                
            # Title
            if scat_title is not None: ax.set_title(scat_title[row][col], fontsize=font_size, y=scat_titlepad, wrap=True)
            # x/y-axis layout
            ax.get_yaxis().set_tick_params(which='both', direction=scat_tickdir, width=scat_tickw, length=scat_tickl, labelsize=font_size, 
                                           left=True, right=True)
            ax.get_xaxis().set_tick_params(which='both', direction=scat_tickdir, width=scat_tickw, length=scat_tickl, labelsize=font_size, 
                                           bottom=True, top=True)
            if scat_labels is None:
                ax.set_ylabel('Inference', fontsize=font_size, labelpad=scat_labelspad[0])
                ax.set_xlabel('Reference', fontsize=font_size, labelpad=scat_labelspad[1])
            else: 
                ax.set_ylabel(scat_labels[row][col][0], fontsize=font_size, labelpad=scat_labelspad[0])
                ax.set_xlabel(scat_labels[row][col][1], fontsize=font_size, labelpad=scat_labelspad[1])
            # Number of ticks
            if scat_ticks[row][col] is not None:
                ax.get_yaxis().set_major_locator(plt.MultipleLocator(scat_ticks[row][col][0]))
                ax.get_xaxis().set_major_locator(plt.MultipleLocator(scat_ticks[row][col][1]))

            # Legend
            plt.plot([], [], ' ', label=r"Spearman: {0:.3f}".format(sp.stats.spearmanr(x_val, y_val)[0]))
            plt.plot([], [], ' ', label=r"MAE: {0:.3f}".format(np.nanmean(absolute_errors_scalar(x_val, y_val))))
            plt.plot([], [], ' ', label=r"MAPE: {0:.3f}%".format(100.*np.nanmedian(relative_errors_scalar(x_val, y_val))))
            ax.legend(loc=legend_loc[row][col], fontsize=legend_font, numpoints=1, labelspacing=0.00, fancybox=False)
                
            # Colorbar
            divider = make_axes_locatable(ax)
            if scat_proj[row][col] == 'scatter_density':
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
              vec_qlength=None, vec_label=None, vec_labelsep=0.05,
              vec_qdecimals=2, vec_qscale=None, vec_qunits='',
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
    if vec_qlength is not None: vec_qlength = make_list(vec_qlength)
    if vec_qunits is not None: vec_qunits = make_list(vec_qunits)
    if vec_label is not None: vec_label = make_list(vec_label)
    if vec_qscale is not None: vec_qscale = make_list(vec_qscale)
        
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
    if vec is not None:
       if vec_step is None: vec_step = np.ones((nrows, ncols)).tolist() 
       if vec_scale is None: vec_scale = np.ones((nrows, ncols)).tolist()
       if vec_qscale is None: vec_qscale = np.ones((nrows, ncols)).tolist()
       if vec_qlength is None: vec_qlength = np.ones((nrows, ncols)).tolist()
       if vec_width is None: vec_width = np.ones((nrows, ncols)).tolist()
       if vec_hwidth is None: vec_hwidth = np.ones((nrows, ncols)).tolist()
       if vec_hlength is None: vec_hlength = np.ones((nrows, ncols)).tolist() 
       if vec_haxislength is None: vec_haxislength = np.ones((nrows, ncols)).tolist()
       if vec_label is None: vec_label = [['pixel', ] * ncols, ] * nrows
       if vec_color is None: vec_color = [['black', ] * ncols, ] * nrows
       # if vec_coord is None: vec_coord = [[None, ] * ncols, ] * nrows
    
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
            
            # TODO: Add vector plotting
            if vec is not None:

                #if vec_coord is None:
                #    vec_coord = [[None, ] * ncols, ] * nrows 

                # Plot vectors
                q = ax.quiver(vec_coord[row][col][1][::vec_step[row][col], ::vec_step[row][col]], 
                              vec_coord[row][col][0][::vec_step[row][col], ::vec_step[row][col]], 
                              vec[row][col][1][::vec_step[row][col], ::vec_step[row][col]], 
                              vec[row][col][0][::vec_step[row][col], ::vec_step[row][col]], units='xy', 
                              scale=vec_scale[row][col], width=vec_width[row][col], headwidth=vec_hwidth[row][col],
                              headlength=vec_hlength[row][col], headaxislength=vec_haxislength[row][col], 
                              pivot='tail', scale_units='xy')
                qk_label = str(np.around(vec_qlength[row][col], decimals=vec_qdecimals))
                qk = ax.quiverkey(q, (fig_specx[row][col] + fig_left + 1.1*fig_lx)/ fig_sizex, 
                                  (fig_specy[row][col] + 0.015*fig_bottom)/ fig_sizey,
                                  vec_qlength[row][col]*vec_qscale[row][col], qk_label + r' {0}'.format(vec_qunits), labelpos='E',
                                  coordinates='figure', fontproperties={'size': str(font_size)}, labelsep=vec_labelsep)

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
                ax.set_xlabel('x-axis', fontsize=font_size, labelpad=img_labelspad[1])
            else: 
                ax.set_ylabel(img_labels[row][col][0], fontsize=font_size, labelpad=img_labelspad[0])
                ax.set_xlabel(img_labels[row][col][1], fontsize=font_size, labelpad=img_labelspad[1])
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
