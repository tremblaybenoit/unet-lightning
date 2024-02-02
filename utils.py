# General (operations):
import os
import sys
import time
import datetime
# General (computations):
import numpy as np
import scipy as sp
import scipy.stats as sc
# import cupy as cp
# For file searches:
import glob
# File formats:
import h5py
from astropy.io import fits
import gc
# To save/load class instances:
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
# Plotting
import matplotlib.pyplot as plt
from pylab import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams['text.usetex'] = True
import imageio
# For downsampling:
from scipy import interpolate
# For downsampling (GPU acceleration):
from scipy.ndimage import map_coordinates
# from cupyx.scipy.ndimage import map_coordinates as map_coordinates_gpu
# Convolution:
# Convolution (GPU acceleration):
from scipy.signal import convolve2d
# from cupyx.scipy.ndimage import convolve as convolve_gpu
import multiprocessing as mp
from functools import partial
# import mpl_scatter_density # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap


class utils():

    def __init__(self, inst='', gpu=0, psfconv=1):

        # Use GPU acceleration
        self.gpu = gpu

        # Filenames (prefix)
        self.prefix_statevector_opt = 'tau_slice_med_'
        self.prefix_ic = 'I_out_med'
        # State vector indices
        self.nb_index = 10
        self.index_rho = 0
        self.index_vx = 1
        self.index_vy = 3
        self.index_vz = 2
        self.index_eint = 4
        self.index_bx = 5
        self.index_by = 7
        self.index_bz = 6
        self.index_temp = 8
        self.index_p = 9
        self.index_ic = 0

        # Conversion factors
        self.cm_to_km = 0.01 * 0.001
        self.km_to_cm = 1.0 / self.cm_to_km
        self.kgauss_to_gauss = 1000.
        self.gauss_to_kgauss = 1.0 / self.kgauss_to_gauss

        # Image dimensions (units: pixels)
        self.ny_sim = 256
        self.nx_sim = 512
        # Pixel size (units: km per pixel)
        self.dy_sim = 192.0
        self.dx_sim = 192.0
        # Timestep length (units: seconds)
        self.dt_sim = 60.
        self.nt_sim = 622

        # Instrument
        self.inst = inst
        # If intrument = SDO/HMI:
        if self.inst == 'HMI':
            # SDO/HMI spatial resolution (units: arcsec per pixel)
            cdelt1 = 0.504365
            cdelt2 = 0.504365
            # Radius in km
            rsun = 6.96e5
            # Radius in arcsec
            rsun_arcsec = 953.288
            # Radius in pixels
            rsun_pixels = np.round(rsun_arcsec/cdelt1)
            # Pixel size (units: km per pixel)
            self.dx = 384.
            self.dy = 384.
            # Image dimensions (units: pixels)
            self.nx = 256
            self.ny = 128
            # Timestep length (units: seconds)
            self.dt = 60.
            self.nt = 622
            self.ny_sim = 256
            self.nx_sim = 512
            self.dy_sim = 192.0
            self.dx_sim = 192.0

            if psfconv==1:

                # HMI PSF
                cdelt1 = 0.504365
                nx = 1001
                ny = 1001
                xmid = 500
                ymid = 500
                arcsec_to_rad = (1.0 / 3600.0) * (np.pi / 180.0)
                #
                wi = [0.641, 0.211, 0.066, 0.0467, 0.035]
                # Arcsec
                si = [0.470, 1.155, 2.09, 4.42, 25.77]
                #
                ai = [0.131, 0.371, 0.54, 0.781, 0.115]
                #
                ui = [1, 1, 2, 1, 1]
                # Radians
                vi = [-1.85, 2.62, -2.34, 1.255, 2.58]
                ni = len(vi)
                # PSF (HMI resolution)
                k1 = np.zeros((ny, nx))
                for i in range(ny):
                    for j in range(nx):
                        # Radial distance
                        r = np.sqrt(1.0 * (xmid - i) ** 2 + 1.0 * (ymid - j) ** 2) * cdelt1 * self.dx/self.dx_sim
                        phi = np.arctan2(self.dx/self.dx_sim * (ymid - j), self.dx/self.dx_sim * (xmid - i))
                        for k in range(ni):
                            k1[j, i] += (1.0 + ai[k] * np.cos(ui[k] * phi + vi[k])) * wi[k] * (
                                        np.exp(-0.5 * (r / si[k]) ** 2) / (2. * np.pi * si[k] ** 2))

                self.k1 = k1[250:750, 250:750] / np.sum(k1)

        elif self.inst == 'HMI_192km':
            # SDO/HMI spatial resolution (units: arcsec per pixel)
            cdelt1 = 0.504365
            cdelt2 = 0.504365
            # Radius in km
            rsun = 6.96e5
            # Radius in arcsec
            rsun_arcsec = 953.288
            # Radius in pixels
            rsun_pixels = np.round(rsun_arcsec/cdelt1)
            # Pixel size (units: km per pixel)
            self.dx = 384.
            self.dy = 384.
            # Image dimensions (units: pixels)
            self.nx = 256
            self.ny = 128
            # Timestep length (units: seconds)
            self.dt = 60.
            self.nt = 622
            self.ny_sim = 256
            self.nx_sim = 512
            self.dy_sim = 192.0
            self.dx_sim = 192.0

            if psfconv==1:

                # HMI PSF
                cdelt1 = 0.504365
                nx = 1001
                ny = 1001
                xmid = 500
                ymid = 500
                arcsec_to_rad = (1.0 / 3600.0) * (np.pi / 180.0)
                #
                wi = [0.641, 0.211, 0.066, 0.0467, 0.035]
                # Arcsec
                si = [0.470, 1.155, 2.09, 4.42, 25.77]
                #
                ai = [0.131, 0.371, 0.54, 0.781, 0.115]
                #
                ui = [1, 1, 2, 1, 1]
                # Radians
                vi = [-1.85, 2.62, -2.34, 1.255, 2.58]
                ni = len(vi)
                # PSF (HMI resolution)
                k1 = np.zeros((ny, nx))
                for i in range(ny):
                    for j in range(nx):
                        # Radial distance
                        r = np.sqrt(1.0 * (xmid - i) ** 2 + 1.0 * (ymid - j) ** 2) * cdelt1 * self.dx/self.dx_sim
                        phi = np.arctan2(self.dx/self.dx_sim * (ymid - j), self.dx/self.dx_sim * (xmid - i))
                        for k in range(ni):
                            k1[j, i] += (1.0 + ai[k] * np.cos(ui[k] * phi + vi[k])) * wi[k] * (
                                        np.exp(-0.5 * (r / si[k]) ** 2) / (2. * np.pi * si[k] ** 2))

                self.k1 = k1[250:750, 250:750] / np.sum(k1)

            # Pixel size (units: km per pixel)
            self.dx = 192.
            self.dy = 192.
            # Image dimensions (units: pixels)
            self.nx = 512
            self.ny = 256
                
        elif self.inst == '192km':
            # Pixel size (units: km per pixel)
            self.dx = 192.  # rsun / rsun_arcsec * cdelt1
            self.dy = 192.  # rsun / rsun_arcsec * cdelt2
            # Image dimensions (units: pixels)
            self.nx = 512
            self.ny = 256
            # Timestep length (units: seconds)
            self.dt = 60.
            self.nt = 622
            self.ny_sim = 256
            self.nx_sim = 512
            self.dy_sim = 192.0
            self.dx_sim = 192.0
        
        elif self.inst == '384km':
            # Pixel size (units: km per pixel)
            self.dx = 384.  
            self.dy = 384.  
            # Image dimensions (units: pixels)
            self.nx = 256
            self.ny = 128
            # Timestep length (units: seconds)
            self.dt = 60.
            self.nt = 622
            self.ny_sim = 256
            self.nx_sim = 512
            self.dy_sim = 192.0
            self.dx_sim = 192.0
        
        # If instrument = simulation:
        else:
            # Image dimensions (units: pixels)
            self.nx = self.nx_sim
            self.ny = self.ny_sim
            # Pixel size (units: km per pixel)
            self.dx = self.dx_sim
            self.dy = self.dy_sim
            # Timestep length (units: seconds)
            self.dt = self.dt_sim

    # Find files
    def find_files(self, filename_prefix, directory=''):

        filenames = sorted(glob.glob(directory + filename_prefix + '*'))

        return filenames

    # Read fits file
    def read_fits(self, filename, hmi=0):
        
        if hmi:
            f = fits.open(filename, memmap=False)
            img = f[1].data.copy()
            f.close()
        else:
            f = fits.open(filename, memmap=False)
            img = f[0].data.copy()
            f.close()
        gc.collect()
    
        return img

    # Write fits file
    def write_fits(self, filename, arr, overw=True):

         hdu = fits.PrimaryHDU(arr)
         hdulist = fits.HDUList([hdu])
         hdulist.writeto(filename, overwrite=overw)

    # Read .pkl file
    def load_class(self, filename):

        filename_load = file(filename, 'r')
        class_load = pickle.load(filename_load)

        return class_load

    # Write .pkl file
    def save_class(self, filename):

        filename_save = file(filename, 'w')
        pickle.dump(self, filename_save, -1)

    # Read MURaM continuum intensity or full state vector
    def read_binary(self, filename):
    
        f = np.fromfile(filename, dtype='float32')
        nvars = f[0].astype('int')
        ny = f[1].astype('int')
        nx = f[2].astype('int')
        t_iteration = f[3].astype('int')
        arr = f[4:]
        arr = arr.reshape((nvars, ny, nx))
        
        return arr

    # Read MURaM state vector variable
    def read_binary_slices(self, filename, index_qty):

        arr = self.read_binary(filename)
        arr_slice = arr[index_qty, :, :]
            
        return arr_slice

    '''
    # Interpolation routine
    # BT: Needs validation re: dimensions.
    def regrid(self, data):

        # Original dimensions:
        ny_data, nx_data = data.shape
        # Mesh:
        x_data = (np.arange(0, nx_data, 1, dtype='float32')-0.5*(nx_data-1))*self.dx_sim
        y_data = (np.arange(0, ny_data, 1, dtype='float32')-0.5*(ny_data-1))*self.dy_sim

        # New dimensions:
        nx_regrid = int(np.round(nx_data*self.dx_sim / self.dx))
        ny_regrid = int(np.round(ny_data*self.dy_sim / self.dy))
        # Mesh:
        x_regrid = (np.arange(0, nx_regrid, 1, dtype='float32')-0.5*(nx_regrid-1))*self.dx
        y_regrid = (np.arange(0, ny_regrid, 1, dtype='float32')-0.5*(ny_regrid-1))*self.dy

        # Regrid
        f = interpolate.interp2d(y_data, x_data, data, kind='linear')
        data_regrid = f(y_regrid, x_regrid)

        return data_regrid

    '''
    # Interpolation routine (with GPU acceleration)
    # BT: Needs validation re: dimensions.
    def regrid(self, data, order=3, mode='nearest'):
        
        # print(self.gpu)
        # Original dimensions:
        ny_data, nx_data = data.shape
        # Mesh:
        x_data = (np.arange(0, nx_data, 1, dtype='float32') - 0.5 * (nx_data - 1)) * self.dx_sim
        y_data = (np.arange(0, ny_data, 1, dtype='float32') - 0.5 * (ny_data - 1)) * self.dy_sim

        # New dimensions:
        nx_regrid = int(np.round(nx_data * self.dx_sim / self.dx))
        ny_regrid = int(np.round(ny_data * self.dy_sim / self.dy))
        # Mesh:
        x_regrid = (np.arange(0, nx_regrid, 1, dtype='float32') - 0.5 * (nx_regrid - 1)) * self.dx
        y_regrid = (np.arange(0, ny_regrid, 1, dtype='float32') - 0.5 * (ny_regrid - 1)) * self.dy
        i_coords, j_coords = np.meshgrid((y_regrid-np.amin(y_data))/self.dy_sim,
                                         (x_regrid-np.amin(x_data))/self.dx_sim, indexing='ij')

        # Regrid
        coordinates = np.array([i_coords, j_coords])

        if self.gpu == 0:
            data_regrid = map_coordinates(data, coordinates, order=order, mode=mode)
        else:
            data_regrid = cp.asnumpy(map_coordinates_gpu(cp.array(data), cp.array(coordinates), order=order, mode=mode))

        return data_regrid

    def make_hmi_k1(self, data, order=3, mode_interp='nearest', mode_conv='reflect'):

        # Convolve
        if self.gpu == 0:
            data_conv = cp.asnumpy(convolve2d(data, self.k1, mode=mode_conv))
        else:
            data_conv = cp.asnumpy(convolve_gpu(cp.array(data.astype('float32')),
                                                cp.array(self.k1), mode=mode_conv))
        # Regrid
        output = self.regrid(data_conv, order=order, mode=mode_interp)

        return output

    def make_hmi_k1_192km(self, data, order=3, mode_interp='nearest', mode_conv='reflect'):

        # Convolve
        if self.gpu == 0:
            data_conv = cp.asnumpy(convolve2d(data, self.k1, mode=mode_conv))
        else:
            data_conv = cp.asnumpy(convolve_gpu(cp.array(data.astype('float32')),
                                                cp.array(self.k1), mode=mode_conv))

        return data_conv

    def divergence(self, vx, vy):
    
        dvx = np.gradient(vx)
        dvx_y = dvx[0]
        dvx_x = dvx[1]
    
        dvy = np.gradient(vy)
        dvy_y = dvy[0]
        dvy_x = dvy[1]
    
        div = dvx_x + dvy_y
    
        return div

    def vorticity(self, vx, vy):
    
        dvx = np.gradient(vx)
        dvx_y = dvx[0]
        dvx_x = dvx[1]
    
        dvy = np.gradient(vy)
        dvy_y = dvy[0]
        dvy_x = dvy[1]
    
        w_z = dvy_x - dvx_y
    
        return w_z

    def sz(self, vx, vy, vz, bx, by, bz):
    
        sz = 0.25/np.pi*((bx*bx + by*by)*vz - (vx*bx + vy*by)*bz)
    
        return sz

    # Metrics for flow reconstructions
    # def metrics(self, v1_x, v1_y, v2_x, v2_y):
    #    # 0: Pearson correlation coefficient
    #    # 1: Mean abs. error
    #    # 2: Mean rel. error
    #    # 3: Normalized dot product
    #    # 4: RMSE
    #    img_prop = np.zeros(5)
    #
    #    # Flatten
    #    v1_x = v1_x.flatten()
    #    v1_y = v1_y.flatten()
    #    v2_x = v2_x.flatten()
    #    v2_y = v2_y.flatten()
    #    # Dimensions
    #    n = float(len(v1_x))
    #
    #    # Metrics
    #    img_prop[1] = np.mean(np.sqrt((v1_x - v2_x) ** 2 + (v1_y - v2_y) ** 2))
    #    img_prop[2] = np.mean(np.sqrt((v1_x - v2_x) ** 2 + (v1_y - v2_y) ** 2) / np.sqrt(v1_x ** 2 + v1_y ** 2))
    #    img_prop[0] = np.sum((v1_x * v2_x) + (v1_y * v2_y)) / np.sqrt(
    #        np.sum(v1_x ** 2 + v1_y ** 2) * np.sum(v2_x ** 2 + v2_y ** 2))
    #    img_prop[3] = (1.0 / n) * np.sum(
    #        (v2_x * v1_x + v2_y * v1_y) / (np.sqrt(v2_x ** 2 + v2_y ** 2) * np.sqrt(v1_x ** 2 + v1_y ** 2)))
    #    img_prop[4] = np.sqrt(np.mean((v1_x - v2_x) ** 2 + (v1_y - v2_y) ** 2))
    #
    #    return img_prop

    # Metrics for flow reconstructions
    def metrics_ar_qs(self, v1_x, v1_y, v2_x, v2_y, b1_x, b1_y, b1_z, bthresh=250):
    
        # Dimensions
        nb_frames, _, _ = v1_x.shape
    
        # Metrics array
        metrics = np.zeros((nb_frames, 5, 5))
    
        # Metrics (QS+AR)
        for i in range(nb_frames):
            m = self.metrics(v1_x[i, :, :], v1_y[i, :, :], v2_x[i, :, :], v2_y[i, :, :])
            metrics[i, :, 0] = m
    
        # Metrics (AR)
        for i in range(nb_frames):
            where_b = np.where(np.sqrt(b1_x[i, :, :] ** 2 + b1_y[i, :, :] ** 2 + b1_z[i, :, :] ** 2) >= bthresh)
            if len(where_b[0]) > 0:
                m = self.metrics(v1_x[i, :, :][where_b], v1_y[i, :, :][where_b], v2_x[i, :, :][where_b],
                                 v2_y[i, :, :][where_b])
                metrics[i, :, 1] = m
            where_b = np.where(np.sqrt(b1_z[i, :, :] ** 2) >= bthresh)
            if len(where_b[0]) > 0:
                m = self.metrics(v1_x[i, :, :][where_b], v1_y[i, :, :][where_b], v2_x[i, :, :][where_b],
                                 v2_y[i, :, :][where_b])
                metrics[i, :, 3] = m
    
        # Metrics (QS)
        for i in range(nb_frames):
            where_b = np.where(np.sqrt(b1_x[i, :, :] ** 2 + b1_y[i, :, :] ** 2 + b1_z[i, :, :] ** 2) < bthresh)
            m = self.metrics(v1_x[i, :, :][where_b], v1_y[i, :, :][where_b], v2_x[i, :, :][where_b],
                             v2_y[i, :, :][where_b])
            metrics[i, :, 2] = m
            where_b = np.where(np.sqrt(b1_z[i, :, :] ** 2) < bthresh)
            m = self.metrics(v1_x[i, :, :][where_b], v1_y[i, :, :][where_b], v2_x[i, :, :][where_b],
                             v2_y[i, :, :][where_b])
            metrics[i, :, 4] = m
    
        return metrics

    # Absolute errors
    def absolute_errors_scalar(self, v1, v2):
    
        return np.sqrt((v1 - v2) ** 2)

    # Relative errors
    def relative_errors_scalar(self, v1, v2):
    
        return np.sqrt((v1 - v2) ** 2) / np.sqrt(v1 ** 2 + 1.e-12)

    # Absolute errors
    def absolute_errors_vector(self, v1_x, v1_y, v2_x, v2_y):
    
        return np.sqrt((v1_x - v2_x) ** 2 + (v1_y - v2_y) ** 2)

    # Relative errors
    def relative_errors_vector(self, v1_x, v1_y, v2_x, v2_y):
    
        return np.sqrt((v1_x - v2_x) ** 2 + (v1_y - v2_y) ** 2) / np.sqrt(v1_x ** 2 + v1_y ** 2)

    # Cosine similatiry
    def cosine_similarity_vector(self, v1_x, v1_y, v2_x, v2_y):
    
        return (v2_x * v1_x + v2_y * v1_y) / (np.sqrt(v2_x ** 2 + v2_y ** 2) * np.sqrt(v1_x ** 2 + v1_y ** 2))

    # Metrics for flow reconstructions
    def metrics_scalar(self, v1, v2):
        # 0: Pearson correlation coefficient
        # 1: Mean abs. error
        # 2: Mean rel. error
        # 3: Normalized dot product
        # 4: RMSE
        img_prop = np.zeros(5)
    
        # Flatten
        v1 = v1.flatten()
        v2 = v2.flatten()
        # Dimensions
        n = float(len(v1))
    
        # Metrics
        img_prop[1] = np.mean(self.absolute_errors_scalar(v1, v2))
        img_prop[2] = np.mean(self.relative_errors_scalar(v1, v2))
        img_prop[0] = sc.spearmanr(v1, v2)[0]  # sc.pearsonr(v1, v2)[0]
        img_prop[3] = 0.
        img_prop[4] = np.median(self.relative_errors_scalar(v1, v2))
    
        return img_prop

    # Metrics for flow reconstructions
    def metrics(self, v1_x, v1_y, v2_x, v2_y):
        # 0: Pearson correlation coefficient
        # 1: Mean abs. error
        # 2: Mean rel. error
        # 3: Normalized dot product
        # 4: RMSE
        img_prop = np.zeros(5)
    
        # Flatten
        v1_x = v1_x.flatten()
        v1_y = v1_y.flatten()
        v2_x = v2_x.flatten()
        v2_y = v2_y.flatten()
        # Dimensions
        n = float(len(v1_x))
    
        # Metrics
        img_prop[1] = np.mean(self.absolute_errors_vector(v1_x, v1_y, v2_x, v2_y))
        img_prop[2] = np.mean(self.relative_errors_vector(v1_x, v1_y, v2_x, v2_y))
        img_prop[0] = np.sum((v1_x * v2_x) + (v1_y * v2_y)) / np.sqrt(
            np.sum(v1_x ** 2 + v1_y ** 2) * np.sum(v2_x ** 2 + v2_y ** 2))
        img_prop[3] = (1.0 / n) * np.sum(self.cosine_similarity_vector(v1_x, v1_y, v2_x, v2_y))
        # img_prop[4] = np.sqrt(np.mean((v1_x - v2_x) ** 2 + (v1_y - v2_y) ** 2))
        img_prop[4] = np.median(self.relative_errors_vector(v1_x, v1_y, v2_x, v2_y))
    
        return img_prop

    def close_plots(self):
    
        plt.close('all')

    def plot_scatterplots(self, filename_output, vx1, vy1, vx2, vy2,
                          title_label, x_label, y_label,
                          imgformat='png', alpha=1.0):
    
        # vh & B
        vh1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
        vh2 = np.sqrt(vx2 ** 2 + vy2 ** 2)
    
        # Font
        font_size = 13
        # Colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
    
        # Fits
        slope = np.zeros((1, 3))
        y0 = np.zeros((1, 3))
        slope[0, 0], y0[0, 0] = np.polyfit(vx1.flatten(), vx2.flatten(), 1)
        slope[0, 1], y0[0, 1] = np.polyfit(vy1.flatten(), vy2.flatten(), 1)
        slope[0, 2], y0[0, 2] = np.polyfit(vh1.flatten(), vh2.flatten(), 1)
    
        # Plot
        nrows = 1
        ncols = 3
        figsize_x = 5.25 * ncols
        figsize_y = 4.5 * nrows
        fig = plt.figure(figsize=(figsize_x, figsize_y))
    
        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                # Components
                if i == 0 and j == 0:
                    v1 = vx1
                    v2 = vx2
                elif i == 0 and j == 1:
                    v1 = vy1
                    v2 = vy2
                elif i == 0 and j == 2:
                    v1 = vh1
                    v2 = vh2
                # Limits
                if np.amin(v1) >= 0:
                    vmin = 0
                    vmax = np.amax([np.amax(v1), np.amax(v2)])
                else:
                    vmin = -np.amax([np.amax(np.abs(v1)), np.amax(np.abs(v2))])
                    vmax = -vmin
                # Subplot
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols + 0.0023 / 2 * ncols,
                                        right=float(j + 1) / ncols - 0.05 / ncols - 0.05 * (
                                                    1.0 / ncols - 0.10 / ncols) / 1.05 + 0.0023 / 2 * ncols,
                                        bottom=0.10 / nrows + float(i) / nrows, top=float(i + 1) / nrows - 0.10 / nrows,
                                        wspace=0.00)
                ax = fig.add_subplot(spec[:, :])
                ax.set_aspect(1)
                ax.scatter(v1.flatten(), v2.flatten(), c=colors[0], alpha=alpha,
                           marker='.', s=0.9)
                xvals = [vmin, vmax]
                yvals = [vmin, vmax]
                ax.plot(xvals, yvals, label="Reference (1:1)", color='black', linewidth=0.5)
                if y0[i, j] < 0:
                    ax.plot(xvals, [xvals[0] * slope[i, j] + y0[i, j], xvals[1] * slope[i, j] + y0[i, j]],
                            label=r'y = {0:.3f}x - {1:.3f} km/s'.format(slope[i, j], np.abs(y0[i, j])), color=colors[1],
                            linewidth=0.5)
                else:
                    ax.plot(xvals, [xvals[0] * slope[i, j] + y0[i, j], xvals[1] * slope[i, j] + y0[i, j]],
                            label=r'y = {0:.3f}x + {1:.3f} km/s'.format(slope[i, j], np.abs(y0[i, j])), color=colors[1],
                            linewidth=0.5)
                ax.grid(True, linewidth=0.25)
                ax.set_xlim(xvals)
                ax.set_ylim(yvals)
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=5.0)
                ax.set_xlabel(x_label[i][j], fontsize=font_size, labelpad=0.5)
                nbticks = 5
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbticks))
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbticks))
                # Title
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005)
                # Legend
                ax.legend(loc='upper left', fontsize=font_size - 4, numpoints=1, labelspacing=0.00, fancybox=False)
    
        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_metrics(self, metrics, filename, title_label, file_format='png'):
    
        # List of combinations of inputs
        inputs = [r'$I_c$',
                  r'$v_z$',
                  r'$B_z$',
                  r'$|\vec{B}_t|$',
                  r'$|\vec{B}|$',
                  r'$I_c$, $v_z$',
                  r'$B_z$, $I_c$',
                  r'$|\vec{B}_t|$, $I_c$',
                  r'$|\vec{B}|$, $I_c$',
                  r'$B_z$, $v_z$',
                  r'$|\vec{B}_t|$, $v_z$',
                  r'$|\vec{B}|$, $v_z$',
                  r'$B_{x, y}$',
                  r'$B_{x, y, z}$',
                  r'$B_{x, y}$, $I_c$',
                  r'$B_{x, y}$, $v_z$',
                  r'$|\vec{B}_t|$, $B_z$, $v_z$',
                  r'$|\vec{B}|$, $B_z$, $v_z$',
                  r'$|\vec{B}|$, $I_c$, $v_z$',
                  r'$|\vec{B}_t|$, $I_c$, $v_z$',
                  r'$B_z$, $I_c$, $v_z$',
                  r'$|\vec{B}|$, $B_z$, $I_c$, $v_z$',
                  r'$|\vec{B}_t|$, $B_z$, $I_c$, $v_z$',
                  r'$B_{x, y}$, $I_c$, $v_z$',
                  r'$B_{x, y, z}$, $v_z$',
                  r'$B_{x, y, z}$, $I_c$',
                  r'$B_{x, y, z}$, $I_c$, $v_z$',
                  r'$|\vec{B}_t|$, $B_z$',
                  r'$|\vec{B}|$, $B_z$',
                  r'$|\vec{B}_t|$, $B_z$, $I_c$']
        # Nb. of combinations
        nb_tests = len(inputs)
    
        # Properties
        font_size = 35
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
    
        fig, ax = plt.subplots(figsize=(13.5, 9))
        # ax.set_aspect(1)
        # x-axis
        x = np.linspace(1, nb_tests, num=nb_tests)
        xvals = [1, nb_tests]
        x_label = ''
        # y-axis
        yvals = [0.2, 1.0]
        y_label = 'Metric(s)'
        # Plot
        ax.plot(x, metrics[:, 0], label="Correlation Coefficient", marker='o', markersize=20, color=colors[0],
                linewidth=0.5)
        ax.plot(x, metrics[:, 3], label="Cosine Similarity", marker='o', markersize=20, color=colors[1], linewidth=0.5)
        ax.plot(x, metrics[:, 1], label=r"Mean Absolute Errors (km s$^{-1}$)", marker='o', markersize=20,
                color=colors[2], linewidth=0.5)
        ax.plot(x, metrics[:, 2], label=r"Mean Absolute Percentage Errors", marker='o', markersize=20, color=colors[3],
                linewidth=0.5)
        ax.grid(True, linewidth=0.25)
        ax.set_xlim(xvals)
        ax.set_ylim(yvals)
        ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=24)
        ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=14, pad=8)
        ax.set_ylabel(y_label, fontsize=font_size, labelpad=15.0)
        ax.set_xlabel(x_label, fontsize=font_size, labelpad=3.0)
        ax.xaxis.set_major_locator(plt.MaxNLocator(nb_tests - 1))
        ax.yaxis.set_major_locator(plt.MaxNLocator(9))
        ax.set_xticklabels(inputs, rotation=90, ha="right")
        # Title
        ax.set_title(title_label, fontsize=30, y=1.05)
        # Legend
        ax.legend(loc='best', fontsize=21, numpoints=1, labelspacing=0.05, fancybox=False)
        # Save
        plt.draw()
        plt.savefig(filename, format=file_format, dpi=300, edgecolor='white')
        # plt.close('all')

    def plot_scatterplots_2(self, filename_output, vx1, vy1, vx2, vy2,
                            title_label, x_label, y_label, imgformat='png',
                            alpha=1.0):
    
        # vh & B
        vh1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
        vh2 = np.sqrt(vx2 ** 2 + vy2 ** 2)
    
        # Font
        font_size = 13
        # Colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
    
        # Fits
        slope = np.zeros((1, 3))
        y0 = np.zeros((1, 3))
        slope[0, 0], y0[0, 0] = np.polyfit(vx1.flatten(), vx2.flatten(), 1)
        slope[0, 1], y0[0, 1] = np.polyfit(vy1.flatten(), vy2.flatten(), 1)
        slope[0, 2], y0[0, 2] = np.polyfit(vh1.flatten(), vh2.flatten(), 1)
    
        # Plot
        nrows = 1
        ncols = 2
        figsize_x = 5.25 * ncols
        figsize_y = 4.5 * nrows
        fig = plt.figure(figsize=(figsize_x, figsize_y))
    
        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                # Components
                if i == 0 and j == 0:
                    v1 = vx1
                    v2 = vx2
                elif i == 0 and j == 1:
                    v1 = vy1
                    v2 = vy2
                elif i == 0 and j == 2:
                    v1 = vh1
                    v2 = vh2
                # Limits
                if np.amin(v1) >= 0:
                    vmin = 0
                    vmax = np.amax([np.amax(v1), np.amax(v2)])
                else:
                    vmin = -np.amax(
                        [np.amax(np.abs(vx1)), np.amax(np.abs(vx2)), np.amax(np.abs(vy1)), np.amax(np.abs(vy2))])
                    vmax = -vmin
                # Subplot
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols + 0.0023,
                                        right=float(j + 1) / ncols - 0.05 / ncols - 0.05 * 0.45 / 1.05 + 0.0023,
                                        bottom=0.10 / nrows + float(i) / nrows, top=float(i + 1) / nrows - 0.10 / nrows,
                                        wspace=0.00)
                ax = fig.add_subplot(spec[:, :])
                ax.set_aspect(1)
                ax.scatter(v1.flatten(), v2.flatten(), c=colors[0], alpha=alpha,
                           marker='.', s=0.9)
                xvals = [vmin, vmax]
                yvals = [vmin, vmax]
                ax.plot(xvals, yvals, label="Reference (1:1)", color='black', linewidth=0.5)
                if y0[i, j] < 0:
                    ax.plot(xvals, [xvals[0] * slope[i, j] + y0[i, j], xvals[1] * slope[i, j] + y0[i, j]],
                            label=r'y = {0:.3f}x - {1:.3f} km/s'.format(slope[i, j], np.abs(y0[i, j])), color=colors[1],
                            linewidth=0.5)
                else:
                    ax.plot(xvals, [xvals[0] * slope[i, j] + y0[i, j], xvals[1] * slope[i, j] + y0[i, j]],
                            label=r'y = {0:.3f}x + {1:.3f} km/s'.format(slope[i, j], np.abs(y0[i, j])), color=colors[1],
                            linewidth=0.5)
                ax.grid(True, linewidth=0.25)
                ax.set_xlim(xvals)
                ax.set_ylim(yvals)
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=5.0)
                ax.set_xlabel(x_label[i][j], fontsize=font_size, labelpad=0.5)
                nbticks = 6
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbticks))
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbticks))
                # Title
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005)
                # Legend
                ax.legend(loc='upper left', fontsize=8, numpoints=1, labelspacing=0.00, fancybox=False)
    
        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_scatterplots_density(self, filename_output, vx1, vy1, vx2, vy2, vh1, vh2,
                                  title_label, x_label, y_label, colorbar_label, imgformat='png',
                                  alpha=1.0):
    
        # vh & B
        # vh1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
        # vh2 = np.sqrt(vx2 ** 2 + vy2 ** 2)
        # vh1 = self.divergence(vx1, vy1)/self.dx*1000.
        # vh2 = self.divergence(vx2, vy2) / self.dx*1000.
    
        # Font
        font_size = 13
        # Colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        # "Viridis-like" colormap with white background
        white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
            (0, '#ffffff'),
            (1e-20, '#440053'),
            (0.2, '#404388'),
            (0.4, '#2a788e'),
            (0.6, '#21a784'),
            (0.8, '#78d151'),
            (1, '#fde624'),
        ], N=256)
    
        # Fits
        slope = np.zeros((1, 3))
        y0 = np.zeros((1, 3))
        slope[0, 0], y0[0, 0] = np.polyfit(vx1.flatten(), vx2.flatten(), 1)
        slope[0, 1], y0[0, 1] = np.polyfit(vy1.flatten(), vy2.flatten(), 1)
        slope[0, 2], y0[0, 2] = np.polyfit(vh1.flatten(), vh2.flatten(), 1)
    
        # Plot
        nrows = 1
        ncols = 3
        figsize_x = 5.25 * ncols
        figsize_y = 4.5 * nrows
        fig = plt.figure(figsize=(figsize_x, figsize_y))
    
        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                # Components
                if i == 0 and j == 0:
                    v1 = vx1
                    v2 = vx2
                elif i == 0 and j == 1:
                    v1 = vy1
                    v2 = vy2
                elif i == 0 and j == 2:
                    v1 = vh1
                    v2 = vh2
                # Limits
                if np.amin(v1) >= 0:
                    vmin = 0
                    vmax = np.amax([np.amax(v1), np.amax(v2)])
                else:
                    vmin = -np.amax(
                        [np.amax(np.abs(v1)), np.amax(np.abs(v2))])
                    vmax = -vmin
                metrics = self.metrics_scalar(v1, v2)
                # Subplot
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols + 0.0023,
                                        right=float(j + 1) / ncols - 0.05 / ncols - 0.05 * 0.45 / 1.05 + 0.0023,
                                        bottom=0.10 / nrows + float(i) / nrows, top=float(i + 1) / nrows - 0.10 / nrows,
                                        wspace=0.00)
                ax = fig.add_subplot(spec[:, :], projection='scatter_density')
                ax.set_aspect(1)
                I = ax.scatter_density(v1.flatten(), v2.flatten(), cmap=white_viridis)
                xvals = [vmin, vmax]
                yvals = [vmin, vmax]
                ax.plot(xvals, yvals, label="Reference (1:1)", color='black', linewidth=0.5)
                if j == 2:
                    units = r'[10$^{-3}$ s$^{-1}$]'
                    # units = r'[ks$^{-1}$]'
                else:
                    units = r'[km s$^{-1}$]'
                if y0[i, j] < 0:
                    ax.plot(xvals, [xvals[0] * slope[i, j] + y0[i, j], xvals[1] * slope[i, j] + y0[i, j]],
                            label=r'y = {0:.3f}x - {1:.3f} '.format(slope[i, j], np.abs(y0[i, j]))+units, color=colors[1],
                            linewidth=0.5)
                else:
                    ax.plot(xvals, [xvals[0] * slope[i, j] + y0[i, j], xvals[1] * slope[i, j] + y0[i, j]],
                            label=r'y = {0:.3f}x + {1:.3f} '.format(slope[i, j], np.abs(y0[i, j]))+units, color=colors[1],
                            linewidth=0.5)
                plt.plot([], [], ' ', label=r"Spearman: {0:.3f}".format(metrics[0]))
                plt.plot([], [], ' ', label=r"MAE: {0:.3f} ".format(metrics[1])+units)
                plt.plot([], [], ' ', label=r"MAPE: {0:.3f} [\%]".format(100.*metrics[4]))
                ax.grid(True, linewidth=0.25)
                ax.set_xlim(xvals)
                ax.set_ylim(yvals)
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                if vmax > 10:
                    print('Test')
                    ax.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=0.0)
                else:
                    ax.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=5.0)
                ax.set_xlabel(x_label[i][j], fontsize=font_size, labelpad=0.5)
                nbticks = 6
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbticks))
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbticks))
                # Title
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005)
                # Legend
                ax.legend(loc='upper left', fontsize=8, numpoints=1, labelspacing=0.2, fancybox=False)
                # Colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0)
                cb = colorbar(I, extend='neither', cax=cax)
                cb.ax.tick_params(axis='y', direction='out', labelsize=font_size, width=1, length=2.5)
                cb.set_label(colorbar_label[i][j], labelpad=20.0, rotation=270, size=font_size)
    
        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_scatterplot(self, filename_output, vx1, vx2,
                         title_label, x_label, y_label, imgformat='png',
                         alpha=1.0):
    
        # vh & B
        # vh1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
        # vh2 = np.sqrt(vx2 ** 2 + vy2 ** 2)
    
        # Font
        font_size = 13
        # Colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']
    
        # Fits
        slope = np.zeros((1, 3))
        y0 = np.zeros((1, 3))
        slope[0, 0], y0[0, 0] = np.polyfit(vx1.flatten(), vx2.flatten(), 1)
        # slope[0, 1], y0[0, 1] = np.polyfit(vy1.flatten(), vy2.flatten(), 1)
        # slope[0, 2], y0[0, 2] = np.polyfit(vh1.flatten(), vh2.flatten(), 1)
    
        # Plot
        nrows = 1
        ncols = 1
        figsize_x = 5.25 * ncols
        figsize_y = 4.5 * nrows
        fig = plt.figure(figsize=(figsize_x, figsize_y))
    
        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                # Components
                if i == 0 and j == 0:
                    v1 = vx1
                    v2 = vx2
                elif i == 0 and j == 1:
                    v1 = vy1
                    v2 = vy2
                elif i == 0 and j == 2:
                    v1 = vh1
                    v2 = vh2
                # Limits
                if np.amin(v1) >= 0:
                    vmin = 0
                    vmax = np.amax([np.amax(v1), np.amax(v2)])
                else:
                    vmin = -np.amax([np.amax(np.abs(vx1)), np.amax(np.abs(vx2))])
                    vmax = -vmin
                # Subplot
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols + 0.0023,
                                        right=float(j + 1) / ncols - 0.05 / ncols - 0.05 * 0.45 / 1.05 + 0.0023,
                                        bottom=0.10 / nrows + float(i) / nrows, top=float(i + 1) / nrows - 0.10 / nrows,
                                        wspace=0.00)
                ax = fig.add_subplot(spec[:, :])
                ax.set_aspect(1)
                ax.scatter(v1.flatten(), v2.flatten(), c=colors[0], alpha=alpha,
                           marker='.', s=0.9)
                xvals = [vmin, vmax]
                yvals = [vmin, vmax]
                ax.plot(xvals, yvals, label="Reference (1:1)", color='black', linewidth=0.5)
                if y0[i, j] < 0:
                    ax.plot(xvals, [xvals[0] * slope[i, j] + y0[i, j], xvals[1] * slope[i, j] + y0[i, j]],
                            label=r'y = {0:.3f}x - {1:.3f} km/s'.format(slope[i, j], np.abs(y0[i, j])), color=colors[1],
                            linewidth=0.5)
                else:
                    ax.plot(xvals, [xvals[0] * slope[i, j] + y0[i, j], xvals[1] * slope[i, j] + y0[i, j]],
                            label=r'y = {0:.3f}x + {1:.3f} km/s'.format(slope[i, j], np.abs(y0[i, j])), color=colors[1],
                            linewidth=0.5)
                ax.grid(True, linewidth=0.25)
                ax.set_xlim(xvals)
                ax.set_ylim(yvals)
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=5.0)
                ax.set_xlabel(x_label[i][j], fontsize=font_size, labelpad=0.5)
                nbticks = 6
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbticks))
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbticks))
                # Title
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005)
                # Legend
                ax.legend(loc='upper left', fontsize=8, numpoints=1, labelspacing=0.00, fancybox=False)
    
        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_scatterplot_density1(self, filename_output, vx1, vy1, vx2, vy2, div1, div2, min1, max1, min2, max2,
                                  title_label, x_label, y_label, colorbar_label, imgformat='png',
                                  alpha=1.0, div=False):

        # vh & B
        # vh1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
        # vh2 = np.sqrt(vx2 ** 2 + vy2 ** 2)

        # Font
        font_size = 13
        # Colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        # "Viridis-like" colormap with white background
        white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
            (0, '#ffffff'),
            (1e-20, '#440053'),
            (0.2, '#404388'),
            (0.4, '#2a788e'),
            (0.6, '#21a784'),
            (0.8, '#78d151'),
            (1, '#fde624'),
        ], N=256)

        ny = (vx1.shape)[0]
        vv1 = np.zeros((ny, 2))
        vv2 = np.zeros((ny, 2))
        vv1[:, 0] = vx1
        vv1[:, 1] = vy1
        vv1 = vv1.flatten()
        vv2[:, 0] = vx2
        vv2[:, 1] = vy2
        vv2 = vv2.flatten()
        # Fits
        slope = np.zeros((1, 2))
        y0 = np.zeros((1, 2))
        slope[0, 0], y0[0, 0] = np.polyfit(vv1.flatten(), vv2.flatten(), 1)
        slope[0, 1], y0[0, 1] = np.polyfit(div1.flatten(), div2.flatten(), 1)
        # slope[0, 2], y0[0, 2] = np.polyfit(vh1.flatten(), vh2.flatten(), 1)

        # Plot
        nrows = 1
        ncols = 1
        figsize_x = 5.25 * ncols
        figsize_y = 4.5 * nrows
        fig = plt.figure(figsize=(figsize_x, figsize_y))

        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                # Components
                if div == False:
                    v1 = vv1
                    v2 = vv2
                    cmin = min1
                    cmax = max1
                else:
                    v1 = div1
                    v2 = div2
                    cmin = min2
                    cmax = max2
                # Limits
                if np.amin(v1) >= 0:
                    vmin = 0
                    vmax = np.amax([np.amax(v1), np.amax(v2)])
                else:
                    vmin = -np.amax([np.amax(np.abs(v1)), np.amax(np.abs(v2))])
                    vmax = -vmin
                # Subplot
                # metrics = self.metrics_scalar(v1, v2)
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols + 0.0023,
                                        right=float(j + 1) / ncols - 0.05 / ncols - 0.05 * 0.45 / 1.05 + 0.0023,
                                        bottom=0.10 / nrows + float(i) / nrows, top=float(i + 1) / nrows - 0.10 / nrows,
                                        wspace=0.00)
                ax = fig.add_subplot(spec[:, :], projection='scatter_density')
                ax.set_aspect(1)
                I = ax.scatter_density(v1.flatten(), v2.flatten(), cmap=white_viridis) #, vmin=cmin, vmax=cmax)
                # ax.scatter(v1.flatten(), v2.flatten(), c=colors[0], alpha=alpha,
                #            marker='.', s=0.9)
                xvals = [vmin, vmax]
                yvals = [vmin, vmax]
                ax.plot(xvals, yvals, label="Reference (1:1)", color='black', linewidth=0.5)
                if y0[i, j] < 0:
                    units = r' [km s$^{-1}$]'
                    ax.plot(xvals, [xvals[0] * slope[i, j] + y0[i, j], xvals[1] * slope[i, j] + y0[i, j]],
                            label=r'y = {0:.3f}x - {1:.3f}'.format(slope[i, j], np.abs(y0[i, j]))+units,
                            color=colors[1],
                            linewidth=0.5)
                else:
                    ax.plot(xvals, [xvals[0] * slope[i, j] + y0[i, j], xvals[1] * slope[i, j] + y0[i, j]],
                            label=r'y = {0:.3f}x + {1:.3f} [km/s]'.format(slope[i, j], np.abs(y0[i, j])),
                            color=colors[1],
                            linewidth=0.5)
                ax.grid(True, linewidth=0.25)
                ax.set_xlim(xvals)
                ax.set_ylim(yvals)
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=5.0)
                ax.set_xlabel(x_label[i][j], fontsize=font_size, labelpad=0.5)
                if div == False:
                    units = r' [km s$^{-1}$]'
                    metrics = self.metrics(vx1, vy1, vx2, vy2)
                    plt.plot([], [], ' ', label=r"Spearman: {0:.3f}".format(metrics[0]))
                    plt.plot([], [], ' ', label=r"MAE: {0:.3f}".format(metrics[1])+units)
                    plt.plot([], [], ' ', label=r"MAPE: {0:.3f} [\%]".format(100. * metrics[4]))
                    plt.plot([], [], ' ', label=r"CSI: {0:.3f}".format(metrics[3]))
                else:
                    units = r' [10$^{-2}$ s$^{-1}$]'
                    metrics = self.metrics_scalar(v1, v2)
                    plt.plot([], [], ' ', label=r"Spearman: {0:.3f}".format(metrics[0]))
                    plt.plot([], [], ' ', label=r"MAE: {0:.3f}".format(metrics[1])+units)
                    plt.plot([], [], ' ', label=r"MAPE: {0:.3f} [\%]".format(100. * metrics[4]))
                nbticks = 6
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbticks))
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbticks))
                # Title
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005)
                # Legend
                ax.legend(loc='upper left', fontsize=8, numpoints=1, labelspacing=0.00, fancybox=False)
                # Colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0)
                cb = colorbar(I, extend='neither', cax=cax)
                cb.ax.tick_params(axis='y', direction='out', labelsize=font_size, width=1, length=2.5)
                cb.set_label(colorbar_label[i][j], labelpad=20.0, rotation=270, size=font_size)

        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')


    def plot_scatterplot_density2(self, filename_output, vx1, vy1, vx2, vy2, div1, div2, min1, max1, min2, max2,
                                  title_label, x_label, y_label, colorbar_label, imgformat='png',
                                  alpha=1.0):

        # vh & B
        # vh1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
        # vh2 = np.sqrt(vx2 ** 2 + vy2 ** 2)

        # Font
        font_size = 13
        # Colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        # "Viridis-like" colormap with white background
        white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
            (0, '#ffffff'),
            (1e-20, '#440053'),
            (0.2, '#404388'),
            (0.4, '#2a788e'),
            (0.6, '#21a784'),
            (0.8, '#78d151'),
            (1, '#fde624'),
        ], N=256)

        ny = (vx1.shape)[0]
        vv1 = np.zeros((ny, 2))
        vv2 = np.zeros((ny, 2))
        vv1[:, 0] = vx1
        vv1[:, 1] = vy1
        vv1 = vv1.flatten()
        vv2[:, 0] = vx2
        vv2[:, 1] = vy2
        vv2 = vv2.flatten()
        # Fits
        slope = np.zeros((1, 2))
        y0 = np.zeros((1, 2))
        slope[0, 0], y0[0, 0] = np.polyfit(vv1.flatten(), vv2.flatten(), 1)
        slope[0, 1], y0[0, 1] = np.polyfit(div1.flatten(), div2.flatten(), 1)
        # slope[0, 2], y0[0, 2] = np.polyfit(vh1.flatten(), vh2.flatten(), 1)

        # Plot
        nrows = 1
        ncols = 2
        figsize_x = 5.25 * ncols
        figsize_y = 4.5 * nrows
        fig = plt.figure(figsize=(figsize_x, figsize_y))

        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                # Components
                if i == 0 and j == 0:
                    v1 = vv1
                    v2 = vv2
                    cmin = min1
                    cmax = max1
                elif i == 0 and j == 1:
                    v1 = div1
                    v2 = div2
                    cmin = min2
                    cmax = max2
                # Limits
                if np.amin(v1) >= 0:
                    vmin = 0
                    vmax = np.amax([np.amax(v1), np.amax(v2)])
                else:
                    vmin = -np.amax([np.amax(np.abs(v1)), np.amax(np.abs(v2))])
                    vmax = -vmin
                # Subplot
                # metrics = self.metrics_scalar(v1, v2)
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols + 0.0023,
                                        right=float(j + 1) / ncols - 0.05 / ncols - 0.05 * 0.45 / 1.05 + 0.0023,
                                        bottom=0.10 / nrows + float(i) / nrows, top=float(i + 1) / nrows - 0.10 / nrows,
                                        wspace=0.00)
                ax = fig.add_subplot(spec[:, :], projection='scatter_density')
                ax.set_aspect(1)
                I = ax.scatter_density(v1.flatten(), v2.flatten(), cmap=white_viridis) #, vmin=cmin, vmax=cmax)
                # ax.scatter(v1.flatten(), v2.flatten(), c=colors[0], alpha=alpha,
                #            marker='.', s=0.9)
                xvals = [vmin, vmax]
                yvals = [vmin, vmax]
                ax.plot(xvals, yvals, label="Reference (1:1)", color='black', linewidth=0.5)
                if y0[i, j] < 0:
                    units = r' [km s$^{-1}$]'
                    ax.plot(xvals, [xvals[0] * slope[i, j] + y0[i, j], xvals[1] * slope[i, j] + y0[i, j]],
                            label=r'y = {0:.3f}x - {1:.3f}'.format(slope[i, j], np.abs(y0[i, j]))+units,
                            color=colors[1],
                            linewidth=0.5)
                else:
                    ax.plot(xvals, [xvals[0] * slope[i, j] + y0[i, j], xvals[1] * slope[i, j] + y0[i, j]],
                            label=r'y = {0:.3f}x + {1:.3f} [km/s]'.format(slope[i, j], np.abs(y0[i, j])),
                            color=colors[1],
                            linewidth=0.5)
                ax.grid(True, linewidth=0.25)
                ax.set_xlim(xvals)
                ax.set_ylim(yvals)
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=5.0)
                ax.set_xlabel(x_label[i][j], fontsize=font_size, labelpad=0.5)
                if i == 0 and j == 0:
                    units = r' [km s$^{-1}$]'
                    metrics = self.metrics(vx1, vy1, vx2, vy2)
                    plt.plot([], [], ' ', label=r"Spearman: {0:.3f}".format(metrics[0]))
                    plt.plot([], [], ' ', label=r"MAE: {0:.3f}".format(metrics[1])+units)
                    plt.plot([], [], ' ', label=r"MAPE: {0:.3f} [\%]".format(100. * metrics[4]))
                    plt.plot([], [], ' ', label=r"CSI: {0:.3f}".format(metrics[3]))
                else:
                    units = r' [10$^{-2}$ s$^{-1}$]'
                    metrics = self.metrics_scalar(v1, v2)
                    plt.plot([], [], ' ', label=r"Spearman: {0:.3f}".format(metrics[0]))
                    plt.plot([], [], ' ', label=r"MAE: {0:.3f}".format(metrics[1])+units)
                    plt.plot([], [], ' ', label=r"MAPE: {0:.3f} [\%]".format(100. * metrics[4]))
                nbticks = 6
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbticks))
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbticks))
                # Title
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005)
                # Legend
                ax.legend(loc='upper left', fontsize=8, numpoints=1, labelspacing=0.00, fancybox=False)
                # Colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0)
                cb = colorbar(I, extend='neither', cax=cax)
                cb.ax.tick_params(axis='y', direction='out', labelsize=font_size, width=1, length=2.5)
                cb.set_label(colorbar_label[i][j], labelpad=20.0, rotation=270, size=font_size)

        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_scatterplot_density(self, filename_output, vx1, vx2,
                                 title_label, x_label, y_label, colorbar_label, imgformat='png',
                                 alpha=1.0):
    
        # vh & B
        # vh1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
        # vh2 = np.sqrt(vx2 ** 2 + vy2 ** 2)
    
        # Font
        font_size = 13
        # Colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        # "Viridis-like" colormap with white background
        white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
            (0, '#ffffff'),
            (1e-20, '#440053'),
            (0.2, '#404388'),
            (0.4, '#2a788e'),
            (0.6, '#21a784'),
            (0.8, '#78d151'),
            (1, '#fde624'),
        ], N=256)
    
        # Fits
        slope = np.zeros((1, 3))
        y0 = np.zeros((1, 3))
        slope[0, 0], y0[0, 0] = np.polyfit(vx1.flatten(), vx2.flatten(), 1)
        # slope[0, 1], y0[0, 1] = np.polyfit(vy1.flatten(), vy2.flatten(), 1)
        # slope[0, 2], y0[0, 2] = np.polyfit(vh1.flatten(), vh2.flatten(), 1)
    
        # Plot
        nrows = 1
        ncols = 1
        figsize_x = 5.25 * ncols
        figsize_y = 4.5 * nrows
        fig = plt.figure(figsize=(figsize_x, figsize_y))
    
        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                # Components
                if i == 0 and j == 0:
                    v1 = vx1
                    v2 = vx2
                elif i == 0 and j == 1:
                    v1 = vy1
                    v2 = vy2
                elif i == 0 and j == 2:
                    v1 = vh1
                    v2 = vh2
                # Limits
                if np.amin(v1) >= 0:
                    vmin = 0
                    vmax = np.amax([np.amax(v1), np.amax(v2)])
                else:
                    vmin = -np.amax([np.amax(np.abs(vx1)), np.amax(np.abs(vx2))])
                    vmax = -vmin
                # Subplot
                metrics = self.metrics_scalar(v1, v2)
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols + 0.0023,
                                        right=float(j + 1) / ncols - 0.05 / ncols - 0.05 * 0.45 / 1.05 + 0.0023,
                                        bottom=0.10 / nrows + float(i) / nrows, top=float(i + 1) / nrows - 0.10 / nrows,
                                        wspace=0.00)
                ax = fig.add_subplot(spec[:, :], projection='scatter_density')
                ax.set_aspect(1)
                I = ax.scatter_density(v1.flatten(), v2.flatten(), cmap=white_viridis)
                # ax.scatter(v1.flatten(), v2.flatten(), c=colors[0], alpha=alpha,
                #            marker='.', s=0.9)
                xvals = [vmin, vmax]
                yvals = [vmin, vmax]
                ax.plot(xvals, yvals, label="Reference (1:1)", color='black', linewidth=0.5)
                if y0[i, j] < 0:
                    ax.plot(xvals, [xvals[0] * slope[i, j] + y0[i, j], xvals[1] * slope[i, j] + y0[i, j]],
                            label=r'y = {0:.3f}x - {1:.3f} [km s$^{-1}$]'.format(slope[i, j], np.abs(y0[i, j])), color=colors[1],
                            linewidth=0.5)
                else:
                    ax.plot(xvals, [xvals[0] * slope[i, j] + y0[i, j], xvals[1] * slope[i, j] + y0[i, j]],
                            label=r'y = {0:.3f}x + {1:.3f} [km s$^{-1}$]'.format(slope[i, j], np.abs(y0[i, j])), color=colors[1],
                            linewidth=0.5)
                ax.grid(True, linewidth=0.25)
                ax.set_xlim(xvals)
                ax.set_ylim(yvals)
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=5.0)
                ax.set_xlabel(x_label[i][j], fontsize=font_size, labelpad=0.5)
                plt.plot([], [], ' ', label=r"Spearman: {0:.3f}".format(metrics[0]))
                plt.plot([], [], ' ', label=r"MAE: {0:.3f} [km s$^{-1}$]".format(metrics[1]))
                plt.plot([], [], ' ', label=r"MAPE: {0:.3f} [\%]".format(100. * metrics[4]))
                nbticks = 6
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbticks))
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbticks))
                # Title
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005)
                # Legend
                ax.legend(loc='upper left', fontsize=8, numpoints=1, labelspacing=0.00, fancybox=False)
                # Colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0)
                cb = colorbar(I, extend='neither', cax=cax)
                cb.ax.tick_params(axis='y', direction='out', labelsize=font_size, width=1, length=2.5)
                cb.set_label(colorbar_label[i][j], labelpad=20.0, rotation=270, size=font_size)
    
        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_scatterplot_density5(self, filename_output, vx1, vx2,
                                 title_label, x_label, y_label, colorbar_label, imgformat='png',
                                 alpha=1.0):
    
        # vh & B
        # vh1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
        # vh2 = np.sqrt(vx2 ** 2 + vy2 ** 2)
    
        # Font
        font_size = 13
        # Colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        # "Viridis-like" colormap with white background
        white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
            (0, '#ffffff'),
            (1e-20, '#440053'),
            (0.2, '#404388'),
            (0.4, '#2a788e'),
            (0.6, '#21a784'),
            (0.8, '#78d151'),
            (1, '#fde624'),
        ], N=256)
    
        # Fits
        slope = np.zeros((1, 3))
        y0 = np.zeros((1, 3))
        # slope[0, 0], y0[0, 0] = np.polyfit(vx1.flatten(), vx2.flatten(), 1)
        # slope[0, 1], y0[0, 1] = np.polyfit(vy1.flatten(), vy2.flatten(), 1)
        # slope[0, 2], y0[0, 2] = np.polyfit(vh1.flatten(), vh2.flatten(), 1)
    
        # Plot
        nrows = 1
        ncols = 1
        figsize_x = 5.25 * ncols
        figsize_y = 4.5 * nrows
        fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout = False)
    
        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                # Components
                if i == 0 and j == 0:
                    v1 = vx1
                    v2 = vx2
                elif i == 0 and j == 1:
                    v1 = vy1
                    v2 = vy2
                elif i == 0 and j == 2:
                    v1 = vh1
                    v2 = vh2
                xmin = 0.8  # np.nanmin(v1)
                xmax = np.nanmax(v1)
                vmin = -np.nanmax(np.abs(v2))
                vmax = -vmin
                # Subplot
                metrics = self.metrics_scalar(v1, v2)
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols + 0.0023,
                                        right=float(j + 1) / ncols - 0.05 / ncols - 0.05 * 0.45 / 1.05 + 0.0023,
                                        bottom=0.10 / nrows + float(i) / nrows, top=float(i + 1) / nrows - 0.10 / nrows,
                                        wspace=0.00)
                ax = fig.add_subplot(spec[:, :], projection='scatter_density')
                # ax.set_aspect(1)
                ax.set_aspect((xmax-xmin)/(vmax-vmin))
                I = ax.scatter_density(v1.flatten(), v2.flatten(), cmap=white_viridis)
                # ax.scatter(v1.flatten(), v2.flatten(), c=colors[0], alpha=alpha,
                #            marker='.', s=0.9)
                xvals = [xmin, xmax]
                yvals = [vmin, vmax]
                ax.plot(xvals, yvals, label="Reference (1:1)", color='black', linewidth=0.5)
                #if y0[i, j] < 0:
                #    ax.plot(xvals, [xvals[0] * slope[i, j] + y0[i, j], xvals[1] * slope[i, j] + y0[i, j]],
                #            label=r'y = {0:.3f}x - {1:.3f} [km s$^{-1}$]'.format(slope[i, j], np.abs(y0[i, j])), color=colors[1],
                #            linewidth=0.5)
                #else:
                #    ax.plot(xvals, [xvals[0] * slope[i, j] + y0[i, j], xvals[1] * slope[i, j] + y0[i, j]],
                #            label=r'y = {0:.3f}x + {1:.3f} [km s$^{-1}$]'.format(slope[i, j], np.abs(y0[i, j])), color=colors[1],
                #            linewidth=0.5)
                ax.grid(True, linewidth=0.25)
                ax.set_xlim(xvals)
                ax.set_ylim(yvals)
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=5.0)
                ax.set_xlabel(x_label[i][j], fontsize=font_size, labelpad=0.5)
                plt.plot([], [], ' ', label=r"Spearman: {0:.3f}".format(metrics[0]))
                #plt.plot([], [], ' ', label=r"MAE: {0:.3f} [km s$^{-1}$]".format(metrics[1]))
                #plt.plot([], [], ' ', label=r"MAPE: {0:.3f} [\%]".format(100. * metrics[4]))
                nbticks = 6
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbticks))
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbticks))
                # Title
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005)
                # Legend
                # ax.legend(loc='upper left', fontsize=8, numpoints=1, labelspacing=0.00, fancybox=False)
                # Colorbar
                #divider = make_axes_locatable(ax)
                #cax = divider.append_axes("right", size="5%", pad=0)
                #cb = colorbar(I, extend='neither', cax=cax)
                #cb.ax.tick_params(axis='y', direction='out', labelsize=font_size, width=1, length=2.5)
                #cb.set_label(colorbar_label[i][j], labelpad=20.0, rotation=270, size=font_size)
    
        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_evolution(self, filename_output, bza, metrics,
                       title_label, x_label, y_label, legend_label, imgformat='png'):

        # vh & B
        # vh1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
        # vh2 = np.sqrt(vx2 ** 2 + vy2 ** 2)

        # Font
        font_size = 13
        # Colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        # Plot
        nrows = 1
        ncols = 1
        figsize_x = 5.25 * ncols
        figsize_y = 4.5 * nrows
        fig = plt.figure(figsize=(figsize_x, figsize_y))

        nt, nm, nz = metrics.shape

        x = np.zeros((nt))
        for i in range(nt):
            x[i] = i

        xmin = np.amin(x)
        xmax = np.amax(x)

        ymin = 0.7  # np.amin(metrics[:, 0, :])
        ymax = 0.95  # np.amax(metrics[:, 0, :])

        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                # Subplot
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols - 0.005,
                                        right=float(j + 1) / ncols - 0.05 / ncols - 0.05 * 0.45 / 1.05 - 0.005,
                                        bottom=0.10 / nrows + float(i) / nrows, top=float(i + 1) / nrows - 0.10 / nrows,
                                        wspace=0.00)
                ax = fig.add_subplot(spec[:, :])  # , projection='scatter_density')

                dx = xmax-xmin
                # dy = np.log10(ymax)-np.log10(ymin)
                dy = ymax - ymin
                print(dy, dx)
                ax.set_aspect(dx/dy)

                # ax.plot(x, bza.flatten(), c=colors[4], markersize=20, label=legend_label[4])


                for k in range(nz):
                    ax.plot(x, metrics[:, 0, k].flatten(), c=colors[k], markersize=20, label=legend_label[k])

                #for k in range(nz):
                #    ax.plot(x, metrics[:, k, 3].flatten(), c=colors[k], markersize=20, label=legend_label[k])

                # ax.plot(x, bza.flatten(), c=colors[4], markersize=20, label=legend_label[4])

                ax.grid(True, linewidth=0.25)
                # ax.set_xscale('log')
                # ax.set_yscale('log')
                # Axis
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=5)
                ax.set_xlabel(x_label[i][j], fontsize=font_size, labelpad=0.5)
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                # Title
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005)
                # Legend
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])
                ax.legend(loc='best', fontsize=8, numpoints=1)


        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_evolution2(self, filename_output, bza, metrics,
                        title_label, x_label, y_label, legend_label, imgformat='png'):

        # vh & B
        # vh1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
        # vh2 = np.sqrt(vx2 ** 2 + vy2 ** 2)

        # Font
        font_size = 13
        # Colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        # Plot
        nrows = 1
        ncols = 1
        figsize_x = 5.25 * ncols
        figsize_y = 4.5 * nrows
        fig = plt.figure(figsize=(figsize_x, figsize_y))

        nt, nm, nz = metrics.shape

        x = np.zeros((nt))
        for i in range(nt):
            x[i] = i

        xmin = np.amin(x)
        xmax = np.amax(x)

        ymin = 0.7  # np.amin(metrics[:, 0, :])
        ymax = 0.95  # np.amax(metrics[:, 0, :])

        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                # Subplot
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols - 0.005,
                                        right=float(j + 1) / ncols - 0.05 / ncols - 0.05 * 0.45 / 1.05 - 0.005,
                                        bottom=0.10 / nrows + float(i) / nrows, top=float(i + 1) / nrows - 0.10 / nrows,
                                        wspace=0.00)
                ax = fig.add_subplot(spec[:, :])  # , projection='scatter_density')
                # ax = fig.add_subplot()  # , projection='scatter_density')
                # ax2 = ax.twinx()

                dx = xmax-xmin
                # dy = np.log10(ymax)-np.log10(ymin)
                dy = ymax - ymin
                print(dy, dx)
                ax.set_aspect(dx/dy)

                # ax.plot(x, bza.flatten(), c=colors[4], markersize=20, label=legend_label[4])


                for k in range(nz):
                    ax.plot(x, metrics[:, 0, k].flatten(), c=colors[k], markersize=20, label=legend_label[k])

                #for k in range(nz):
                #    ax.plot(x, metrics[:, k, 3].flatten(), c=colors[k], markersize=20, label=legend_label[k])

                # ax2.plot(x, bza.flatten(), c=colors[4], markersize=20, label=legend_label[4])
                # ax2.plot(x, metrics[:, 0, 0].flatten(), c=colors[4], markersize=20)

                ax.grid(True, linewidth=0.25)
                # ax.set_xscale('log')
                # ax.set_yscale('log')
                # Axis
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=5)
                ax.set_xlabel(x_label[i][j], fontsize=font_size, labelpad=0.5)
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                # Title
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005)
                # Legend
                ax.set_xlim([xmin, xmax])
                # ax2.set_ylim([ymin, ymax])
                ax.set_ylim([ymin, ymax])

                ax.legend(loc='best', fontsize=8, numpoints=1)

                # ax2 = ax.twinx()
                # ax2.set_ylim([ymin, ymax])
                # ax2.plot(x, metrics[:, 0, 0].flatten(), c=colors[4], markersize=20)
                # ax2.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=5)
                ax2 = ax.secondary_yaxis('right')
                ax2.set_ylim([ymin, ymax])
                ax2.plot(x, bza.flatten(), c=colors[4], markersize=20, label=legend_label[4])

        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_evolution3(self, filename_output, bza, metrics,
                        title_label, x_label, y_label, legend_label, imgformat='png'):

        x = np.linspace(0, 1.6, 50) + 50.0

        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax2 = ax.twinx()

        XLIM = [50.0, 51.6]
        YLIM = [0.0, 1.1, 0.0, 11.0]

        ax.plot(x, np.sin(x - 50.0), 'b')
        ax2.plot(x, np.cos(x - 50.0) * 10., 'r')

        ax.set_xlim([XLIM[0], XLIM[1]])
        ax.set_ylim([YLIM[0], YLIM[1]])
        ax2.set_ylim([YLIM[2], YLIM[3]])

        ax.set_xticks(np.arange(XLIM[0], XLIM[1], 0.2))
        ax.set_yticks(np.arange(YLIM[0], YLIM[1] + 0.1, 0.1)[:-1])
        ax2.set_yticks(np.arange(YLIM[2], YLIM[3] + 1.0, 1.0))

        ax.grid(True, which='major', linestyle='solid')

        # ax.set_aspect((XLIM[1] - XLIM[0]) / (YLIM[1] - YLIM[0]))
        # ax2.set_aspect((XLIM[1] - XLIM[0]) / (YLIM[3] - YLIM[2]))
        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_evolution4(self, filename_output, bza, metrics,
                        title_label, x_label, y_label, legend_label, imgformat='png'):

        # vh & B
        # vh1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
        # vh2 = np.sqrt(vx2 ** 2 + vy2 ** 2)

        # Font
        font_size = 13
        # Colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        # Plot
        nrows = 1
        ncols = 1
        figsize_x = 5.25 * ncols
        figsize_y = 4.5 * nrows
        fig = plt.figure(figsize=(figsize_x, figsize_y))

        nt, nm, nz = metrics.shape

        x = np.zeros((nt))
        for i in range(nt):
            x[i] = i

        xmin = np.amin(x)
        xmax = np.amax(x)

        ymin = 0.7  # np.amin(metrics[:, 0, :])
        ymax = 0.95  # np.amax(metrics[:, 0, :])

        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                # Subplot
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols - 0.005,
                                        right=float(j + 1) / ncols - 0.05 / ncols - 0.05 * 0.45 / 1.05 - 0.005,
                                        bottom=0.10 / nrows + float(i) / nrows, top=float(i + 1) / nrows - 0.10 / nrows,
                                        wspace=0.00)
                ax = fig.add_subplot(spec[:, :])  # , projection='scatter_density')
                # ax = fig.add_subplot()  # , projection='scatter_density')
                # ax2 = ax.twinx()

                dx = xmax-xmin
                # dy = np.log10(ymax)-np.log10(ymin)
                dy = ymax - ymin
                print(dy, dx)
                # ax.set_aspect(dx/dy)

                # ax.plot(x, bza.flatten(), c=colors[4], markersize=20, label=legend_label[4])


                for k in range(nz):
                    ax.plot(x, metrics[:, 0, k].flatten(), c=colors[k], markersize=20, label=legend_label[k])

                #for k in range(nz):
                #    ax.plot(x, metrics[:, k, 3].flatten(), c=colors[k], markersize=20, label=legend_label[k])

                # ax2.plot(x, bza.flatten(), c=colors[4], markersize=20, label=legend_label[4])
                # ax2.plot(x, metrics[:, 0, 0].flatten(), c=colors[4], markersize=20)

                ax.grid(True, linewidth=0.25)
                # ax.set_xscale('log')
                # ax.set_yscale('log')
                # Axis
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=5)
                ax.set_xlabel(x_label[i][j], fontsize=font_size, labelpad=0.5)
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                # Title
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005)
                # Legend
                ax.set_xlim([xmin, xmax])
                # ax2.set_ylim([ymin, ymax])
                ax.set_ylim([ymin, ymax])

                ax.legend(loc='best', fontsize=8, numpoints=1)

                ax2 = ax.twinx()

                ax.set(adjustable='box-forced')
                ax2.set(adjustable='box-forced')

                ax2.plot(x, metrics[:, 0, 0].flatten(), c=colors[4], markersize=20)
                ax2.set_ylim([ymin, ymax])
                # ax2.set_aspect(2400*0.75)
                # print(dx/dy)
                ax2.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax2.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=5)
                # ax2 = ax.secondary_yaxis('right')
                # ax2.set_ylim([ymin, ymax])
                # ax2.plot(x, bza.flatten(), c=colors[4], markersize=20, label=legend_label[4])

        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_evolution5(self, filename_output, bza, metrics, nb,
                        title_label, x_label, y_label, y_label2, legend_label, imgformat='png'):

        # vh & B
        # vh1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
        # vh2 = np.sqrt(vx2 ** 2 + vy2 ** 2)

        # Font
        font_size = 13
        # Colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        # Plot
        nrows = 1
        ncols = 1
        figsize_x = 5.25 * ncols
        figsize_y = 4.5 * nrows
        fig = plt.figure(figsize=(figsize_x, figsize_y))

        nt, nm, nz = metrics.shape

        x = np.zeros((nt))
        for i in range(nt):
            x[i] = i

        xmin = np.amin(x)
        xmax = np.amax(x)

        #ymin = 0.99*np.amin(metrics[:, 0, :]) # 0.7  # np.amin(metrics[:, 0, :])
        #ymax = 0.95  # np.amax(metrics[:, 0, :])

        ymin = 0.5 * (np.amax(metrics[:, nb, :]) + np.amin(metrics[:, nb, :])) - 0.5 * 1.05 * (np.amax(metrics[:, nb, :]) - np.amin(metrics[:, nb, :]))
        ymax = 0.5 * (np.amax(metrics[:, nb, :]) + np.amin(metrics[:, nb, :])) + 0.5 * 1.05 * (np.amax(metrics[:, nb, :]) - np.amin(metrics[:, nb, :]))

        #ymin2 = np.amin(bza)
        #ymax2 = np.amax(bza)

        # ymin2 = 0.5*(np.amax(bza)+np.amin(bza))-0.5*1.05*(np.amax(bza)-np.amin(bza))
        # ymax2 = 0.5*(np.amax(bza)+np.amin(bza))+0.5*1.05*(np.amax(bza)-np.amin(bza))
        ymin2 = 0.5 * (0.8  + 0.4) - 0.5 * 1.05 * (0.8 - 0.4)
        ymax2 = 0.5 * (0.8  + 0.4) + 0.5 * 1.05 * (0.8 - 0.4)

        #ymin2 = 0.5 * (0.8 + 0.7) - 0.5 * 1.05 * (0.8 - 0.7)
        #ymax2 = 0.5 * (0.8 + 0.7) + 0.5 * 1.05 * (0.8 - 0.7)

        #if ymin2 < 0:
        #    ymin2 = 1.05*ymin2
        #else:
        #    ymin2 = 0.95*ymin2
        #if ymax2 < 0:
        #    ymax2 = 0.95 * ymax2
        #else:
        #    ymax2 = 1.05 * ymax2

        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                # Subplot
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols - 0.005,
                                        right=float(j + 1) / ncols - 0.05 / ncols - 0.05 * 0.45 / 1.05 - 0.005,
                                        bottom=0.10 / nrows + float(i) / nrows, top=float(i + 1) / nrows - 0.10 / nrows,
                                        wspace=0.00)
                ax = fig.add_subplot(spec[:, :])  # , projection='scatter_density')
                # ax = fig.add_subplot()  # , projection='scatter_density')
                # ax2 = ax.twinx()

                dx = xmax-xmin
                # dy = np.log10(ymax)-np.log10(ymin)
                dy = ymax - ymin
                dy2 = ymax2 - ymin2
                # print(dy, dx)
                ax.set_aspect(dx/dy)

                # ax.plot(x, bza.flatten(), c=colors[4], markersize=20, label=legend_label[4])


                for k in range(nz):
                    ax.plot(x, metrics[:, nb, k].flatten(), c=colors[k], markersize=20, label=legend_label[k])

                #for k in range(nz):
                #    ax.plot(x, metrics[:, k, 3].flatten(), c=colors[k], markersize=20, label=legend_label[k])

                # ax2.plot(x, bza.flatten(), c=colors[4], markersize=20, label=legend_label[4])
                # ax2.plot(x, metrics[:, 0, 0].flatten(), c=colors[4], markersize=20)

                ax.grid(True, linewidth=0.25)
                # ax.set_xscale('log')
                # ax.set_yscale('log')
                # Axis
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=5)
                ax.set_xlabel(x_label[i][j], fontsize=font_size, labelpad=0.5)
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                # Title
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005)
                # Legend
                ax.set_xlim([xmin, xmax])
                # ax2.set_ylim([ymin, ymax])
                ax.set_ylim([ymin, ymax])

                #ax2 = ax.twinx()

                #ax.set(adjustable='box-forced')
                #ax2.set(adjustable='box-forced')
                ax2 = fig.add_axes(ax.get_position())
                ax2.set_facecolor("None")
                # ax2.plot(x, metrics[:, 0, 0].flatten(), c=colors[4], markersize=20, label=legend_label[4])
                ax2.plot(x, bza.flatten(), c=colors[5], markersize=20, label=legend_label[5])
                ax2.tick_params(bottom=0, top=0, left=0, right=1,
                                labelbottom=0, labeltop=0, labelleft=0, labelright=1)
                ax2.set_xlim([xmin, xmax])
                ax2.set_ylim([ymin2, ymax2])
                ax2.set_aspect(dx/dy2)
                # print(dx/dy)
                ax2.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax2.yaxis.set_label_position("right")
                ax2.set_ylabel(y_label2[i][j], fontsize=font_size, labelpad=18, rotation=270)
                # ax2 = ax.secondary_yaxis('right')
                # ax2.set_ylim([350000., 400000.])
                # ax2.plot(x, bza.flatten(), c=colors[4], markersize=20, label=legend_label[4])

                lines, labels = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines + lines2, labels + labels2, loc=6, fontsize=8, numpoints=1)
                # ax.legend(loc='best', fontsize=8, numpoints=1)

        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_pixel_histogram(self, filename_output, bins, hist, xmin, xmax, ymin, ymax,
                             title_label, x_label, y_label, legend_label, imgformat='png'):

        # vh & B
        # vh1 = np.sqrt(vx1 ** 2 + vy1 ** 2)
        # vh2 = np.sqrt(vx2 ** 2 + vy2 ** 2)

        # Font
        font_size = 13
        # Colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        nz, _ = hist.shape

        # Plot
        nrows = 1
        ncols = 1
        figsize_x = 5.25 * ncols
        figsize_y = 4.5 * nrows
        fig = plt.figure(figsize=(figsize_x, figsize_y))

        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                # Subplot
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols - 0.005,
                                        right=float(j + 1) / ncols - 0.05 / ncols - 0.05 * 0.45 / 1.05 - 0.005,
                                        bottom=0.10 / nrows + float(i) / nrows, top=float(i + 1) / nrows - 0.10 / nrows,
                                        wspace=0.00)
                ax = fig.add_subplot(spec[:, :])  # , projection='scatter_density')

                dx = xmax-xmin
                dy = np.log10(ymax)-np.log10(ymin)
                print(dy, dx)
                ax.set_aspect(dx/dy)




                for k in range(nz):
                    ax.plot(bins, hist[k, :].flatten(), c=colors[k], markersize=20, label=legend_label[k])

                ax.grid(True, linewidth=0.25)
                # ax.set_xscale('log')
                ax.set_yscale('log')
                # Axis
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=5)
                ax.set_xlabel(x_label[i][j], fontsize=font_size, labelpad=0.5)
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                # Title
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005)
                # Legend
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])
                ax.legend(loc='best', fontsize=8, numpoints=1)


        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_power_spectrum(self, filename_output, x, y, xmin, xmax, ymin, ymax,
                            title_label, x_label, y_label, legend_label, imgformat='png'):

        # Font
        font_size = 13
        # Colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                  '#bcbd22', '#17becf']

        nz, _ = hist.shape

        # Plot
        nrows = 1
        ncols = 1
        figsize_x = 5.25 * ncols
        figsize_y = 4.5 * nrows
        fig = plt.figure(figsize=(figsize_x, figsize_y))

        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                # Subplot
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols - 0.005,
                                        right=float(j + 1) / ncols - 0.05 / ncols - 0.05 * 0.45 / 1.05 - 0.005,
                                        bottom=0.10 / nrows + float(i) / nrows, top=float(i + 1) / nrows - 0.10 / nrows,
                                        wspace=0.00)
                ax = fig.add_subplot(spec[:, :])  # , projection='scatter_density')

                dx = np.log10(xmax)-np.log10(xmin)
                dy = np.log10(ymax)-np.log10(ymin)
                print(dy, dx)
                ax.set_aspect(dx/dy)

                # Axis
                for k in range(nz):
                    ax.plot(x, y.flatten(), c=colors[k], markersize=20, label=legend_label[k])

                ax.grid(True, linewidth=0.25)
                ax.set_xscale('log')
                ax.set_yscale('log')
                # Axis
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size)
                ax.set_ylabel(y_label[i][j], fontsize=font_size, labelpad=5)
                ax.set_xlabel(x_label[i][j], fontsize=font_size, labelpad=0.5)
                ax.xaxis.set_major_locator(plt.MaxNLocator(5))
                # Title
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005)
                # Legend
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])
                ax.legend(loc='best', fontsize=8, numpoints=1)


        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_scalar_maps(self, filename_output, img, title_label, min_cb, max_cb, colorbar_label,
                         imgformat='png', grid=0, xi_crop=[[0]], yi_crop=[[0]], nx_crop=[[50]], ny_crop=[[50]],
                         x_majorticks=[[1]], y_majorticks=[[1]],
                         xi_rect=[], yi_rect=[], nx_rect=[], ny_rect=[], colors_rect=[], linewidth_rect=[],
                         labels_rect=[], xi_labels=[], yi_labels=[]):
    
        # Dimensions
        nrows, ncols, ny, nx = img.shape
    
        # Position
        dx = self.dx * 0.001
        x_label = 'x (Mm)'
        y_label = 'y (Mm)'
    
        # Properties
        font_size = 13
        figsize_x = ncols * 5.25
        figsize_y = nrows * 4.5
        nb_rect = len(xi_rect)
        fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=False)
    
        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                img_plot = img[i, j, yi_crop[i][j]:yi_crop[i][j] + ny_crop[i][j],
                           xi_crop[i][j]:xi_crop[i][j] + nx_crop[i][j]]
                extent = np.array(
                    [xi_crop[i][j], xi_crop[i][j] + nx_crop[i][j], yi_crop[i][j], yi_crop[i][j] + ny_crop[i][j]]) * dx
                # Colormap
                if min_cb[i][j] * max_cb[i][j] < 0:
                    colorbar_cmap = 'RdBu_r'
                else:
                    colorbar_cmap = 'GnBu_r'
                # Plot
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols,
                                        right=float(j + 1) / ncols - 0.05 / ncols,
                                        bottom=0.10 / nrows + float((nrows - 1 - i)) / nrows,
                                        top=float(nrows - i) / nrows - 0.10 / nrows, wspace=0.00)
                ax = fig.add_subplot(spec[:, :])
                I = ax.imshow(img_plot, extent=extent, cmap=colorbar_cmap, aspect=1,
                              interpolation='none',
                              vmin=min_cb[i][j], vmax=max_cb[i][j], alpha=1.0, origin='lower')
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size,
                                               left=True, right=True)
                ax.set_ylabel(y_label, fontsize=font_size, labelpad=5.0)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size,
                                               bottom=True, top=True)
                ax.set_xlabel(x_label, fontsize=font_size, labelpad=3.0)
                ax.get_xaxis().set_major_locator(plt.MultipleLocator(x_majorticks[i][j]))
                ax.get_yaxis().set_major_locator(plt.MultipleLocator(y_majorticks[i][j]))
                if grid == 1:
                    ax.grid(color='red', linewidth=1)
                if nb_rect > 0:
                    for k in range(nb_rect):
                        rect = patches.Rectangle((xi_rect[k] * self.dx * 0.001, yi_rect[k] * self.dy * 0.001),
                                                 nx_rect[k] * self.dx * 0.001,
                                                 ny_rect[k] * self.dy * 0.001, linewidth=linewidth_rect[k],
                                                 edgecolor=colors_rect[k],
                                                 facecolor='none')
                        ax.add_patch(rect)
                        if len(labels_rect) > 0:
                            text(xi_labels[k], yi_labels[k], labels_rect[k], horizontalalignment='center', verticalalignment='center', 
                            transform=ax.transAxes, fontsize=font_size)
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005, wrap=True)
                # Colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0)
                cb = colorbar(I, extend='neither', cax=cax)
                cb.ax.tick_params(axis='y', direction='out', labelsize=font_size, width=1, length=2.5)
                cb.set_label(colorbar_label[i][j], labelpad=20.0, rotation=270, size=font_size)
    
        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_contours_maps(self, filename_output, img, clusters1, clusters2, title_label, min_cb, max_cb, colorbar_label,
                           imgformat='png', grid=0, xi_crop=[[0]], yi_crop=[[0]], nx_crop=[[50]], ny_crop=[[50]],
                           x_majorticks=[[1]], y_majorticks=[[1]],
                           xi_rect=[], yi_rect=[], nx_rect=[], ny_rect=[], colors_rect=[], linewidth_rect=[]):
    
        # Dimensions
        nrows, ncols, ny, nx = img.shape
    
        # Position
        dx = self.dx * 0.001
        x_label = 'x (Mm)'
        y_label = 'y (Mm)'
    
        # Properties
        font_size = 13
        figsize_x = ncols * 5.25
        figsize_y = nrows * 4.5
        nb_rect = len(xi_rect)
        fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=False)
    
        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                img_plot = img[i, j, yi_crop[i][j]:yi_crop[i][j] + ny_crop[i][j],
                           xi_crop[i][j]:xi_crop[i][j] + nx_crop[i][j]]
                extent = np.array(
                    [xi_crop[i][j], xi_crop[i][j] + nx_crop[i][j], yi_crop[i][j], yi_crop[i][j] + ny_crop[i][j]]) * dx
                # Colormap
                if min_cb[i][j] * max_cb[i][j] < 0:
                    colorbar_cmap = 'RdBu_r'
                else:
                    colorbar_cmap = 'GnBu_r'
                # Plot
                if j == 0:
                    ii_qs = np.where(clusters1 == 0)
                    ii_pe = np.where(clusters1 == 1)
                    cluster = np.zeros((256, 256))
                    cluster[ii_qs] = 1
                    output = np.zeros((256, 256))
                    um_mask = np.zeros((256, 256))
                    z = np.zeros((256, 256))
                    xp = np.concatenate((z[:1, :], cluster[0:255, :]), axis=0)
                    xm = np.concatenate((cluster[1:, :], z[255:, :]), axis=0)
                    yp = np.concatenate((z[:, :1], cluster[:, 0:255]), axis=1)
                    ym = np.concatenate((cluster[:, 1:], z[:, 255:]), axis=1)
                    print(cluster.shape, xp.shape, xm.shape, yp.shape, ym.shape)
                    ii = np.where(cluster + xp == 1)
                    output[ii] = 1
                    ii = np.where(cluster + xm == 1)
                    output[ii] = 1
                    ii = np.where(cluster + yp == 1)
                    output[ii] = 1
                    ii = np.where(cluster + ym == 1)
                    output[ii] = 1
                    z[80:170, 0:170] = 1
                    output = output*z
                    pe_mask = np.ma.masked_where(output == 0, output)
                    
                    cluster = np.zeros((256, 256))
                    cluster[ii_pe] = 1
                    output = np.zeros((256, 256))
                    um_mask = np.zeros((256, 256))
                    z = np.zeros((256, 256))
                    xp = np.concatenate((z[:1, :], cluster[0:255, :]), axis=0)
                    xm = np.concatenate((cluster[1:, :], z[255:, :]), axis=0)
                    yp = np.concatenate((z[:, :1], cluster[:, 0:255]), axis=1)
                    ym = np.concatenate((cluster[:, 1:], z[:, 255:]), axis=1)
                    print(cluster.shape, xp.shape, xm.shape, yp.shape, ym.shape)
                    ii = np.where(cluster + xp == 1)
                    output[ii] = 1
                    ii = np.where(cluster + xm == 1)
                    output[ii] = 1
                    ii = np.where(cluster + yp == 1)
                    output[ii] = 1
                    ii = np.where(cluster + ym == 1)
                    output[ii] = 1
                    z[80:170, 0:170] = 1
                    output = output * z
                    um_mask = np.ma.masked_where(output == 0, output)
                else:
                    ii_qs = np.where(clusters2 == 0)
                    ii_pe = np.where(clusters2 == 1)
                    cluster = np.zeros((377, 744))
                    cluster[ii_qs] = 1
                    output = np.zeros((377, 744))
                    um_mask = np.zeros((377, 744))
                    z = np.zeros((377, 744))
                    xp = np.concatenate((z[:1, :], cluster[0:376, :]), axis=0)
                    xm = np.concatenate((cluster[1:, :], z[376:, :]), axis=0)
                    yp = np.concatenate((z[:, :1], cluster[:, 0:743]), axis=1)
                    ym = np.concatenate((cluster[:, 1:], z[:, 743:]), axis=1)
                    print(cluster.shape, xp.shape, xm.shape, yp.shape, ym.shape)
                    ii = np.where(cluster + xp == 1)
                    output[ii] = 1
                    ii = np.where(cluster + xm == 1)
                    output[ii] = 1
                    ii = np.where(cluster + yp == 1)
                    output[ii] = 1
                    ii = np.where(cluster + ym == 1)
                    output[ii] = 1
                    # z[80:170, 0:170] = 1
                    # output = output * z
                    pe_mask = np.ma.masked_where(output == 0, output)
                    pe_mask = pe_mask[yi_crop[i][j]:yi_crop[i][j] + ny_crop[i][j],
                              xi_crop[i][j]:xi_crop[i][j] + nx_crop[i][j]]
    
                    cluster = np.zeros((377, 744))
                    cluster[ii_pe] = 1
                    output = np.zeros((377, 744))
                    um_mask = np.zeros((377, 744))
                    z = np.zeros((377, 744))
                    xp = np.concatenate((z[:1, :], cluster[0:376, :]), axis=0)
                    xm = np.concatenate((cluster[1:, :], z[376:, :]), axis=0)
                    yp = np.concatenate((z[:, :1], cluster[:, 0:743]), axis=1)
                    ym = np.concatenate((cluster[:, 1:], z[:, 743:]), axis=1)
                    print(cluster.shape, xp.shape, xm.shape, yp.shape, ym.shape)
                    ii = np.where(cluster + xp == 1)
                    output[ii] = 1
                    ii = np.where(cluster + xm == 1)
                    output[ii] = 1
                    ii = np.where(cluster + yp == 1)
                    output[ii] = 1
                    ii = np.where(cluster + ym == 1)
                    output[ii] = 1
                    # z[80:170, 0:170] = 1
                    # output = output * z
                    um_mask = np.ma.masked_where(output == 0, output)
                    um_mask = um_mask[yi_crop[i][j]:yi_crop[i][j] + ny_crop[i][j],
                           xi_crop[i][j]:xi_crop[i][j] + nx_crop[i][j]]
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols,
                                        right=float(j + 1) / ncols - 0.05 / ncols,
                                        bottom=0.10 / nrows + float((nrows - 1 - i)) / nrows,
                                        top=float(nrows - i) / nrows - 0.10 / nrows, wspace=0.00)
                ax = fig.add_subplot(spec[:, :])
                I = ax.imshow(img_plot, extent=extent, cmap=colorbar_cmap, aspect=1,
                              interpolation='none',
                              vmin=min_cb[i][j], vmax=max_cb[i][j], alpha=1.0, origin='lower')
                I02 = ax.imshow(um_mask, extent=extent, cmap='spring_r',
                                interpolation='none', origin='lower', alpha=1.0)
                I01 = ax.imshow(pe_mask, extent=extent, cmap='winter',
                                interpolation='none', origin='lower', alpha=1.0)
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size,
                                               left=True, right=True)
                ax.set_ylabel(y_label, fontsize=font_size, labelpad=5.0)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size,
                                               bottom=True, top=True)
                ax.set_xlabel(x_label, fontsize=font_size, labelpad=3.0)
                ax.get_xaxis().set_major_locator(plt.MultipleLocator(x_majorticks[i][j]))
                ax.get_yaxis().set_major_locator(plt.MultipleLocator(y_majorticks[i][j]))
                if grid == 1:
                    ax.grid(color='red', linewidth=1)
                if nb_rect > 0:
                    for k in range(nb_rect):
                        rect = patches.Rectangle((xi_rect[k] * self.dx * 0.001, yi_rect[k] * self.dy * 0.001),
                                                 nx_rect[k] * self.dx * 0.001,
                                                 ny_rect[k] * self.dy * 0.001, linewidth=linewidth_rect[k],
                                                 edgecolor=colors_rect[k],
                                                 facecolor='none')
                        ax.add_patch(rect)
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005, wrap=True)
                # Colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0)
                cb = colorbar(I, extend='neither', cax=cax)
                cb.ax.tick_params(axis='y', direction='out', labelsize=font_size, width=1, length=2.5)
                cb.set_label(colorbar_label[i][j], labelpad=20.0, rotation=270, size=font_size)
    
        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_cluster_maps(self, filename_output, img, title_label, min_cb, max_cb, colorbar_label,
                          imgformat='png', grid=0, xi_crop=[[0]], yi_crop=[[0]], nx_crop=[[50]], ny_crop=[[50]],
                          x_majorticks=[[1]], y_majorticks=[[1]],
                          xi_rect=[], yi_rect=[], nx_rect=[], ny_rect=[], colors_rect=[], linewidth_rect=[]):
    
        # Dimensions
        nrows, ncols, ny, nx = img.shape
    
        # Position
        dx = self.dx * 0.001
        x_label = 'x (Mm)'
        y_label = 'y (Mm)'
    
        # Properties
        font_size = 13
        figsize_x = ncols * 5.25
        figsize_y = nrows * 4.5
        nb_rect = len(xi_rect)
        fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=False)
    
        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                img_plot = img[i, j, yi_crop[i][j]:yi_crop[i][j] + ny_crop[i][j],
                           xi_crop[i][j]:xi_crop[i][j] + nx_crop[i][j]]
                extent = np.array(
                    [xi_crop[i][j], xi_crop[i][j] + nx_crop[i][j], yi_crop[i][j], yi_crop[i][j] + ny_crop[i][j]]) * dx
                # Colormap
                if min_cb[i][j] * max_cb[i][j] < 0:
                    colorbar_cmap = 'RdBu_r'
                else:
                    if j == 0:
                        colorbar_cmap = 'GnBu_r'
                    else:
                        colorbar_cmap = plt.cm.get_cmap('Accent', 3)
                # Plot
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols,
                                        right=float(j + 1) / ncols - 0.05 / ncols,
                                        bottom=0.10 / nrows + float((nrows - 1 - i)) / nrows,
                                        top=float(nrows - i) / nrows - 0.10 / nrows, wspace=0.00)
                ax = fig.add_subplot(spec[:, :])
                I = ax.imshow(img_plot, extent=extent, cmap=colorbar_cmap, aspect=1,
                              interpolation='none',
                              vmin=min_cb[i][j], vmax=max_cb[i][j], alpha=1.0, origin='lower')
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size,
                                               left=True, right=True)
                ax.set_ylabel(y_label, fontsize=font_size, labelpad=5.0)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size,
                                               bottom=True, top=True)
                ax.set_xlabel(x_label, fontsize=font_size, labelpad=3.0)
                ax.get_xaxis().set_major_locator(plt.MultipleLocator(x_majorticks[i][j]))
                ax.get_yaxis().set_major_locator(plt.MultipleLocator(y_majorticks[i][j]))
                if grid == 1:
                    ax.grid(color='red', linewidth=1)
                if nb_rect > 0:
                    for k in range(nb_rect):
                        rect = patches.Rectangle((xi_rect[k] * self.dx * 0.001, yi_rect[k] * self.dy * 0.001),
                                                 nx_rect[k] * self.dx * 0.001,
                                                 ny_rect[k] * self.dy * 0.001, linewidth=linewidth_rect[k],
                                                 edgecolor=colors_rect[k],
                                                 facecolor='none')
                        ax.add_patch(rect)
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005, wrap=True)
                # Colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0)
                cb = colorbar(I, extend='neither', cax=cax)
                cb.ax.tick_params(axis='y', direction='out', labelsize=font_size, width=1, length=2.5)
                if j ==1:
                    cb.set_ticks(np.arange(0, 3))
                cb.set_label(colorbar_label[i][j], labelpad=20.0, rotation=270, size=font_size)
    
        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_scalar_maps2(self, filename_output, img, title_label, min_cb, max_cb, colorbar_label,
                          imgformat='png', grid=0, xi_crop=[[0]], yi_crop=[[0]], nx_crop=[[50]], ny_crop=[[50]],
                          x_majorticks=[[1]], y_majorticks=[[1]],
                          xi_rect=[], yi_rect=[], nx_rect=[], ny_rect=[], colors_rect=[], linewidth_rect=[],
                          labels_rect=[], xi_labels=[], yi_labels=[]):
    
        # Dimensions
        nrows, ncols, ny, nx = img.shape
    
        # Position
        dx = self.dx * 0.001
        x_label = 'x (Mm)'
        y_label = 'y (Mm)'
    
        # Properties
        font_size = 13
        figsize_x = ncols * 5.25 * 2
        figsize_y = nrows * 4.5
        nb_rect = len(xi_rect)
        fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=False)
    
        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                img_plot = img[i, j, yi_crop[i][j]:yi_crop[i][j] + ny_crop[i][j],
                           xi_crop[i][j]:xi_crop[i][j] + nx_crop[i][j]]
                extent = np.array(
                    [xi_crop[i][j], xi_crop[i][j] + nx_crop[i][j], yi_crop[i][j], yi_crop[i][j] + ny_crop[i][j]]) * dx
                # Colormap
                if min_cb[i][j] * max_cb[i][j] < 0:
                    colorbar_cmap = 'RdBu_r'
                else:
                    colorbar_cmap = 'GnBu_r'
                # Plot
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols,
                                        right=float(j + 1) / ncols - 0.05 / ncols,
                                        bottom=0.10 / nrows + float((nrows - 1 - i)) / nrows,
                                        top=float(nrows - i) / nrows - 0.10 / nrows, wspace=0.00)
                ax = fig.add_subplot(spec[:, :])
                I = ax.imshow(img_plot, extent=extent, cmap=colorbar_cmap, aspect=1,
                              interpolation='none',
                              vmin=min_cb[i][j], vmax=max_cb[i][j], alpha=1.0, origin='lower')
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size,
                                               left=True, right=True)
                ax.set_ylabel(y_label, fontsize=font_size, labelpad=5.0)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size,
                                               bottom=True, top=True)
                ax.set_xlabel(x_label, fontsize=font_size, labelpad=3.0)
                ax.get_xaxis().set_major_locator(plt.MultipleLocator(x_majorticks[i][j]))
                ax.get_yaxis().set_major_locator(plt.MultipleLocator(y_majorticks[i][j]))
                if grid == 1:
                    ax.grid(color='red', linewidth=1)
                if nb_rect > 0:
                    for k in range(nb_rect):
                        rect = patches.Rectangle((xi_rect[k] * self.dx * 0.001, yi_rect[k] * self.dy * 0.001),
                                                 nx_rect[k] * self.dx * 0.001,
                                                 ny_rect[k] * self.dy * 0.001, linewidth=linewidth_rect[k],
                                                 edgecolor=colors_rect[k],
                                                 facecolor='none')
                        ax.add_patch(rect)
                        if len(labels_rect) > 0:
                            text(xi_labels[k], yi_labels[k], labels_rect[k], horizontalalignment='center', verticalalignment='center', 
                            transform=ax.transAxes, fontsize=font_size)
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005, wrap=True)
                # Colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="2.5%", pad=0)
                cb = colorbar(I, extend='neither', cax=cax)
                cb.ax.tick_params(axis='y', direction='out', labelsize=font_size, width=1, length=2.5)
                cb.set_label(colorbar_label[i][j], labelpad=20.0, rotation=270, size=font_size)
    
        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_vector_gifs(self, filename_output, img, vx, vy, title_label, min_cb, max_cb, colorbar_label,
                         imgformat='png', xi_crop=[[0]], yi_crop=[[0]], nx_crop=[[50]], ny_crop=[[50]],
                         qk_length=[[0.5]],
                         scale=[[1.0]], width=[[0.1]], headwidth=[[2]], headlength=[[4]], headaxislength=[[4.5]],
                         step=[[1]],
                         x_majorticks=[[1]], y_majorticks=[[1]],
                         xi_rect=[], yi_rect=[], nx_rect=[], ny_rect=[], colors_rect=[], linewidth_rect=[]):
    
        # Dimensions
        ncols, ny, nx = img.shape
    
        # Vector parameters
        dx = self.dx * 0.001
        vunit = self.dx / self.dt
        qk_x = 0.750
        qk_y = 0.030
    
        # Position
        x_label = 'x (Mm)'
        y_label = 'y (Mm)'
    
        # Properties
        font_size = 13
        figsize_x = 4.5
        figsize_y = 4.5
        nb_rect = len(xi_rect)
        # fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=False)
    
        # Rows
        filename = 'img_tmp.png'
        with imageio.get_writer('MURaM_wl_movie.gif', mode='I', fps=1) as writer:
            for j in range(ncols):
                fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=False)
                img_plot = img[j, yi_crop[0][0]:yi_crop[0][0] + ny_crop[0][0],
                               xi_crop[0][0]:xi_crop[0][0] + nx_crop[0][0]]
                vx_plot = vx[j, yi_crop[0][0]:yi_crop[0][0] + ny_crop[0][0],
                             xi_crop[0][0]:xi_crop[0][0] + nx_crop[0][0]] / vunit * dx
                vy_plot = vy[j, yi_crop[0][0]:yi_crop[0][0] + ny_crop[0][0],
                             xi_crop[0][0]:xi_crop[0][0] + nx_crop[0][0]] / vunit * dx
                extent = np.array(
                         [xi_crop[0][0], xi_crop[0][0] + nx_crop[0][0], yi_crop[0][0], yi_crop[0][0] + ny_crop[0][0]]) * dx
                x_plot = np.arange(xi_crop[0][0], xi_crop[0][0] + nx_crop[0][0], dtype='float') * dx
                y_plot = np.arange(yi_crop[0][0], yi_crop[0][0] + ny_crop[0][0], dtype='float') * dx
                v_slice = np.s_[::step[0][0], ::step[0][0]]
                # Grid of the flow fields, first define the default, then slice it to match anchors spaced by "step" pixels
                x_plot = x_plot[::step[0][0]]
                y_plot = y_plot[::step[0][0]]
                # Colormap
                if min_cb[0][0] * max_cb[0][0] < 0:
                    colorbar_cmap = 'RdBu_r'
                else:
                    colorbar_cmap = 'GnBu_r'
                # Plot
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0,
                                        right=1,
                                        bottom=0,
                                        top=1, wspace=0.00)
                ax = fig.add_subplot(spec[:, :])
                ax.set_axis_off()
                # fig.add_axes(ax)
                I = ax.imshow(img_plot, extent=extent, cmap=colorbar_cmap, aspect=1,
                              interpolation='none',
                              vmin=min_cb[0][0], vmax=max_cb[0][0], alpha=1.0, origin='lower')
                q = ax.quiver(x_plot, y_plot, vx_plot[::step[0][0], ::step[0][0]], vy_plot[::step[0][0], ::step[0][0]],
                              units='xy', scale=scale[0][0], width=width[0][0], headwidth=headwidth[0][0],
                              headlength=headlength[0][0],
                              headaxislength=headaxislength[0][0], pivot='tail')
                #qk_label = str(np.around(qk_length[0][0], decimals=2))
                #qk = ax.quiverkey(q, 0.05 / 1 + float(0) / 1 + 0.60 / 1, 0.015 / 1 + 0 / 1,
                #                  qk_length[0][0] / vunit * dx,
                #                  qk_label + r' km s$^{-1}$', labelpos='E',
                #                  coordinates='figure', fontproperties={'size': str(font_size)},
                #                  labelsep=0.05)
                # ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size,
                #                                left=True, right=True)
                # ax.set_ylabel(y_label, fontsize=font_size, labelpad=5.0)
                # ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size,
                #                                bottom=True, top=True)
                # ax.set_xlabel(x_label, fontsize=font_size, labelpad=3.0)
                # ax.get_xaxis().set_major_locator(plt.MultipleLocator(x_majorticks[i][j]))
                # ax.get_yaxis().set_major_locator(plt.MultipleLocator(y_majorticks[i][j]))
    
                # Save
                # plt.draw()
                plt.savefig(filename, format=imgformat, dpi=300)
                plt.close()
                image = imageio.imread(filename)
                writer.append_data(image)

    def plot_scalar_gifs(self, filename_output, img, title_label, min_cb, max_cb, colorbar_label,
                         imgformat='png', xi_crop=[[0]], yi_crop=[[0]], nx_crop=[[50]], ny_crop=[[50]],
                         x_majorticks=[[1]], y_majorticks=[[1]], t_crop=[[0]],
                         xi_rect=[], yi_rect=[], nx_rect=[], ny_rect=[], colors_rect=[], linewidth_rect=[]):

        # Dimensions
        nt, ny, nx = img.shape

        # Vector parameters
        dx = self.dx * 0.001
        vunit = self.dx / self.dt
        qk_x = 0.750
        qk_y = 0.030

        # Position
        x_label = 'x (Mm)'
        y_label = 'y (Mm)'

        # Properties
        font_size = 13
        ncols = 1
        nrows = 1
        # figsize_x = ncols * 5.25 * 2
        # figsize_y = nrows * 4.5
        # nb_rect = len(xi_rect)
        # Properties
        font_size = 13
        figsize_x = ncols * 5.25 * 2
        figsize_y = nrows * 4.5
        nb_rect = len(xi_rect)
        # fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=False)
        # fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=False)

        # Rows
        filename = 'img_tmp.png'
        i = 0
        k = 0
        with imageio.get_writer(filename_output, mode='I', fps=1) as writer:
            for j in range(nt):
                fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=False)
                img_plot = img[j, yi_crop[0][0]:yi_crop[0][0] + ny_crop[0][0],
                           xi_crop[0][0]:xi_crop[0][0] + nx_crop[0][0]]
                print(img_plot.shape)
                extent = np.array(
                    [xi_crop[0][0], xi_crop[0][0] + nx_crop[0][0], yi_crop[0][0], yi_crop[0][0] + ny_crop[0][0]]) * dx
                #extent = np.array(
                #    [yi_crop[0][0], yi_crop[0][0] + ny_crop[0][0], xi_crop[0][0], xi_crop[0][0] + nx_crop[0][0]]) * dx
                # Colormap
                if min_cb[0][0] * max_cb[0][0] < 0:
                    colorbar_cmap = 'RdBu_r'
                else:
                    colorbar_cmap = 'GnBu_r'
                # Plot
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(k) / ncols,
                                        right=float(k + 1) / ncols - 0.05 / ncols,
                                        bottom=0.10 / nrows + float((nrows - 1 - i)) / nrows,
                                        top=float(nrows - i) / nrows - 0.10 / nrows, wspace=0.00)
                ax = fig.add_subplot(spec[:, :])
                # ax.set_axis_off()
                # fig.add_axes(ax)
                I = ax.imshow(img_plot, extent=extent, cmap=colorbar_cmap, aspect=1,
                              interpolation='none',
                              vmin=min_cb[0][0], vmax=max_cb[0][0], alpha=1.0, origin='lower')
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size,
                                               left=True, right=True)
                ax.set_ylabel(y_label, fontsize=font_size, labelpad=5.0)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size,
                                               bottom=True, top=True)
                ax.set_xlabel(x_label, fontsize=font_size, labelpad=3.0)
                ax.get_xaxis().set_major_locator(plt.MultipleLocator(x_majorticks[i][k]))
                ax.get_yaxis().set_major_locator(plt.MultipleLocator(y_majorticks[i][k]))

                ax.set_title(title_label[0][0]+' - Timestep = {0}'.format(j+t_crop[0][0]), fontsize=font_size, y=1.005, wrap=True)
                # Colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="2.5%", pad=0)
                cb = colorbar(I, extend='neither', cax=cax)
                cb.ax.tick_params(axis='y', direction='out', labelsize=font_size, width=1, length=2.5)
                cb.set_label(colorbar_label[0][0], labelpad=20.0, rotation=270, size=font_size)

                # Save
                # plt.draw()
                plt.savefig(filename, format=imgformat, dpi=300)
                plt.close()
                image = imageio.imread(filename)
                writer.append_data(image)

    def plot_vector_maps(self, filename_output, img, vx, vy, title_label, min_cb, max_cb, colorbar_label,
                         imgformat='png', xi_crop=[[0]], yi_crop=[[0]], nx_crop=[[50]], ny_crop=[[50]],
                         qk_length=[[0.5]],
                         scale=[[1.0]], width=[[0.1]], headwidth=[[2]], headlength=[[4]], headaxislength=[[4.5]],
                         step=[[1]],
                         x_majorticks=[[1]], y_majorticks=[[1]],
                         xi_rect=[], yi_rect=[], nx_rect=[], ny_rect=[], colors_rect=[], linewidth_rect=[]):
    
        # Dimensions
        nrows, ncols, ny, nx = img.shape
    
        # Vector parameters
        dx = self.dx * 0.001
        vunit = self.dx  # / self.dt
        qk_x = 0.750
        qk_y = 0.030
    
        # Position
        x_label = 'x (Mm)'
        y_label = 'y (Mm)'
    
        # Properties
        font_size = 13
        figsize_x = ncols * 5.25
        figsize_y = nrows * 4.5
        nb_rect = len(xi_rect)
        fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=False)
    
        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                img_plot = img[i, j, yi_crop[i][j]:yi_crop[i][j] + ny_crop[i][j],
                           xi_crop[i][j]:xi_crop[i][j] + nx_crop[i][j]]
                vx_plot = vx[i, j, yi_crop[i][j]:yi_crop[i][j] + ny_crop[i][j],
                          xi_crop[i][j]:xi_crop[i][j] + nx_crop[i][j]] / vunit * dx
                vy_plot = vy[i, j, yi_crop[i][j]:yi_crop[i][j] + ny_crop[i][j],
                          xi_crop[i][j]:xi_crop[i][j] + nx_crop[i][j]] / vunit * dx
                extent = np.array(
                    [xi_crop[i][j], xi_crop[i][j] + nx_crop[i][j], yi_crop[i][j], yi_crop[i][j] + ny_crop[i][j]]) * dx
                x_plot = np.arange(xi_crop[i][j], xi_crop[i][j] + nx_crop[i][j], dtype='float') * dx
                y_plot = np.arange(yi_crop[i][j], yi_crop[i][j] + ny_crop[i][j], dtype='float') * dx
                v_slice = np.s_[::step[i][j], ::step[i][j]]
                # Grid of the flow fields, first define the default, then slice it to match anchors spaced by "step" pixels
                x_plot = x_plot[::step[i][j]]
                y_plot = y_plot[::step[i][j]]
                # Colormap
                if min_cb[i][j] * max_cb[i][j] < 0:
                    colorbar_cmap = 'RdBu_r'
                else:
                    colorbar_cmap = 'GnBu_r'
                # Plot
                spec = fig.add_gridspec(nrows=1, ncols=1, left=0.05 / ncols + float(j) / ncols,
                                        right=float(j + 1) / ncols - 0.05 / ncols,
                                        bottom=0.10 / nrows + float((nrows - 1 - i)) / nrows,
                                        top=float(nrows - i) / nrows - 0.10 / nrows, wspace=0.00)
                ax = fig.add_subplot(spec[:, :])
                I = ax.imshow(img_plot, extent=extent, cmap=colorbar_cmap, aspect=1,
                              interpolation='none',
                              vmin=min_cb[i][j], vmax=max_cb[i][j], alpha=1.0, origin='lower')
                q = ax.quiver(x_plot, y_plot, vx_plot[::step[i][j], ::step[i][j]], vy_plot[::step[i][j], ::step[i][j]],
                              units='xy', scale=scale[i][j], width=width[i][j], headwidth=headwidth[i][j],
                              headlength=headlength[i][j],
                              headaxislength=headaxislength[i][j], pivot='tail', scale_units='xy')
                qk_label = str(np.around(qk_length[i][j], decimals=2))
                qk = ax.quiverkey(q, 0.05 / ncols + float(j) / ncols + 0.60 / ncols, 0.015 / nrows + i / nrows,
                                  qk_length[i][j] / 1000.,
                                  qk_label + r' km s$^{-1}$', labelpos='E',
                                  coordinates='figure', fontproperties={'size': str(font_size)},
                                  labelsep=0.05)
                ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size,
                                               left=True, right=True)
                ax.set_ylabel(y_label, fontsize=font_size, labelpad=5.0)
                ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5, labelsize=font_size,
                                               bottom=True, top=True)
                ax.set_xlabel(x_label, fontsize=font_size, labelpad=3.0)
                ax.get_xaxis().set_major_locator(plt.MultipleLocator(x_majorticks[i][j]))
                ax.get_yaxis().set_major_locator(plt.MultipleLocator(y_majorticks[i][j]))
                if nb_rect > 0:
                    for k in range(nb_rect):
                        rect = patches.Rectangle((xi_rect[k] * self.dx * 0.001, yi_rect[k] * self.dy * 0.001),
                                                 nx_rect[k] * self.dx * 0.001,
                                                 ny_rect[k] * self.dy * 0.001, linewidth=linewidth_rect[k],
                                                 edgecolor=colors_rect[k],
                                                 facecolor='none')
                        ax.add_patch(rect)
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005, wrap=True)
                # Colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0)
                cb = colorbar(I, extend='neither', cax=cax)
                cb.ax.tick_params(axis='y', direction='out', labelsize=font_size, width=1, length=2.5)
                cb.set_label(colorbar_label[i][j], labelpad=20.0, rotation=270, size=font_size)
    
        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_scalar_maps_shared(self, filename_output, img, title_label, min_cb, max_cb, colorbar_label,
                                imgformat='png', xi_crop=[[0]], yi_crop=[[0]], nx_crop=[[50]], ny_crop=[[50]],
                                x_majorticks=[[1]], y_majorticks=[[1]],
                                xi_rect=[], yi_rect=[], nx_rect=[], ny_rect=[], colors_rect=[], linewidth_rect=[]):
    
        # Dimensions
        nrows, ncols, ny, nx = img.shape
    
        # Vector parameters
        dx = self.dx * 0.001
    
        # Position
        x_label = 'x (Mm)'
        y_label = 'y (Mm)'
    
        # Properties
        font_size = 13
        figsize_x = ncols * 5.25 * (1.0 / ncols + (ncols - 1) * (0.05 / ncols + 1.00 / 1.05 * 0.72 / ncols))
        figsize_y = nrows * 4.5 * (1.0 / nrows + (nrows - 1) * (0.10 / nrows + 0.80 / nrows))
        nb_rect = len(xi_rect)
        fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=False)
    
        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                img_plot = img[i, j, yi_crop[i][j]:yi_crop[i][j] + ny_crop[i][j],
                           xi_crop[i][j]:xi_crop[i][j] + nx_crop[i][j]]
                extent = np.array(
                    [xi_crop[i][j], xi_crop[i][j] + nx_crop[i][j], yi_crop[i][j], yi_crop[i][j] + ny_crop[i][j]]) * dx
                # Colormap
                if min_cb[i][j] * max_cb[i][j] < 0:
                    colorbar_cmap = 'RdBu_r'
                else:
                    colorbar_cmap = 'GnBu_r'
                # Plot
                new_spec = (1.0 / ncols + (ncols - 1) * (0.05 / ncols + 1.00 / 1.05 * 0.72 / ncols))
                ver_spec = (1.0 / nrows + (nrows - 1) * (0.10 / nrows + 0.80 / nrows))
                bottom_spec = 0.10 / nrows + float((nrows - 1 - i)) / nrows
                top_spec = float(nrows - i) / nrows - 0.10 / nrows
                if j == 0:
                    left_spec = 0.14 / ncols / new_spec
                    right_spec = (0.14 / ncols + 1.00 / 1.05 * 0.72 / ncols) / new_spec
            
                elif 0 < j < ncols - 1:
                    left_spec = (0.14 / ncols + 1.00 / 1.05 * 0.72 / ncols) / new_spec + (j - 1) * (
                                1.00 / 1.05 * 0.72 / ncols) / new_spec + (j) * (0.05 / ncols) / new_spec
                    right_spec = left_spec + (1.00 / 1.05 * 0.72 / ncols) / new_spec
                else:
                    left_spec = 1 - 0.86 / ncols / new_spec
                    right_spec = 1 - 0.14 / ncols / new_spec
                if i == 0:
                    bottom_spec = 1.0 - (0.9 / nrows) / ver_spec
                    top_spec = 1.0 - (0.1 / nrows) / ver_spec
                elif 0 < i < nrows - 1:
                    bottom_spec = (nrows - i - 1) / nrows / ver_spec
                    top_spec = bottom_spec + 0.80 / nrows / ver_spec
                else:
                    bottom_spec = (0.10 / nrows) / ver_spec
                    top_spec = bottom_spec + (0.80 / nrows) / ver_spec
                spec = fig.add_gridspec(nrows=1, ncols=1, left=left_spec,
                                        right=right_spec,
                                        bottom=bottom_spec,
                                        top=top_spec, wspace=0.00)
                ax = fig.add_subplot(spec[:, :])
                I = ax.imshow(img_plot, extent=extent, cmap=colorbar_cmap, aspect=1,
                              interpolation='none',
                              vmin=min_cb[i][j], vmax=max_cb[i][j], alpha=1.0, origin='lower')
                if (j == 0):
                    ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5,
                                                   labelsize=font_size,
                                                   left=True, right=True)
                    ax.set_ylabel(y_label, fontsize=font_size, labelpad=5.0)
                else:
                    ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5,
                                                   labelsize=font_size,
                                                   left=True, right=True, labelleft=False, labelright=False)
                if (i == nrows - 1):
                    ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5,
                                                   labelsize=font_size,
                                                   bottom=True, top=True)
                    ax.set_xlabel(x_label, fontsize=font_size, labelpad=3.0)
                else:
                    ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5,
                                                   labelsize=font_size,
                                                   bottom=True, top=True, labelbottom=False, labeltop=False)
                ax.get_xaxis().set_major_locator(plt.MultipleLocator(x_majorticks[i][j]))
                ax.get_yaxis().set_major_locator(plt.MultipleLocator(y_majorticks[i][j]))
                if nb_rect > 0:
                    for k in range(nb_rect):
                        rect = patches.Rectangle((xi_rect[k] * self.dx * 0.001, yi_rect[k] * self.dy * 0.001),
                                                 nx_rect[k] * self.dx * 0.001,
                                                 ny_rect[k] * self.dy * 0.001, linewidth=linewidth_rect[k],
                                                 edgecolor=colors_rect[k],
                                                 facecolor='none')
                        ax.add_patch(rect)
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005, wrap=True)
                # Colorbar
                if j == ncols - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0)
                    cb = colorbar(I, extend='neither', cax=cax)
                    cb.ax.tick_params(axis='y', direction='out', labelsize=font_size, width=1, length=2.5)
                    cb.set_label(colorbar_label[i][j], labelpad=20.0, rotation=270, size=font_size)
    
        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')

    def plot_vector_maps_shared(self, filename_output, img, vx, vy, title_label, min_cb, max_cb, colorbar_label,
                                imgformat='png', xi_crop=[[0]], yi_crop=[[0]], nx_crop=[[50]], ny_crop=[[50]],
                                qk_length=[[0.5]],
                                scale=[[1.0]], width=[[0.1]], headwidth=[[2]], headlength=[[4]], headaxislength=[[4.5]],
                                step=[[1]],
                                x_majorticks=[[1]], y_majorticks=[[1]],
                                xi_rect=[], yi_rect=[], nx_rect=[], ny_rect=[], colors_rect=[], linewidth_rect=[]):
    
        # Dimensions
        nrows, ncols, ny, nx = img.shape
    
        # Vector parameters
        dx = self.dx * 0.001
        vunit = self.dx / self.dt
        qk_x = 0.750
        qk_y = 0.015
    
        # Position
        x_label = 'x (Mm)'
        y_label = 'y (Mm)'
    
        # Properties
        font_size = 13
        figsize_x = ncols * 5.25 * (1.0 / ncols + (ncols - 1) * (0.05 / ncols + 1.00 / 1.05 * 0.72 / ncols))
        figsize_y = nrows * 4.5 * (1.0 / nrows + (nrows - 1) * (0.10 / nrows + 0.80 / nrows))
        nb_rect = len(xi_rect)
        fig = plt.figure(figsize=(figsize_x, figsize_y), constrained_layout=False)
    
        # Rows
        for i in range(nrows):
            # Cols
            for j in range(ncols):
                img_plot = img[i, j, yi_crop[i][j]:yi_crop[i][j] + ny_crop[i][j],
                           xi_crop[i][j]:xi_crop[i][j] + nx_crop[i][j]]
                vx_plot = vx[i, j, yi_crop[i][j]:yi_crop[i][j] + ny_crop[i][j],
                          xi_crop[i][j]:xi_crop[i][j] + nx_crop[i][j]] / vunit * dx
                vy_plot = vy[i, j, yi_crop[i][j]:yi_crop[i][j] + ny_crop[i][j],
                          xi_crop[i][j]:xi_crop[i][j] + nx_crop[i][j]] / vunit * dx
                extent = np.array(
                    [xi_crop[i][j], xi_crop[i][j] + nx_crop[i][j], yi_crop[i][j], yi_crop[i][j] + ny_crop[i][j]]) * dx
                x_plot = np.arange(xi_crop[i][j], xi_crop[i][j] + nx_crop[i][j], dtype='float') * dx
                y_plot = np.arange(yi_crop[i][j], yi_crop[i][j] + ny_crop[i][j], dtype='float') * dx
                v_slice = np.s_[::step[i][j], ::step[i][j]]
                # Grid of the flow fields, first define the default, then slice it to match anchors spaced by "step" pixels
                x_plot = x_plot[::step[i][j]]
                y_plot = y_plot[::step[i][j]]
                # Colormap
                if min_cb[i][j] * max_cb[i][j] < 0:
                    colorbar_cmap = 'RdBu_r'
                else:
                    colorbar_cmap = 'GnBu_r'
                # Plot
                new_spec = (1.0 / ncols + (ncols - 1) * (0.05 / ncols + 1.00 / 1.05 * 0.72 / ncols))
                ver_spec = (1.0 / nrows + (nrows - 1) * (0.10 / nrows + 0.80 / nrows))
                bottom_spec = 0.10 / nrows + float((nrows - 1 - i)) / nrows
                top_spec = float(nrows - i) / nrows - 0.10 / nrows
                if j == 0:
                    left_spec = 0.14 / ncols / new_spec
                    right_spec = (0.14 / ncols + 1.00 / 1.05 * 0.72 / ncols) / new_spec
            
                elif 0 < j < ncols - 1:
                    left_spec = (0.14 / ncols + 1.00 / 1.05 * 0.72 / ncols) / new_spec + (j - 1) * (
                                1.00 / 1.05 * 0.72 / ncols) / new_spec + (j) * (0.05 / ncols) / new_spec
                    right_spec = left_spec + (1.00 / 1.05 * 0.72 / ncols) / new_spec
                else:
                    left_spec = 1 - 0.86 / ncols / new_spec
                    right_spec = 1 - 0.14 / ncols / new_spec
                if i == 0:
                    bottom_spec = 1.0 - (0.9 / nrows) / ver_spec
                    top_spec = 1.0 - (0.1 / nrows) / ver_spec
                elif 0 < i < nrows - 1:
                    bottom_spec = (nrows - i - 1) / nrows / ver_spec
                    top_spec = bottom_spec + 0.80 / nrows / ver_spec
                else:
                    bottom_spec = (0.10 / nrows) / ver_spec
                    top_spec = bottom_spec + (0.80 / nrows) / ver_spec
                spec = fig.add_gridspec(nrows=1, ncols=1, left=left_spec,
                                        right=right_spec,
                                        bottom=bottom_spec,
                                        top=top_spec, wspace=0.00)
                ax = fig.add_subplot(spec[:, :])
                I = ax.imshow(img_plot, extent=extent, cmap=colorbar_cmap, aspect=1,
                              interpolation='none',
                              vmin=min_cb[i][j], vmax=max_cb[i][j], alpha=1.0, origin='lower')
                q = ax.quiver(x_plot, y_plot, vx_plot[::step[i][j], ::step[i][j]], vy_plot[::step[i][j], ::step[i][j]],
                              units='xy', scale=scale[i][j], width=width[i][j], headwidth=headwidth[i][j],
                              headlength=headlength[i][j],
                              headaxislength=headaxislength[i][j], pivot='tail')
                qk_label = str(np.around(qk_length[i][j], decimals=2))
                qk = ax.quiverkey(q, left_spec + 0.5 / ncols / new_spec, 0.015 / nrows + i / nrows,
                                  qk_length[i][j] / vunit * dx,
                                  qk_label + r' km s$^{-1}$', labelpos='E',
                                  coordinates='figure', fontproperties={'size': str(font_size)},
                                  labelsep=0.05)
                if (j == 0):
                    ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5,
                                                   labelsize=font_size,
                                                   left=True, right=True)
                    ax.set_ylabel(y_label, fontsize=font_size, labelpad=5.0)
                else:
                    ax.get_yaxis().set_tick_params(which='both', direction='out', width=1, length=2.5,
                                                   labelsize=font_size,
                                                   left=True, right=True, labelleft=False, labelright=False)
                if (i == nrows - 1):
                    ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5,
                                                   labelsize=font_size,
                                                   bottom=True, top=True)
                    ax.set_xlabel(x_label, fontsize=font_size, labelpad=3.0)
                else:
                    ax.get_xaxis().set_tick_params(which='both', direction='out', width=1, length=2.5,
                                                   labelsize=font_size,
                                                   bottom=True, top=True, labelbottom=False, labeltop=False)
                ax.get_xaxis().set_major_locator(plt.MultipleLocator(x_majorticks[i][j]))
                ax.get_yaxis().set_major_locator(plt.MultipleLocator(y_majorticks[i][j]))
                if nb_rect > 0:
                    for k in range(nb_rect):
                        rect = patches.Rectangle((xi_rect[k] * self.dx * 0.001, yi_rect[k] * self.dy * 0.001),
                                                 nx_rect[k] * self.dx * 0.001,
                                                 ny_rect[k] * self.dy * 0.001, linewidth=linewidth_rect[k],
                                                 edgecolor=colors_rect[k],
                                                 facecolor='none')
                        ax.add_patch(rect)
                ax.set_title(title_label[i][j], fontsize=font_size, y=1.005, wrap=True)
                # Colorbar
                if j == ncols - 1:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0)
                    cb = colorbar(I, extend='neither', cax=cax)
                    cb.ax.tick_params(axis='y', direction='out', labelsize=font_size, width=1, length=2.5)
                    cb.set_label(colorbar_label[i][j], labelpad=20.0, rotation=270, size=font_size)
    
        # Save
        plt.draw()
        plt.savefig(filename_output, format=imgformat, dpi=300)
        plt.close('all')
