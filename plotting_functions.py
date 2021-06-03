"""plotting_functions

Functions for plotting pulses.
All values are in SI units (m, s, T, etc) unless otherwise noted.

Dependencies: 
    sigpy (https://sigpy.readthedocs.io/en/latest/mri_rf.html)
    numpy
    scipy
    matplotlib

Author: Victor Han
Last Modified: 6/2/21

"""


import numpy as np
import sigpy as sp
import sigpy.mri as mr
import sigpy.mri.rf as rf
import sigpy.plot as pl
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.special import *
from scipy.integrate import odeint
import matplotlib.gridspec as gridspec
import csv

from matplotlib.patches import Rectangle

from various_constants import *


def plot_waveform(fig, subplot, t, RF_mag, RF_phase, B1z, Gz, zoom_time=[2.9, 3.1]):
    """Plots pulse sequence waveforms in a subplot for the slice selective excitation.

    Args:
        fig: The figure to plot on
        subplot: The subplot of the figure to plot on
        t: Time axis to plot
        RF_mag: Magnitude of B1xy to plot
        RF_phase: Phase of B1xy to plot
        B1z: B1z to plot
        Gz: Slice selective gradient to plot
        zoom_time: Time period in ms to make a zoomed-in view of
    """

    t = t * 1000 # Change units to ms

    # Set some font sizes
    plt.rc('font', size=DEFAULT_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=DEFAULT_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=DEFAULT_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=DEFAULT_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=DEFAULT_SIZE)   # fontsize of the figure title

    # Create subplots for each waveform inside the given subplot
    subsubplot = gridspec.GridSpecFromSubplotSpec(4, 4, subplot_spec=subplot, wspace=0.1, hspace=0)

    # Plot B1xy magnitude
    ax = plt.Subplot(fig, subsubplot[0:3])
    ax.plot(t, RF_mag*1e6)
    ax.set_xticks([])
    ax.set_ylabel('|$B_{1xy}$| (uT)', rotation=0, labelpad=60)
    ylimits = ax.get_ylim()
    rect = Rectangle( (zoom_time[0], ylimits[0]), zoom_time[1]-zoom_time[0], ylimits[1]-ylimits[0], linestyle='dashed', facecolor='none', edgecolor='red')
    ax.add_patch(rect)
    fig.add_subplot(ax)

    # Plot B1xy phase
    ax = plt.Subplot(fig, subsubplot[4:7])
    ax.plot(t, RF_phase*180/np.pi)
    ax.set_xticks([])
    ax.set_ylim([-180,180])
    ax.set_ylabel(r'$\angle B_{1xy}$ (deg)', rotation=0, labelpad=40)
    ylimits = ax.get_ylim()
    rect = Rectangle( (zoom_time[0], ylimits[0]), zoom_time[1]-zoom_time[0], ylimits[1]-ylimits[0], linestyle='dashed', facecolor='none', edgecolor='red')
    ax.add_patch(rect)
    fig.add_subplot(ax)

    # Plot B1z
    ax = plt.Subplot(fig, subsubplot[8:11])
    ax.plot(t, B1z*1e3)
    ax.set_xticks([])
    ax.set_ylim([-0.3,0.3])
    ax.set_ylabel('$B_{1z}$ (mT)', rotation=0, labelpad=35)
    ylimits = ax.get_ylim()
    rect = Rectangle( (zoom_time[0], ylimits[0]), zoom_time[1]-zoom_time[0], ylimits[1]-ylimits[0], linestyle='dashed', facecolor='none', edgecolor='red')
    ax.add_patch(rect)
    fig.add_subplot(ax)

    # Plot Gz
    ax = plt.Subplot(fig, subsubplot[12:15])
    ax.plot(t, Gz*1e3)
    ax.set_ylim([-40,40])
    ax.set_ylabel('$G_{z}$ (mT/m)', rotation=0, labelpad=50)
    ax.set_xlabel('Time (ms)')
    ylimits = ax.get_ylim()
    rect = Rectangle( (zoom_time[0], ylimits[0]), zoom_time[1]-zoom_time[0], ylimits[1]-ylimits[0], linestyle='dashed', facecolor='none', edgecolor='red')
    ax.add_patch(rect)
    fig.add_subplot(ax)

    # Plot zoomed in
    ax = plt.Subplot(fig, subsubplot[3])
    ax.plot(t, RF_mag*1e6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(zoom_time)
    fig.add_subplot(ax)


    ax = plt.Subplot(fig, subsubplot[7])
    ax.plot(t, RF_phase*180/np.pi)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(zoom_time)
    ax.set_ylim([-180,180])
    fig.add_subplot(ax)

    ax = plt.Subplot(fig, subsubplot[11])
    ax.plot(t, B1z*1e3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim([-0.3,0.3])
    ax.set_xlim(zoom_time)
    fig.add_subplot(ax)

    ax = plt.Subplot(fig, subsubplot[15])
    ax.plot(t, Gz*1e3)
    ax.set_yticks([])
    ax.set_ylim([-40,40])
    ax.set_xlim(zoom_time)
    ax.set_xlabel('Close-up (ms)')
    fig.add_subplot(ax)


def plot_sim(fig, subplot, z, M_mag, M_phase):
    """Plots simulation results in a subplot for the slice selective excitation.

    Args:
        fig: The figure to plot on
        subplot: The subplot of the figure to plot on
        z: Position axis to plot
        M_mag: Magnitude of xy magnitization to plot
        M_phase: Phase of xy magnitization to plot
    """

    # Set some font sizes
    plt.rc('font', size=DEFAULT_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=DEFAULT_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=DEFAULT_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=DEFAULT_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=DEFAULT_SIZE)   # fontsize of the figure title

    # Create subplots for each waveform inside the given subplot
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=subplot, wspace=0.1, hspace=0)

    # Plot magnitude of xy magnitization vs position
    ax = plt.Subplot(fig, inner[0])
    ax.plot(z*1e3, M_mag)
    ax.set_xticks([])
    ax.set_ylim([0,1])
    ax.set_ylabel('|$M_{xy}$|', rotation=0, labelpad=25)
    fig.add_subplot(ax)

    # Plot phase of xy magnitization vs position
    ax = plt.Subplot(fig, inner[1])
    ax.plot(z*1e3, M_phase*180/np.pi)
    ax.set_ylim([-180,180])
    ax.set_xlabel('Position (mm)')
    ax.set_ylabel(r'$\angle M_{xy}$ (deg)', rotation=0, labelpad=30)
    fig.add_subplot(ax)

def plot_experiment(fig, subplot, image_data_filename, ylim=[0,40000]):
    """Plots experimental results in a subplot for the slice selective excitation.

    Args:
        fig: The figure to plot on
        subplot: The subplot of the figure to plot on
        image_name: Name of image file
        ylim: Limits on y-axis of line plot of experimental data. Arbitrary units.
    """

    # Set some font sizes
    plt.rc('font', size=DEFAULT_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=DEFAULT_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=DEFAULT_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALLER_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=DEFAULT_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=DEFAULT_SIZE)   # fontsize of the figure title


    # Create subplots for each waveform inside the given subplot
    inner = gridspec.GridSpecFromSubplotSpec(5, 4, subplot_spec=subplot, wspace=0, hspace=0)

    im = np.load(image_data_filename)
    image_fov = 0.22
    image_n = 512
    image_n_to_keep = int(np.round(XLIM*2/image_fov * image_n))
    if image_n_to_keep % 2 == 1:
        image_n_to_keep = image_n_to_keep + 1 # Want to make it an even number
    image_start = int((image_n - image_n_to_keep) / 2)
    im = im[int(image_start/2):int((image_n/2-image_start/2)), image_start:(image_n-image_start)]
    image_fov_new = image_fov*image_n_to_keep/image_n *1000 # in mm

    # Plot magnitude of xy magnitization vs position
    ax = plt.Subplot(fig, inner[0:4])
    ax.plot(np.linspace(-image_fov_new/2,image_fov_new/2, image_n_to_keep), np.abs(im[int(image_n_to_keep/4),:]), linewidth=1)
    ax.set_ylim(ylim)
    ax.set_xlim([-image_fov_new/2,image_fov_new/2])
    ax.set_aspect(1/4./ax.get_data_ratio())
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel('Line Plot', rotation=0, labelpad=30)
    fig.add_subplot(ax)

    # Plot phase of xy magnitization vs position
    ax = plt.Subplot(fig, inner[4:20])
    ax.imshow(np.abs(im),cmap='gray', extent=[-image_fov_new/2,image_fov_new/2,-image_fov_new/2,image_fov_new/2])
    ax.set_xlabel('Position (mm)')
    ax.set_yticks([])
    ax.set_ylabel('Image', rotation=0, labelpad=25)
    fig.add_subplot(ax)

