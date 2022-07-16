"""pulse_generator_functions

Functions for generating RF pulses.
All values are in SI units (m, s, T, etc) unless otherwise noted.

Dependencies: 
    sigpy (https://sigpy.readthedocs.io/en/latest/mri_rf.html)
    numpy
    scipy
    matplotlib

Author: Victor Han
Last Modified: 7/1/22

"""

import numpy as np
import sigpy as sp
import sigpy.mri as mr
import sigpy.mri.rf as rf
import sigpy.plot as pl
import scipy.signal as signal
import matplotlib.pyplot as plt

from scipy.special import *
from scipy.integrate import odeint
import matplotlib.gridspec as gridspec
import csv

from various_constants import *
from simulation_functions import *

def write_rf_pulse_for_heartvista(pulse, filename):
    """Saves a B1xy RF pulse for use with Spinbench/HeartVista. 
    Note though that for version 2.5.1 of Spinbench, we had to manually edit 
    the file after importing to get it correctly imported. Contact us if more 
    information is desired.

    Args:
        pulse: The pulse to save
        filename: Name of output file
    """

    # Convert to Big Endian Format
    scaling_factor=2**15-1
    rescaled=np.zeros(pulse.shape[0]*2)
    rescaled[0:pulse.shape[0]]=np.real(pulse)
    rescaled[pulse.shape[0]:2*pulse.shape[0]]=np.imag(pulse)
    rescaled=rescaled*scaling_factor/np.max(np.abs(rescaled))
    testpulse=rescaled.astype('>i2')
    # Save File
    f = open(filename, 'w+b')
    f.write(testpulse)
    f.close()


def write_gz_pulse_for_heartvista(gz_pulse, slice_peak, pulse_duration, filename, area_offset=0):
    """Saves a gradient pulse for use with HeartVista.

    Args:
        gz_pulse: The pulse to save
        slice_peak: Plateau value for slice select gradient for desired slice thickness
        pulse_duration: Duration of RF pulse
        filename: Name of output file
    """

    t1 = 0
    gz_to_save = []
    while (gz_waveform(t1, slice_peak, pulse_duration, gz_pulse, area_offset=area_offset) != 0) or (t1 < pulse_duration+4*slice_peak/SLEW_LIMIT):
        gz_to_save.append(gz_waveform(t1, slice_peak, pulse_duration, gz_pulse))
        t1 = t1 + 2e-6
    scaling_factor=2**15-1
    rescaled=np.real(gz_to_save)*scaling_factor/np.max(np.abs(gz_to_save))
    savepulse=rescaled.astype('>i2')
    # Save pulse file
    f = open(filename, 'w+b')
    f.write(savepulse)
    f.close()


def slr_pulse(N, tb, FA, freq=0, phase=0, d1=0.001, d2=0.001, ptype='st', ftype='ls', name='slr'):
    """Generates a SLR RF pulse for use with simulations and HeartVista.

    Args:
        N: The number of data points in the waveform
        tb: The time-bandwidth product of the pulse
        freq: Frequency in Hz to decrease the RF carrier frequency
        phase: Initial phase of the RF pulse
        d1: Passband ripple (see sigpy)
        d2: Stopband ripple (see sigpy)
        ptype: Pulse type (see sigpy)
        ftype: Filter type (see sigpy)
        name: Name of file to save

    Returns:
        pulse: A RF pulse designed with the SLR algorithm.
    """

    # Generate SLR pulse with sigpy
    pulse = rf.slr.dzrf(N, tb, ptype, ftype, d1, d2, False)
    pulse = pulse.astype(complex)

    # Scale the pulse to have the given flip angle
    pulse = pulse / np.max(np.abs(pulse))
    FA_norm = 1e-6*g*np.trapz(np.real(pulse), dx=DT)*180/np.pi
    pulse = 1e-6 * pulse * FA/FA_norm

    # Shift the pulse in frequency
    for i in range(N):
        pulse[i] = pulse[i] * np.complex(np.cos(DT*i*2*np.pi*freq), np.sin(DT*i*2*np.pi*freq))

    # Change the initial phase of the pulse
    pulse = pulse * np.complex(np.cos(phase), np.sin(-1*phase))

    # Write pulse for use with HeartVista if desired
    if WRITE_WAVEFORM_FILES:
        write_rf_pulse_for_heartvista(pulse, name)

    return pulse

def fm_pulse(N, tb, FA, freq, B1z, phase=0, d1=0.001, d2=0.001, ptype='st', ftype='ls', name='slr_fm'):
    """Generates a frequency modulated SLR RF pulse for use with simulations and HeartVista.
    The frequency modulation simulates the effect of a B1z hard pulse with a certain frequency and amplitude.
    It is assumed that only a sideband of the pulse, corresponding to a two-photon pulse, will be used for excitation.

    Args:
        N: The number of data points in the waveform
        tb: The time-bandwidth product of the pulse
        freq: Frequency in Hz to decrease the RF carrier frequency and equivalent B1z frequency
        B1z: The equivalent B1z amplitude that the frequency modulation simulates
        phase: Initial phase of the RF pulse
        d1: Passband ripple (see sigpy)
        d2: Stopband ripple (see sigpy)
        ptype: Pulse type (see sigpy)
        ftype: Filter type (see sigpy)
        name: Name of file to save

    Returns:
        pulse: A RF pulse designed with the SLR algorithm.
    """

    # Generate SLR pulse with sigpy
    pulse = rf.slr.dzrf(N, tb, ptype, ftype, d1, d2, False)
    pulse = pulse.astype(complex)

    # Scale the pulse to have the given flip angle
    pulse = pulse / np.max(np.abs(pulse))
    FA_norm = 1e-6*g*np.trapz(np.real(pulse), dx=DT)*180/np.pi
    pulse = 1e-6 * pulse * FA/FA_norm

    gBz_w = g/(2*np.pi) * B1z/freq;
    pulse_duration = N * DT;

    for i in range(N):
        # Do a frequency modulation
        pulse[i] = pulse[i] * np.complex(np.cos(gBz_w-1*gBz_w*np.cos(i/N*pulse_duration*2*np.pi*freq)), np.sin(gBz_w-1*gBz_w*np.cos(i/N*pulse_duration*2*np.pi*freq)))
        # Shift the pulse in frequency
        pulse[i] = pulse[i] * np.complex(np.cos(DT*i*2*np.pi*freq), np.sin(DT*i*2*np.pi*freq))
    # Scale the pulse to compensate for lowered sideband efficiency
    pulse = pulse / (j1(gBz_w))

    # Change the initial phase of the pulse
    pulse = pulse * np.complex(np.cos(phase), np.sin(-1*phase))

    # Write pulse for use with HeartVista if desired
    if WRITE_WAVEFORM_FILES:
        write_rf_pulse_for_heartvista(pulse, name)
    
    return pulse

def make_b1z_csv(pulse, slice_peak, pulse_duration, name, dc_value=0):
    """Saves a B1z RF pulse for use with a Siglent SDG6022X function generator.

    Args:
        pulse: The pulse to save
        duration: Total duration of the pulse
        name: Name of output file
        dc_value: DC offset on Bz
    """

    t1 = 0
    b1z_to_save = []
    while (bz_waveform(t1, slice_peak, pulse_duration, pulse, dc_value=dc_value) != 0) or (t1 < pulse_duration+2*slice_peak/SLEW_LIMIT):
        b1z_to_save.append(bz_waveform(t1, slice_peak, pulse_duration, pulse, dc_value=dc_value))
        t1 = t1 + 2e-6

    pulse = b1z_to_save / np.max(np.abs(pulse))
    with open(name, mode='w') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerow(['data length',len(pulse)])
        writer.writerow(['frequency',1/t1])
        writer.writerow(['amp',1])
        writer.writerow(['offset',0])
        writer.writerow(['phase',0])
        for i in range(7):
            writer.writerow([])
        writer.writerow(['xpos','value'])
        for i,val in enumerate(pulse):
            writer.writerow([i+1,val])
