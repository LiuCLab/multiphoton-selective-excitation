"""simulation_functions

Functions for performing Bloch equation simulations of pulses.
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

from scipy.special import *
from scipy.integrate import odeint
import matplotlib.gridspec as gridspec
import csv

from various_constants import *




def gz_waveform(t, slice_peak, pulse_duration, pulse, area_offset=0, slew_limit=SLEW_LIMIT, grad_limit=GRAD_LIMIT):
    """Returns value of slice select gradient at a given time.
    This function interpolates the given waveform and generates ramping and rewinding gradients as needed.

    Args:
        t: Time point for desired value to be returned
        slice_peak: Plateau value for slice select gradient for desired slice thickness
        pulse_duration: Duration of RF pulse
        pulse: Array of values defining a gradient waveform to be added to the slice select gradient
        area_offset: Area to reduce the rewinding gradient area by
        slew_limit: Gradient slew limit
        grad_limit: Maximum gradient value used for rewinder

    Returns:
        The value of the slice select gradient at the desired time
    """

    # Calculate some timings based on given values
    rise_time = slice_peak/slew_limit
    rewind_area = 0.5*rise_time*slice_peak + 0.5*pulse_duration*slice_peak - area_offset
    rewind_rise_time = grad_limit/slew_limit

    trap_flat_time = 0
    if (rewind_area <= grad_limit*rewind_rise_time):
        trap_flat_time = 0
        rewind_rise_time = np.sqrt(rewind_area/slew_limit)
    else:
        trap_flat_time = (rewind_area - grad_limit*rewind_rise_time)/grad_limit

    if (t<=rise_time): 
        # Initial rise to gradient plateau
        return t*slew_limit
    elif (t<=rise_time+pulse_duration): 
        # Gradient plateau superimposed with interpolated additional waveform
        t = t-rise_time
        pos = t / pulse_duration * len(pulse)
        if np.ceil(pos) >= len(pulse):
            return slice_peak + ((pos-np.floor(pos))*(0 - pulse[int(np.floor(pos))]) + pulse[int(np.floor(pos))]) 
        else:
            return slice_peak + ((pos-np.floor(pos))*(pulse[int(np.ceil(pos))] - pulse[int(np.floor(pos))]) + pulse[int(np.floor(pos))]) 
    elif (t<=2*rise_time+pulse_duration):
        # Fall to zero gradient
        return slice_peak - (t-(rise_time+pulse_duration))*slew_limit
    elif (t<=2*rise_time+pulse_duration+rewind_rise_time):
        # Increasing negative gradient for rewind
        time = t - (2*rise_time+pulse_duration)
        return -1*time*slew_limit
    elif (t<=2*rise_time+pulse_duration+rewind_rise_time + trap_flat_time):
        # Rewind gradient plateau
        return -1*grad_limit
    elif (t<=2*rise_time+pulse_duration+2*rewind_rise_time + trap_flat_time):
        # Back to zero gradient from rewind
        time = (2*rise_time+pulse_duration+2*rewind_rise_time+trap_flat_time) - t
        return -1*time*slew_limit
    else:
        # Zero gradient at all other times
        return 0


def bxy_waveform(t, slice_peak, pulse_duration, pulse, slew_limit=SLEW_LIMIT):
    """Returns value of B1xy at a given time with gradient rise and fall time information. 
    This function interpolates the given waveform as needed.

    Args:
        t: Time point for desired value to be returned
        slice_peak: Plateau value for slice select gradient for desired slice thickness
        pulse_duration: Duration of RF pulse
        pulse: Array of values defining a B1xy waveform
        slew_limit: Gradient slew limit

    Returns:
        The value of B1xy at the desired time
    """

    rise_time = slice_peak/slew_limit
    if t<rise_time:
        # Initial rise to gradient plateau
        return 0
    elif t<=(rise_time+pulse_duration):
        # During gradient plateau, return an interpolation of pulse
        t = t - rise_time
        pos = t / pulse_duration * len(pulse)
        if np.ceil(pos) >= len(pulse):
            # Assume that the next value to interpolate to is 0
            return ((pos-np.floor(pos))*(0 - pulse[int(np.floor(pos))]) + pulse[int(np.floor(pos))]) 
        else:
            return ((pos-np.floor(pos))*(pulse[int(np.ceil(pos))] - pulse[int(np.floor(pos))]) + pulse[int(np.floor(pos))]) 
    else:
        # No RF otherwise
        return 0


def bz_waveform(t, slice_peak, pulse_duration, pulse, dc_value=0, slew_limit=SLEW_LIMIT):
    """Returns value of B1z at a given time with gradient rise and fall time information. 
    This function interpolates the given waveform as needed.

    Args:
        t: Time point for desired value to be returned
        slice_peak: Plateau value for slice select gradient for desired slice thickness
        pulse_duration: Duration of RF pulse
        pulse: Array of values defining a B1xy waveform
        slew_limit: Gradient slew limit

    Returns:
        The value of B1z at the desired time
    """

    rise_time = slice_peak/slew_limit
    if t<rise_time:
        # Initial rise to gradient plateau
        return t/rise_time * dc_value
    elif t<=(rise_time+pulse_duration):
        # During gradient plateau, return an interpolation of pulse
        t = t - rise_time
        pos = t / pulse_duration * len(pulse)
        if np.ceil(pos) >= len(pulse):
            # Assume that the next value to interpolate to is the DC value
            return ((pos-np.floor(pos))*(dc_value - pulse[int(np.floor(pos))]) + pulse[int(np.floor(pos))]) 
        else:
            bz_val = ((pos-np.floor(pos))*(pulse[int(np.ceil(pos))] - pulse[int(np.floor(pos))]) + pulse[int(np.floor(pos))])
            return bz_val + dc_value
    elif t<=(2*rise_time+pulse_duration):
        # Return DC B1z to zero if needed
        t = t - (rise_time+pulse_duration)
        return (1-t/rise_time) * dc_value
    else:
        # No B1z otherwise
        return 0



def bloch(M, t, x, y, bxy_pulse, bz_pulse, gz_pulse, slice_peak, pulse_duration, area_offset=0, b1z_dc_value=0, slew_limit=SLEW_LIMIT):
    """Returns the derivative of the magnetization vector at a given time for given B1xy, B1z, Gz pulses and other parameters.
    This function is used in a numerical ODE solver to solve the Bloch equations.

    Args:
        M: Current magnetization
        t: Current time
        x: x spatial position
        y: y spatial position (currently unused)
        bxy_pulse: B1xy waveform
        bz_pulse: B1z waveform
        gz_pulse: Slice select gradient waveform
        slice_peak: Plateau value for slice select gradient for desired slice thickness
        pulse_duration: RF pulse duration
        area_offset: Area to reduce the rewinding gradient area by
        b1z_dc_value: DC value on Bz pulse
        slew_limit: Gradient slew limit

    Returns:
        The derivative of the magnetization vector at the desired time
    """

    rise_time = slice_peak/slew_limit

    Bz = gz_waveform(t, slice_peak, pulse_duration, gz_pulse, area_offset) * x
    Bz = Bz + bz_waveform(t, slice_peak, pulse_duration, bz_pulse, dc_value=b1z_dc_value)

    Bxy = bxy_waveform(t, slice_peak, pulse_duration, bxy_pulse)

   
    Bx = np.real(Bxy)
    By = np.imag(Bxy) 
    Mat = np.array([[-1/T2, g*Bz, -g*By], [-g*Bz, -1/T2, g*Bx], [g*By, -g*Bx, -1/T1]])
    dMdt = Mat@np.reshape(M, (3,1)) + np.array([[0, 0, M0/T1]]).T;
    return np.reshape(dMdt,(3,))

