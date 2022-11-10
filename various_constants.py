"""cosntants

Physical constants and constants for simulations.
All values are in SI units (m, s, T, etc) unless otherwise noted.

Dependencies: 
    sigpy (https://sigpy.readthedocs.io/en/latest/mri_rf.html)
    numpy
    scipy
    matplotlib

Author: Victor Han
Last Modified: 7/7/22

"""

import numpy as np

#############################
# Physical constants

# T1 relaxation used in simulations
T1 = 10e10;
# T2 relaxation used in simulations
T2 = 10e10;
# Inital magnetization
M0 = 1;
# Gyromagnetic ratio in rad/s/T
g = 2*np.pi*42.5759e6;
# Maximum RF allowed in uT. Currently unused
RF_LIMIT = 14.7e-6
# Maximum gradient strength in T/m
GRAD_LIMIT = 3.2934e-2 
# Maximum gradient slew rate in T/m/s
SLEW_LIMIT = 119.064/2  


#############################
# RF pulse property constants

# Flip angle in degrees
FA = 30
# Time-bandwidth product of pulse
TB = 6
# Time-bandwidth product of pulse for figure 4
TB_FIG4 = 6
# Time-bandwidth product of pulse for figure 5
TB_FIG5 = 6
# Time-bandwidth product of pulse for figure 6
TB_FIG6 = 6
# Duration of pulse in seconds
PULSE_DURATION = 0.006
# Duration of pulses in figure 4 in seconds
PULSE_DURATION_FIG4 = 0.024
# Duration of pulses in figure 5 in seconds
PULSE_DURATION_FIG5 = 0.012
# Duration of pulses in figure 6 in seconds
PULSE_DURATION_FIG6 = 0.012
# Number of points in pulse
N = int(PULSE_DURATION*500*1000) # 2us per point
# Number of points in pulse
N_FIG4 = int(PULSE_DURATION_FIG4*500*1000) # 2us per point
# Number of points in pulse
N_FIG5 = int(PULSE_DURATION_FIG5*500*1000) # 2us per point
# Number of points in pulse
N_FIG6 = int(PULSE_DURATION_FIG6*500*1000) # 2us per point
# Slice thickness in m
THICKNESS = 0.005
# Slice select gradient value based on pulse parameters
SLICE_PEAK = TB/PULSE_DURATION * 2*np.pi/g * 1/THICKNESS
# Slice select gradient value based on pulse parameters
SLICE_PEAK_FIG4 = TB_FIG4/PULSE_DURATION_FIG4 * 2*np.pi/g * 1/THICKNESS
# Slice select gradient value based on pulse parameters
SLICE_PEAK_FIG5 = TB_FIG5/PULSE_DURATION_FIG5 * 2*np.pi/g * 1/THICKNESS
# Slice select gradient value based on pulse parameters
SLICE_PEAK_FIG6 = TB_FIG6/PULSE_DURATION_FIG6 * 2*np.pi/g * 1/THICKNESS
# Amplitude of uniform B1z
B1Z_AMP = 0.17e-3
# Frequency of uniform B1z
FZ = 25e3


#############################
# Simulation constants

# Number of spatial points for slice selective simulations
XRES = 301
# Limit on spatial position for simulations. Simulations go from -xlim to xlim
XLIM = 20e-3
# Limit on spatial position for simulations. Simulations go from -xlim to xlim
XLIM_FIG5 = 40e-3
# Limit on spatial position for simulations. Simulations go from -xlim to xlim
XLIM_FIG6 = 40e-3


#############################
# Plotting constants

# Number of columns that the pulse sequence spans when plotting
SEQUENCE_PLOT_END = 4
# Number of points to plot for waveforms in pulse sequence
WAVEFORM_RES = int(1e5)
# Default plot font size
DEFAULT_SIZE = 12
# Smaller plot font size
SMALLER_SIZE = 10
# Font size for category labels
CATEGORY_SIZE = 14


#############################
# Other constants

# Whether or not waveform files are written out for experimental use
WRITE_WAVEFORM_FILES = False
# Whether or not to print max waveform values (helpful for experimental implementation)
PRINT_MAX_VALS = True
# Time between points
DT = 2e-6