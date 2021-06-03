"""cosntants

Physical constants and constants for simulations.
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
SLEW_LIMIT = 119.064  


#############################
# RF pulse property constants

# Flip angle in degrees
FA = 30
# Time-bandwidth product of pulse
TB = 3
# Time-bandwidth product of pulse for figure 4
TB_FIG4 = 3
# Duration of pulse in seconds
PULSE_DURATION = 0.003
# Duration of pulses in figure 4 in seconds
PULSE_DURATION_FIG4 = 0.012
# Number of points in pulse
N = int(PULSE_DURATION*500*1000) # 2us per point
# Number of points in pulse
N_FIG4 = int(PULSE_DURATION_FIG4*500*1000) # 2us per point
# Slice thickness in m
THICKNESS = 0.005
# Slice select gradient value based on pulse parameters
SLICE_PEAK = TB/PULSE_DURATION * 2*np.pi/g * 1/THICKNESS
# Slice select gradient value based on pulse parameters
SLICE_PEAK_FIG4 = TB_FIG4/PULSE_DURATION_FIG4 * 2*np.pi/g * 1/THICKNESS
# Amplitude of uniform B1z
B1Z_AMP = 0.2e-3
# Frequency of uniform B1z
FZ = 25e3


#############################
# Simulation constants

# Number of spatial points for slice selective simulations
XRES = 101
# Limit on spatial position for simulations. Simulations go from -xlim to xlim
XLIM = 20e-3


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
PRINT_MAX_VALS = False
# Time between points
DT = 2e-6