"""Supplementary Figure 2

Supplemental file to run to create supplementary figure 2.
All values are in SI units (m, s, T, etc) unless otherwise noted.

Dependencies: 
    sigpy (https://sigpy.readthedocs.io/en/latest/mri_rf.html)
    numpy
    scipy
    matplotlib

Author: Victor Han
Last Modified: 11/15/22

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
from matplotlib.transforms import Bbox
import csv

from various_constants import *
from pulse_generator_functions import *
from simulation_functions import *
from plotting_functions import *


#################################################################
# Make Figure s2: equivalent one-photon, two-photon, and freq mod slice select using optimal B1z and shorter pulse duration

# Make a grid of subplots and label the rows and columns
fig = plt.figure(figsize=(20, 10))
cols = 1+SEQUENCE_PLOT_END # Number of columns used for pulse sequence plot
outer = gridspec.GridSpec(4, cols, wspace=1, hspace=0.3)
ax = plt.Subplot(fig, outer[cols])
t = ax.text(0.7,0.2, 'One-Photon', fontsize=CATEGORY_SIZE, rotation=90)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)
ax = plt.Subplot(fig, outer[2*cols])
t = ax.text(0.7,0.2, 'Two-Photon', fontsize=CATEGORY_SIZE, rotation=90)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)
ax = plt.Subplot(fig, outer[3*cols])
t = ax.text(0.7,-0.1, 'Frequency Modulation', fontsize=CATEGORY_SIZE, rotation=90)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)

ax = plt.Subplot(fig, outer[1:SEQUENCE_PLOT_END])
t = ax.text(0.5,0, 'Pulse Sequence', fontsize=CATEGORY_SIZE)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)
ax = plt.Subplot(fig, outer[SEQUENCE_PLOT_END])
t = ax.text(0.5,0, 'Simulation Slice Profile', fontsize=CATEGORY_SIZE)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)


# Make the pulse duration short, but not so short that we exceed max B1xy limits
PULSE_DURATION = 0.0014
SLICE_PEAK = TB/PULSE_DURATION * 2*np.pi/g * 1/THICKNESS # Change gradient strength based on pulse duration

# Increase the sampling rate because without it, the frequency modulation pulse is not accurate
DT = 0.5e-6
N = int(PULSE_DURATION/DT) # DT per point

# s2a) Make one photon version
B1Z_AMP = 1.84118378*FZ*2*np.pi/g
pulse = slr_pulse(N, TB, FA, name='s2', DT=DT)
bz_pulse = np.zeros(N)
gz_pulse = np.zeros(N)
sim_duration = PULSE_DURATION*2.5

t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK, PULSE_DURATION, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, np.zeros(N))
    Gz[i] = gz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, gz_pulse)

plot_waveform(fig, outer[cols+1:cols+SEQUENCE_PLOT_END], t, np.abs(RF), np.angle(RF), B1z, Gz, zoom_time=[0.9, 1.1], RF_lim=[0,15], B1z_lim=[-1.5,1.5], Gz_lim=[-40,30])
if PRINT_MAX_VALS:
    print('s2a B1xy^2 integral:' + str(np.trapz(np.power(abs(pulse),2), dx=DT)))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK, PULSE_DURATION),atol=1e-7, rtol=1e-11, hmax=DT, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), 1*np.angle(final_m))


# s2b) Make two-photon version
pulse = slr_pulse(N, TB, FA, freq=FZ, phase=g*B1Z_AMP/(2*np.pi*FZ)-np.pi/2, name='s1', DT=DT) / (j1(g/(2*np.pi*FZ) * B1Z_AMP))
bz_pulse = B1Z_AMP * np.sin(DT*np.arange(N)*2*np.pi*FZ)

t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK, PULSE_DURATION, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, gz_pulse)

plot_waveform(fig, outer[2*cols+1:2*cols+SEQUENCE_PLOT_END], t, np.abs(RF), np.angle(RF), B1z, Gz, zoom_time=[0.9, 1.1], RF_lim=[0,15], B1z_lim=[-1.5,1.5], Gz_lim=[-40,30])
if PRINT_MAX_VALS:
    print('s2b B1xy^2 integral:' + str(np.trapz(np.power(abs(pulse),2), dx=DT)))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK, PULSE_DURATION),atol=1e-7, rtol=1e-11, hmax=DT, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[2*cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), 1*np.angle(final_m))



# s2c) Make frequency modulated version
pulse = fm_pulse(N, TB, FA, FZ, B1Z_AMP, phase=g*B1Z_AMP/(2*np.pi*FZ)-np.pi/2, name='s2', DT=DT)
bz_pulse = np.zeros(N)

t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK, PULSE_DURATION, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, gz_pulse)

plot_waveform(fig, outer[3*cols+1:3*cols+SEQUENCE_PLOT_END], t, np.abs(RF), np.angle(RF), B1z, Gz, zoom_time=[0.9, 1.1], RF_lim=[0,15], B1z_lim=[-1.5,1.5], Gz_lim=[-40,30])
if PRINT_MAX_VALS:
    print('s2c B1xy^2 integral:' + str(np.trapz(np.power(abs(pulse),2), dx=DT)))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK, PULSE_DURATION),atol=1e-7, rtol=1e-11, hmax=DT, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[3*cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), np.angle(final_m))

plt.savefig("supplementary_figure2.pdf", bbox_inches=Bbox([[3.1, 0.5], [18.6, 7.5]]))