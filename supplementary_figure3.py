"""Supplementary Figure 3

Supplemental file to run to create supplementary figure 3.
All values are in SI units (m, s, T, etc) unless otherwise noted.
This code takes a long time to run.

Dependencies: 
    sigpy (https://sigpy.readthedocs.io/en/latest/mri_rf.html)
    numpy
    scipy
    matplotlib

Author: Victor Han
Last Modified: 11/05/22

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
# Make Figure s3: Modulation using B1z or B1xy with larger wz and shorter pulse duration

# Make a grid of subplots and label the rows and columns
fig = plt.figure(figsize=(20, 10))
cols = 1+SEQUENCE_PLOT_END
outer = gridspec.GridSpec(4, cols, wspace=1, hspace=0.3)
ax = plt.Subplot(fig, outer[cols])
t = ax.text(0.7,0.1, r'$B_{1xy}$ Mod Two-Photon', fontsize=CATEGORY_SIZE-2, rotation=90)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)
ax = plt.Subplot(fig, outer[2*cols])
t = ax.text(0.7,0.1, r'$B_{1z}$ Mod Two-Photon', fontsize=CATEGORY_SIZE-2, rotation=90)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)
ax = plt.Subplot(fig, outer[3*cols])
t = ax.text(0.7,0.1, 'Both Mod Two-Photon', fontsize=CATEGORY_SIZE-2, rotation=90)
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

# Increase the sampling rate because without it, the pulses are not accurate with the high wz
TB_FIG4 = 6
PULSE_DURATION_FIG4 = 0.006
DT = 0.2e-6
N_FIG4 = int(PULSE_DURATION_FIG4/DT) # 0.2us per point
SLICE_PEAK_FIG4 = TB_FIG4/PULSE_DURATION_FIG4 * 2*np.pi/g * 1/THICKNESS
FZ = 150e3
B1Z_AMP = 1.84118378*FZ*2*np.pi/g

# s3a) Two-photon slice selection using B1xy modulation
pulse = slr_pulse(N_FIG4, TB_FIG4, FA, freq=FZ, name='s3', DT=DT) / (j1(g/(2*np.pi*FZ) * B1Z_AMP))
bz_pulse = -1*B1Z_AMP * np.cos(DT*np.arange(N_FIG4)*2*np.pi*FZ)
gz_pulse = np.zeros(N_FIG4)
sim_duration_fig4 = PULSE_DURATION_FIG4*1.5


t = np.linspace(0,sim_duration_fig4,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, gz_pulse)

plot_waveform(fig, outer[cols+1:cols+SEQUENCE_PLOT_END], t, np.abs(RF), np.angle(RF), B1z, Gz, zoom_time=[3, 3.1], RF_lim=[0,5], B1z_lim=[-12,12])
if PRINT_MAX_VALS:
    print('s3a B1xy^2 integral:' + str(np.trapz(np.power(abs(pulse),2), dx=DT)))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration_fig4, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK_FIG4, PULSE_DURATION_FIG4),atol=1e-8, rtol=1e-12, hmax=DT, mxstep=50000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), np.angle(final_m))


# s3b) Two-photon slice selection using B1z modulation
max_bessel_arg = g*B1Z_AMP/(2*np.pi*FZ)
pulse = slr_pulse(N_FIG4, TB_FIG4, FA, freq=0, name='s2', DT=DT)
scale = np.max(np.abs(pulse)) / j1(max_bessel_arg) 
pulse = pulse / np.max(np.abs(pulse)) * j1(max_bessel_arg)
pulse_bz1 = np.zeros(len(pulse))
from scipy.optimize import minimize
def diff(x,a):
    yt = j1(x)
    return (yt - a )**2
for i in range(len(pulse)):
    res = minimize(diff, 0, args=(np.real(pulse[i])), bounds=[(-max_bessel_arg, max_bessel_arg)])
    pulse_bz1[i] = res.x[0]

bz_pulse = -1*pulse_bz1 * 2*np.pi*FZ/g * np.cos(DT*np.arange(N_FIG4)*2*np.pi*FZ) # lets make this a cos to get rid of phase differences
pulse = np.zeros(len(pulse), dtype=complex)
for i in range(len(pulse)):
    pulse[i] = scale * np.complex(np.cos(DT*i*2*np.pi*FZ), np.sin(DT*i*2*np.pi*FZ))

t = np.linspace(0,sim_duration_fig4,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, gz_pulse)

plot_waveform(fig, outer[2*cols+1:2*cols+SEQUENCE_PLOT_END], t, np.abs(RF), np.angle(RF), B1z, Gz, zoom_time=[3, 3.1], RF_lim=[0,5], B1z_lim=[-12,12])
if PRINT_MAX_VALS:
    print('s3b B1xy^2 integral:' + str(np.trapz(np.power(abs(pulse),2), dx=DT)))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration_fig4, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK_FIG4, PULSE_DURATION_FIG4),atol=1e-8, rtol=1e-12, hmax=DT, mxstep=50000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[2*cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), np.angle(final_m))

# s3c) Two-photon slice selection using both B1xy and B1z modulation
pulse = slr_pulse(N_FIG4, TB_FIG4, FA, freq=0, DT=DT) / j1(g/(2*np.pi*FZ) * B1Z_AMP)
for i in range(len(pulse)):
    if i<N_FIG4/2:
        pulse[i] = scale
    else:
        bz_pulse[i] = -1*B1Z_AMP * np.cos(DT*i*2*np.pi*FZ)
    pulse[i] = pulse[i] * np.complex(np.cos(DT*i*2*np.pi*FZ), np.sin(DT*i*2*np.pi*FZ))


t = np.linspace(0,sim_duration_fig4,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, gz_pulse)

plot_waveform(fig, outer[3*cols+1:3*cols+SEQUENCE_PLOT_END], t, np.abs(RF), np.angle(RF), B1z, Gz, zoom_time=[3, 3.1], RF_lim=[0,5], B1z_lim=[-12,12])
if PRINT_MAX_VALS:
    print('s3c B1xy^2 integral:' + str(np.trapz(np.power(abs(pulse),2), dx=DT)))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration_fig4, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK_FIG4, PULSE_DURATION_FIG4),atol=1e-8, rtol=1e-12, hmax=DT, mxstep=50000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[3*cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), np.angle(final_m))

plt.savefig("supplementary_figure3.pdf", bbox_inches=Bbox([[3.1, 0.5], [18.6, 7.5]]))

