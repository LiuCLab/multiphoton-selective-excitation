"""Supplementary Figure 5

Supplemental file to run to create supplementary figure 5.
All values are in SI units (m, s, T, etc) unless otherwise noted.

Dependencies: 
    sigpy (https://sigpy.readthedocs.io/en/latest/mri_rf.html)
    numpy
    scipy
    matplotlib

Author: Victor Han
Last Modified: 11/10/22

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
# Make Figure s5: Equal Amplitude Multiphoton Multislice with 90 Degree Flip Angle
FA = 90

# Make a grid of subplots and label the rows and columns
fig = plt.figure(figsize=(20, 20))
cols = 2+SEQUENCE_PLOT_END
outer = gridspec.GridSpec(6, cols, wspace=1, hspace=0.3)
ax = plt.Subplot(fig, outer[cols])
t = ax.text(0.7,0.2, 'Center Multiphoton', fontsize=CATEGORY_SIZE, rotation=90)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)
ax = plt.Subplot(fig, outer[2*cols])
t = ax.text(0.7,0.0, 'Left-Shifted Multiphoton', fontsize=CATEGORY_SIZE, rotation=90)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)
ax = plt.Subplot(fig, outer[3*cols])
t = ax.text(0.7,0.0, 'Right-Shifted Multiphoton', fontsize=CATEGORY_SIZE, rotation=90)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)
ax = plt.Subplot(fig, outer[4*cols])
t = ax.text(0.7,-0.3, 'Summed Multiphoton Mulislice', fontsize=CATEGORY_SIZE, rotation=90)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)
ax = plt.Subplot(fig, outer[5*cols])
t = ax.text(0.7,0.0, 'Naive Multislice', fontsize=CATEGORY_SIZE, rotation=90)
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
ax = plt.Subplot(fig, outer[SEQUENCE_PLOT_END+1])
t = ax.text(0.5,0, 'Experimental Slice Profile', fontsize=CATEGORY_SIZE)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)


# s5a) Multiphoton multislice RF pulse using oscillating gradients
acdc_ratio = 1.5
center_pulse_scale = (j0(0) + jv(2,2*acdc_ratio) - 2*j1(acdc_ratio)) / (1 - j1(acdc_ratio))
normalization_factor = center_pulse_scale + 2*j1(acdc_ratio)
FZ = TB_FIG6/PULSE_DURATION_FIG6 * 6
SLICE_PEAK = TB/PULSE_DURATION * 2*np.pi/g * 1/THICKNESS
pulse = center_pulse_scale * slr_pulse(N_FIG6, TB, FA, ptype='ex', name="s5a") / normalization_factor
bz_pulse = np.zeros(N_FIG6)
gz_pulse = SLICE_PEAK_FIG6 * acdc_ratio * np.sin(2e-6*np.arange(N_FIG6)*2*np.pi*FZ) 
sim_duration = 1.5 * PULSE_DURATION_FIG6

t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, gz_pulse)
    
# Now to save the gz waveform
if WRITE_WAVEFORM_FILES:
    write_gz_pulse_for_heartvista(gz_pulse, SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, 's5a_gz', area_offset=acdc_ratio*SLICE_PEAK_FIG6/(2*np.pi*FZ))

plot_waveform(fig, outer[cols+1:cols+SEQUENCE_PLOT_END], t, np.abs(RF), np.angle(RF), B1z, Gz, zoom_time=[5.5, 6.5], RF_lim=[0,11])
if PRINT_MAX_VALS:
    print('s5a max B1xy: ' + str(np.max(np.abs(pulse))))
    print('s5a max gz: ' + str(np.max(np.abs(Gz))))
    print('s5a time to RF start: ' + str(SLICE_PEAK_FIG6/SLEW_LIMIT))
    print('s5a B1xy^2 integral:' + str(np.trapz(np.power(abs(pulse),2), dx=DT)))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM_FIG6,XLIM_FIG6,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, acdc_ratio*SLICE_PEAK_FIG6/(2*np.pi*FZ)),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), np.angle(final_m), M_mag_lim=[0,1.1])
plot_experiment(fig, outer[cols+SEQUENCE_PLOT_END+1], 's5a.npy', xlim=XLIM_FIG6, ylim=[0,45000]) 


# s5b) Same as s5a, but shifted to the left
shift_f = FZ
B1Z_AMP = 2*np.pi*shift_f/g * acdc_ratio
pulse = fm_pulse(N_FIG6, TB, FA*j1(g*B1Z_AMP/(2*np.pi*FZ)), FZ, B1Z_AMP, phase=g*B1Z_AMP/(2*np.pi*FZ)-np.pi/2, ptype='ex', name='s5b') / normalization_factor
bz_pulse = np.zeros(N_FIG6)
gz_pulse = SLICE_PEAK_FIG6 * acdc_ratio * np.sin(2e-6*np.arange(N_FIG6)*2*np.pi*FZ) 
sim_duration = 1.5 * PULSE_DURATION_FIG6

t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, gz_pulse)

plot_waveform(fig, outer[2*cols+1:2*cols+SEQUENCE_PLOT_END], t, np.abs(RF), np.angle(RF), B1z, Gz, zoom_time=[5.5, 6.5], RF_lim=[0,11])
if PRINT_MAX_VALS:
    print('s5b max B1xy: ' + str(np.max(np.abs(pulse))))
    print('s5b B1xy^2 integral:' + str(np.trapz(np.power(abs(pulse),2), dx=DT)))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM_FIG6,XLIM_FIG6,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, acdc_ratio*SLICE_PEAK_FIG6/(2*np.pi*FZ)),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[2*cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), np.angle(final_m), M_mag_lim=[0,1.1])
plot_experiment(fig, outer[2*cols+SEQUENCE_PLOT_END+1], 's5b.npy', xlim=XLIM_FIG6, ylim=[0,45000]) 

# s5c) Same as s5a, but shifted to the right
pulse = fm_pulse(N_FIG6, TB, FA*j1(g*B1Z_AMP/(2*np.pi*FZ)), -FZ, B1Z_AMP, phase=g*B1Z_AMP/(2*np.pi*-FZ)-np.pi/2, ptype='ex', name='s5c') / normalization_factor
bz_pulse = np.zeros(N_FIG6)
gz_pulse = SLICE_PEAK_FIG6 * acdc_ratio * np.sin(2e-6*np.arange(N_FIG6)*2*np.pi*FZ) 
sim_duration = 1.5 * PULSE_DURATION_FIG6

t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, gz_pulse)

plot_waveform(fig, outer[3*cols+1:3*cols+SEQUENCE_PLOT_END], t, np.abs(RF), np.angle(RF), B1z, Gz, zoom_time=[5.5, 6.5], RF_lim=[0,11])
if PRINT_MAX_VALS:
    print('s5c max B1xy: ' + str(np.max(np.abs(pulse))))
    print('s5c B1xy^2 integral:' + str(np.trapz(np.power(abs(pulse),2), dx=DT)))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM_FIG6,XLIM_FIG6,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, acdc_ratio*SLICE_PEAK_FIG6/(2*np.pi*FZ)),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[3*cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), np.angle(final_m), M_mag_lim=[0,1.1])
plot_experiment(fig, outer[3*cols+SEQUENCE_PLOT_END+1], 's5c.npy', xlim=XLIM_FIG6) 


# s5d) Adding the pulses from s5a-c creates a multislice excitation with equal flip angles on three slices
#Center pulse 
pulse1 = slr_pulse(N_FIG6, TB, FA, ptype='ex', name="s5d_center")
#Left pulse
pulse2 = fm_pulse(N_FIG6, TB, FA*j1(g*B1Z_AMP/(2*np.pi*FZ)), FZ, B1Z_AMP, phase=g*B1Z_AMP/(2*np.pi*FZ)-np.pi/2, ptype='ex', name='s5d_left')
#Right pulse
pulse3 = fm_pulse(N_FIG6, TB, FA*j1(g*B1Z_AMP/(2*np.pi*FZ)), -FZ, B1Z_AMP, phase=g*B1Z_AMP/(2*np.pi*-FZ)-np.pi/2, ptype='ex', name='s5d_right')

# Sum of the three pulses
pulse = (center_pulse_scale*pulse1 + pulse2 + pulse3) / normalization_factor
if WRITE_WAVEFORM_FILES:
    write_rf_pulse_for_heartvista(pulse, 's5d_combined')
bz_pulse = np.zeros(N_FIG6)
gz_pulse = SLICE_PEAK_FIG6 * acdc_ratio * np.sin(2e-6*np.arange(N_FIG6)*2*np.pi*FZ) 
sim_duration = 1.5 * PULSE_DURATION_FIG6

t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, gz_pulse)

plot_waveform(fig, outer[4*cols+1:4*cols+SEQUENCE_PLOT_END], t, np.abs(RF), np.angle(RF), B1z, Gz, zoom_time=[5.5, 6.5], RF_lim=[0,11])
if PRINT_MAX_VALS:
    print('s5d max B1xy: ' + str(np.max(np.abs(pulse))))
    print('s5d B1xy^2 integral:' + str(np.trapz(np.power(abs(pulse),2), dx=DT)))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM_FIG6,XLIM_FIG6,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, acdc_ratio*SLICE_PEAK_FIG6/(2*np.pi*FZ)),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[4*cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), np.angle(final_m), M_mag_lim=[0,1.1])
plot_experiment(fig, outer[4*cols+SEQUENCE_PLOT_END+1], 's5d.npy', xlim=XLIM_FIG6) 

# 6e) Summed Standard SLR Pulses
pulse = slr_pulse(N_FIG6, TB, FA, freq=FZ, ptype='ex', name="s5e_left")*np.complex(0,1) + slr_pulse(N_FIG6, TB, FA, freq=0, ptype='ex', name="s5e_center") + slr_pulse(N_FIG6, TB, FA, freq=-FZ, ptype='ex', name="s5e_right")*np.complex(0,-1)
if WRITE_WAVEFORM_FILES:
    write_rf_pulse_for_heartvista(pulse, 's5e_combined')
bz_pulse = np.zeros(N_FIG6)
gz_pulse = np.zeros(N_FIG6)
sim_duration = 1.5 * PULSE_DURATION_FIG6

# Now to save the gz waveform
if WRITE_WAVEFORM_FILES:
    write_gz_pulse_for_heartvista(gz_pulse, SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, 's5e_gz', area_offset=0)

t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, gz_pulse)

plot_waveform(fig, outer[5*cols+1:5*cols+SEQUENCE_PLOT_END], t, np.abs(RF), np.angle(RF), B1z, Gz, zoom_time=[5.5, 6.5], RF_lim=[0,11])
if PRINT_MAX_VALS:
    print('s5e max B1xy: ' + str(np.max(np.abs(pulse))))
    print('s5e B1xy^2 integral:' + str(np.trapz(np.power(abs(pulse),2), dx=DT)))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM_FIG6,XLIM_FIG6,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK_FIG6, PULSE_DURATION_FIG6, 0),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[5*cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), np.angle(final_m), M_mag_lim=[0,1.1])
plot_experiment(fig, outer[5*cols+SEQUENCE_PLOT_END+1], 's5e.npy', xlim=XLIM_FIG6) 

plt.savefig("supplementary_figure5.pdf", bbox_inches=Bbox([[3.1, 1.5], [18.6, 16]]))

