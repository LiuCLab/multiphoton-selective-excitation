"""main

Main file to run to create figures 2-5 in the paper.
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
from pulse_generator_functions import *
from simulation_functions import *
from plotting_functions import *



#################################################################
# Make Figure 2: equivalent single-photon, two-photon, and freq mod slice select

# Make a grid of subplots and label the rows and columns
fig = plt.figure(figsize=(20, 10))
cols = 2+SEQUENCE_PLOT_END # Number of columns used for pulse sequence plot
outer = gridspec.GridSpec(4, cols, wspace=1, hspace=0.3)
ax = plt.Subplot(fig, outer[cols])
t = ax.text(0.7,0.2, 'Single-Photon', fontsize=CATEGORY_SIZE, rotation=90)
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
ax = plt.Subplot(fig, outer[SEQUENCE_PLOT_END+1])
t = ax.text(0.5,0, 'Experimental Slice Profile', fontsize=CATEGORY_SIZE)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)


# 2a) Make single photon version
pulse = slr_pulse(N, TB, FA, name='2a')
bz_pulse = np.zeros(N)
gz_pulse = np.zeros(N)
sim_duration = PULSE_DURATION*1.5

t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK, PULSE_DURATION, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, np.zeros(N))
    Gz[i] = gz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, gz_pulse)

plot_waveform(fig, outer[cols+1:cols+SEQUENCE_PLOT_END], t, np.abs(RF), -1*np.angle(RF), B1z, Gz)
if PRINT_MAX_VALS:
    print('2a max B1xy: ' + str(np.max(np.abs(pulse))))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK, PULSE_DURATION),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), -1*np.angle(final_m))
plot_experiment(fig, outer[cols+SEQUENCE_PLOT_END+1], '2a.npy')


# 2b) Make two-photon version
pulse = slr_pulse(N, TB, FA, freq=FZ, phase=g*B1Z_AMP/(2*np.pi*FZ)-np.pi/2, name='2b') / (j1(g/(2*np.pi*FZ) * B1Z_AMP))
bz_pulse = B1Z_AMP * np.sin(2e-6*np.arange(N)*2*np.pi*FZ)

t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK, PULSE_DURATION, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, gz_pulse)

plot_waveform(fig, outer[2*cols+1:2*cols+SEQUENCE_PLOT_END], t, np.abs(RF), -1*np.angle(RF), B1z, Gz)
if PRINT_MAX_VALS:
    print('2b max B1xy: ' + str(np.max(np.abs(pulse))))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK, PULSE_DURATION),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[2*cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), -1*np.angle(final_m))
plot_experiment(fig, outer[2*cols+SEQUENCE_PLOT_END+1], '2b.npy')



# 2c) Make frequency modulated version
pulse = fm_pulse(N, TB, FA, FZ, B1Z_AMP, phase=g*B1Z_AMP/(2*np.pi*FZ)-np.pi/2, name='2c')
bz_pulse = np.zeros(N)

t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK, PULSE_DURATION, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, gz_pulse)

plot_waveform(fig, outer[3*cols+1:3*cols+SEQUENCE_PLOT_END], t, np.abs(RF), -1*np.angle(RF), B1z, Gz)
if PRINT_MAX_VALS:
    print('2c max B1xy: ' + str(np.max(np.abs(pulse))))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK, PULSE_DURATION),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[3*cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), -1*np.angle(final_m))
plot_experiment(fig, outer[3*cols+SEQUENCE_PLOT_END+1], '2c.npy')

plt.savefig("figure2.pdf")



#################################################################
# Make Figure 3: slice shifting

# Make a grid of subplots and label the rows and columns
fig = plt.figure(figsize=(20, 10))
cols = 2+SEQUENCE_PLOT_END
outer = gridspec.GridSpec(4, cols, wspace=1, hspace=0.3)
ax = plt.Subplot(fig, outer[cols])
t = ax.text(0.7,0.3, r'$\omega_{xy}$ Shift', fontsize=CATEGORY_SIZE, rotation=90)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)
ax = plt.Subplot(fig, outer[2*cols])
t = ax.text(0.7,0.2, r'Constant $B_{1z}$', fontsize=CATEGORY_SIZE, rotation=90)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)
ax = plt.Subplot(fig, outer[3*cols])
t = ax.text(0.7,0.3, r'$\omega_{z}$ Shift', fontsize=CATEGORY_SIZE, rotation=90)
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


# 3a) Shifting using wxy offset
f_offset = 2*TB/PULSE_DURATION
if PRINT_MAX_VALS:
    print('3 frequency offset: ' + str(f_offset))
pulse = slr_pulse(N, TB, FA, freq=(FZ-f_offset), phase=g*B1Z_AMP/(2*np.pi*FZ)-np.pi/2, name='3a') / (j1(g/(2*np.pi*FZ) * B1Z_AMP)) # this is actually increasing the frequency
bz_pulse = B1Z_AMP * np.sin(2e-6*np.arange(N)*2*np.pi*FZ)

t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK, PULSE_DURATION, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, gz_pulse)

plot_waveform(fig, outer[cols+1:cols+SEQUENCE_PLOT_END], t, np.abs(RF), -1*np.angle(RF), B1z, Gz)
if PRINT_MAX_VALS:
    print('3a max B1xy: ' + str(np.max(np.abs(pulse))))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK, PULSE_DURATION),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), -1*np.angle(final_m))
plot_experiment(fig, outer[cols+SEQUENCE_PLOT_END+1], '3a.npy')


# 3b) Shifting using constant B1z
pulse = slr_pulse(N, TB, FA, freq=FZ, phase=g*B1Z_AMP/(2*np.pi*FZ)-np.pi/2, name='3b') / (j1(g/(2*np.pi*FZ) * B1Z_AMP))
bz_pulse = B1Z_AMP * np.sin(2e-6*np.arange(N)*2*np.pi*FZ) 

t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK, PULSE_DURATION, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, bz_pulse, dc_value=-2*np.pi*f_offset/g) # subtract to get the same direction as adding wxy
    Gz[i] = gz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, gz_pulse)

plot_waveform(fig, outer[2*cols+1:2*cols+SEQUENCE_PLOT_END], t, np.abs(RF), -1*np.angle(RF), B1z, Gz)
if PRINT_MAX_VALS:
    print('3b max B1xy: ' + str(np.max(np.abs(pulse))))
    print('3b rise-time: ' + str(SLICE_PEAK/SLEW_LIMIT))
if WRITE_WAVEFORM_FILES:
    make_b1z_csv(bz_pulse, SLICE_PEAK, PULSE_DURATION, '3b.csv', dc_value=-2*np.pi*f_offset/g)

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK, PULSE_DURATION, 0, -2*np.pi*f_offset/g),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[2*cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), -1*np.angle(final_m))
plot_experiment(fig, outer[2*cols+SEQUENCE_PLOT_END+1], '3b.npy')

# 3c) Shifitng using wz offset
pulse = slr_pulse(N, TB, FA, freq=FZ, phase=g*B1Z_AMP/(2*np.pi*FZ)-np.pi/2, name='3c') / (j1(g/(2*np.pi*(FZ+f_offset)) * B1Z_AMP)) # compensate for different wz freq
bz_pulse = B1Z_AMP * np.sin(2e-6*np.arange(N)*2*np.pi*(FZ+f_offset))

t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK, PULSE_DURATION, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, gz_pulse)

plot_waveform(fig, outer[3*cols+1:3*cols+SEQUENCE_PLOT_END], t, np.abs(RF), -1*np.angle(RF), B1z, Gz)
if PRINT_MAX_VALS:
    print('3c max B1xy: ' + str(np.max(np.abs(pulse))))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK, PULSE_DURATION),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[3*cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), -1*np.angle(final_m))
plot_experiment(fig, outer[3*cols+SEQUENCE_PLOT_END+1], '3c.npy')

plt.savefig("figure3.pdf")



#################################################################
# Make Figure 4: Modulation using B1z or B1xy

# Make a grid of subplots and label the rows and columns
fig = plt.figure(figsize=(20, 10))
cols = 2+SEQUENCE_PLOT_END
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
ax = plt.Subplot(fig, outer[SEQUENCE_PLOT_END+1])
t = ax.text(0.5,0, 'Experimental Slice Profile', fontsize=CATEGORY_SIZE)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)


# 4a) Two-photon slice selection using B1xy modulation
pulse = slr_pulse(N_FIG4, TB_FIG4, FA, freq=FZ, name='4a') / (j1(g/(2*np.pi*FZ) * B1Z_AMP))
bz_pulse = -1*B1Z_AMP * np.cos(2e-6*np.arange(N_FIG4)*2*np.pi*FZ)
sim_duration_fig4 = PULSE_DURATION_FIG4*1.5

if WRITE_WAVEFORM_FILES:
    write_rf_pulse_for_heartvista(pulse, '4a')

t = np.linspace(0,sim_duration_fig4,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, gz_pulse)

plot_waveform(fig, outer[cols+1:cols+SEQUENCE_PLOT_END], t, np.abs(RF), -1*np.angle(RF), B1z, Gz, zoom_time=[5.9, 6.1])
if PRINT_MAX_VALS:
    print('4a max B1xy: ' + str(np.max(np.abs(pulse))))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration_fig4, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK_FIG4, PULSE_DURATION_FIG4),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), -1*np.angle(final_m))
plot_experiment(fig, outer[cols+SEQUENCE_PLOT_END+1], '4a.npy')


# 4b) Two-photon slice selection using B1z modulation
max_bessel_arg = g*B1Z_AMP/(2*np.pi*FZ)
pulse = slr_pulse(N_FIG4, TB_FIG4, FA, freq=0)
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

bz_pulse = -1*pulse_bz1 * 2*np.pi*FZ/g * np.cos(2e-6*np.arange(N_FIG4)*2*np.pi*FZ) # lets make this a cos to get rid of phase differences
if WRITE_WAVEFORM_FILES:
    make_b1z_csv(bz_pulse, SLICE_PEAK, PULSE_DURATION_FIG4, '4b.csv')
pulse = np.zeros(len(pulse), dtype=complex)
for i in range(len(pulse)):
    pulse[i] = scale * np.complex(np.cos(2e-6*i*2*np.pi*FZ), np.sin(2e-6*i*2*np.pi*FZ))

if WRITE_WAVEFORM_FILES:
    write_rf_pulse_for_heartvista(pulse, '4b')

t = np.linspace(0,sim_duration_fig4,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, gz_pulse)

plot_waveform(fig, outer[2*cols+1:2*cols+SEQUENCE_PLOT_END], t, np.abs(RF), -1*np.angle(RF), B1z, Gz, zoom_time=[5.9, 6.1])
if PRINT_MAX_VALS:
    print('4b max B1xy: ' + str(np.max(np.abs(pulse))))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration_fig4, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK_FIG4, PULSE_DURATION_FIG4),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[2*cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), -1*np.angle(final_m))
plot_experiment(fig, outer[2*cols+SEQUENCE_PLOT_END+1], '4b.npy')

# 4c) Two-photon slice selection using both B1xy and B1z modulation
pulse = slr_pulse(N_FIG4, TB_FIG4, FA, freq=0) / j1(g/(2*np.pi*FZ) * B1Z_AMP)
for i in range(len(pulse)):
    if i<N_FIG4/2:
        pulse[i] = scale
    else:
        bz_pulse[i] = -1*B1Z_AMP * np.cos(2e-6*i*2*np.pi*FZ)
    pulse[i] = pulse[i] * np.complex(np.cos(2e-6*i*2*np.pi*FZ), np.sin(2e-6*i*2*np.pi*FZ))

if WRITE_WAVEFORM_FILES:
    write_rf_pulse_for_heartvista(pulse, '4c')
    make_b1z_csv(bz_pulse, SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, '4c.csv')

t = np.linspace(0,sim_duration_fig4,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK_FIG4, PULSE_DURATION_FIG4, gz_pulse)

plot_waveform(fig, outer[3*cols+1:3*cols+SEQUENCE_PLOT_END], t, np.abs(RF), -1*np.angle(RF), B1z, Gz, zoom_time=[5.9, 6.1])
if PRINT_MAX_VALS:
    print('4c max B1xy: ' + str(np.max(np.abs(pulse))))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration_fig4, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK_FIG4, PULSE_DURATION_FIG4),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[3*cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), -1*np.angle(final_m))
plot_experiment(fig, outer[3*cols+SEQUENCE_PLOT_END+1], '4c.npy')

plt.savefig("figure4.pdf")



#################################################################
# Make Figure 5: Multislice

# Make a grid of subplots and label the rows and columns
fig = plt.figure(figsize=(20, 10))
cols = 2+SEQUENCE_PLOT_END
outer = gridspec.GridSpec(4, cols, wspace=1, hspace=0.3)
ax = plt.Subplot(fig, outer[cols])
t = ax.text(0.7,0.0, 'Free Unequal Slices', fontsize=CATEGORY_SIZE, rotation=90)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)
ax = plt.Subplot(fig, outer[2*cols])
t = ax.text(0.7,0.0, r'Naive $\omega_{xy}$ Shifting', fontsize=CATEGORY_SIZE, rotation=90)
t.set_ha('center')
ax.axis("off")
fig.add_subplot(ax)
ax = plt.Subplot(fig, outer[3*cols])
t = ax.text(0.7,0.0, 'Shifted Multislice', fontsize=CATEGORY_SIZE, rotation=90)
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


# 5a) Multiphoton multislice RF pulse using oscillating gradients
acdc_ratio = 1.5
FZ = TB/PULSE_DURATION * 2
SLICE_PEAK = TB/PULSE_DURATION * 2*np.pi/g * 1/THICKNESS
pulse = slr_pulse(N, TB, FA, name="5a")
bz_pulse = np.zeros(N)
gz_pulse = SLICE_PEAK * acdc_ratio * np.sin(2e-6*np.arange(N)*2*np.pi*FZ) 
sim_duration = 1.5 * PULSE_DURATION

t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK, PULSE_DURATION, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, gz_pulse)
    
# Now to save the gz waveform
if WRITE_WAVEFORM_FILES:
    write_gz_pulse_for_heartvista(gz_pulse, SLICE_PEAK, PULSE_DURATION, '5a_gz')

plot_waveform(fig, outer[cols+1:cols+SEQUENCE_PLOT_END], t, np.abs(RF), -1*np.angle(RF), B1z, Gz, zoom_time=[2, 3.1])
if PRINT_MAX_VALS:
    print('5a max B1xy: ' + str(np.max(np.abs(pulse))))
    print('5a max gz: ' + str(np.max(np.abs(Gz))))
    print('5a time to RF start: ' + str(SLICE_PEAK/SLEW_LIMIT))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK, PULSE_DURATION, acdc_ratio*SLICE_PEAK/(2*np.pi*FZ)),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), -1*np.angle(final_m))
plot_experiment(fig, outer[cols+SEQUENCE_PLOT_END+1], '5a.npy', ylim=[0,10000]) # Max y-value is decreased because higher receiver attenuation was used to not saturate the receiver


# 5b) Shifting the slice via B1xy frequency alone does not work
shift_f = FZ
pulse = slr_pulse(N, TB, FA, freq=FZ, name="5b")


t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK, PULSE_DURATION, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, gz_pulse)

plot_waveform(fig, outer[2*cols+1:2*cols+SEQUENCE_PLOT_END], t, np.abs(RF), -1*np.angle(RF), B1z, Gz, zoom_time=[2, 3.1])
if PRINT_MAX_VALS:
    print('5b max B1xy: ' + str(np.max(np.abs(pulse))))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK, PULSE_DURATION, acdc_ratio*SLICE_PEAK/(2*np.pi*FZ)),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[2*cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), -1*np.angle(final_m))
plot_experiment(fig, outer[2*cols+SEQUENCE_PLOT_END+1], '5b.npy', ylim=[0,10000]) # Max y-value is decreased because higher receiver attenuation was used to not saturate the receiver


# 5c) Shifting the slice via B1xy frequency and a B1z works
B1Z_AMP = 2*np.pi*shift_f/g * acdc_ratio
pulse = fm_pulse(N, TB, FA*j1(g*B1Z_AMP/(2*np.pi*FZ)), FZ, B1Z_AMP, phase=g*B1Z_AMP/(2*np.pi*FZ)-np.pi/2, name='5c')

t = np.linspace(0,sim_duration,WAVEFORM_RES)
RF = np.zeros(len(t), dtype=complex)
B1z = np.zeros(len(t))
Gz = np.zeros(len(t))
for i in range(len(t)):
    RF[i] = bxy_waveform(t[i], SLICE_PEAK, PULSE_DURATION, pulse)
    B1z[i] = bz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, bz_pulse)
    Gz[i] = gz_waveform(t[i], SLICE_PEAK, PULSE_DURATION, gz_pulse)

plot_waveform(fig, outer[3*cols+1:3*cols+SEQUENCE_PLOT_END], t, np.abs(RF), -1*np.angle(RF), B1z, Gz, zoom_time=[2, 3.1])
if PRINT_MAX_VALS:
    print('5c max B1xy: ' + str(np.max(np.abs(pulse))))

M = np.array([0, 0, M0])
t = np.linspace(0, sim_duration, 101)
final_m = np.zeros(XRES, dtype=complex)
x_vals = np.linspace(-XLIM,XLIM,XRES)
for i in range(XRES):
    x = x_vals[i]
    y = 0
    sol = odeint(bloch, M, t, args=(x, y, pulse, bz_pulse, gz_pulse, SLICE_PEAK, PULSE_DURATION, acdc_ratio*SLICE_PEAK/(2*np.pi*FZ)),atol=1e-7, rtol=1e-11, hmax=2e-6, mxstep=5000)
    final_m[i] = np.complex(sol[-1,0], sol[-1,1])

plot_sim(fig, outer[3*cols+SEQUENCE_PLOT_END], x_vals, np.abs(final_m), -1*np.angle(final_m))
plot_experiment(fig, outer[3*cols+SEQUENCE_PLOT_END+1], '5c.npy', ylim=[0,10000]) # Max y-value is decreased because higher receiver attenuation was used to not saturate the receiver

plt.savefig("figure5.pdf")


plt.show()


