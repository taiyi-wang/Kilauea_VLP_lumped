#!/usr/bin/env python
# coding: utf-8

# Make plots for publicaton

# Note for the moment tensor to show, do:
# conda activate gp130 
# before running the script

import numpy as np
import matplotlib.pyplot as plt
#import emcee
import os

import calcol as calcol
import synthetic as syn
import ButterWorth as bw
import load_all_gfs

directory = os.getcwd()
#reader = emcee.backends.HDFBackend(directory+'/output.h5')


# 1. Example u(t), p(t), s(t); M(t), F(t)

# Best fit model from fitting both GPS and VLP waveform, with magma inertia in single force
rho_p = 2.39285412e+03; eff_rho_m = 2.45706703e+02; g = 9.8; R = 4.28985214e+02; beta = 1.66247858e-10 ; dtau = 1.95766369e+05; V_c = 4.05529763e+09 ; mu = 3e9; alpha = 8.87691256e-01;
L = 1940 - (alpha**2 * V_c * 3 / (4*np.pi))**(1/3);

# Paul's best fit model from fitting GPS, with vertical fault case
#rho_p = 2300; eff_rho_m = 0; g = 9.8; R = 1000; beta = 2e-10; V_c = 3.9e9; mu = 3e9; alpha = 1.23; L = 1940 - (alpha**2 * V_c * 3 / (4*np.pi))**(1/3); 
#dp = 3.3e6;
#dtau = dp*R/(2*L)/2

# These parameters recreate the 3.3 MPa pressure increase in the GRL paper, assuming the same chamber parameters. Note dp = 4*L*dtau_str/R
#rho_p = 2.50e+03; eff_rho_m = 1.72e+02; g = 9.8;  beta = 1.85e-10;  V_c = 3.94e+09; mu = 3e9; alpha = 1.23;
#L = 1940 - (alpha**2 * V_c * 3 / (4*np.pi))**(1/3); R = 4*L; dtau = 3.3e+06;

# MAP model for fitting only GPS
#rho_p = 2.52e+03; eff_rho_m = 7.45e+02; g = 9.8; R = 1.29e+03; beta = 1.2e-09 ; dtau = 8.49e+05; V_c = 4.52e+09; mu = 3e9; alpha = 0.991;
#L = 1940 - (alpha**2 * V_c * 3 / (4*np.pi))**(1/3); 

print('L is', L, 'm')

T = np.arange(-20, 60, 0.05);
T[np.argmin(abs(T))] = 0  

# calculate characteristic quantities:
pi = np.pi
A = pi * R**2              # cross-sectional area of piston
C = 2 * pi * R             # circumference of piston
V_p = A * L                # volume of piston
m_p = rho_p * V_p          # mass of piston
M = m_p + V_c*eff_rho_m    # effective mass for the whole system

pi_0 = 2*pi*R*L/(M*g)*dtau

l_star = beta*V_c*M*g/(pi**2 * R**4)
t_star = (beta*V_c*M/(pi**2 * R**4))**(1/2)
p_star = M*g/(pi*R**2)

print('pi_0', pi_0)

# for comparison to 2D dynamic rupture
#[time1, displacement1, pressure1, shear_stress1, DeltaT1, DeltaU1, DeltaP1, DeltaS1] = calcol2.calcol_alyt([rho_p, 0, L, g, w, beta, dtau, H, T], plot = 'NO') # without magma inertia
#[time2, displacement2, pressure2, shear_stress2, DeltaT2, DeltaU2, DeltaP2, DeltaS2] = calcol2.calcol_alyt([rho_p, eff_rho_m, L, g, w, beta, dtau, H, T], plot = 'NO') # with magma inertia'

[time1, displacement1, pressure1, shear_stress1, DeltaT1, DeltaU1, DeltaP1, DeltaS1] = calcol.calcol_alyt([rho_p, 0, L, g, R, beta, dtau, V_c, T], plot = 'NO') # without magma inertia
[time2, displacement2, pressure2, shear_stress2, DeltaT2, DeltaU2, DeltaP2, DeltaS2] = calcol.calcol_alyt([rho_p, eff_rho_m, L, g, R, beta, dtau, V_c, T], plot = 'NO') # with magma inertia'

#np.savetxt('alyt_results_to_cmpr_with_dynamic_rupture.txt', (time2, displacement2, pressure2, shear_stress2)) 

# compute energy balance 
g = 9.8; f_s = 0.8; 
sigma_n = (2100 - 1000) * g * L/2
tau_s = sigma_n*f_s
tau_d = tau_s - DeltaS2

p0 = m_p*g/A - 2*L*tau_s/R # p_0 needs to satisfy initial equilibrium
print('initial pressure is', p0/1e6,'MPa')
#dE_gravity = m_p*DeltaU2*g                               # gravitational potential
dE_chamber = 0.5*DeltaU2**2*pi**2*R**4/(beta*V_c)        # elastic energy of chamber + free energy of magma (sink)
dE_fault = -2*pi*R*L*DeltaS2/2*DeltaU2                  # elalstic strain energy released from fault (source)
#W_friction = 2*pi*R*L*tau_d*DeltaU2
#W_pressure = pi*R**2*p0*DeltaU2

#print('change in gravitational potential is', dE_gravity/1e12, 'x 10^12 J')
print('change in chamber elastic energy + free energy of magma is', dE_chamber/1e12, 'x 10^12 J')
print('change in elastic energy on fault is', dE_fault/1e12, 'x 10^12 J')
#print('Work done against friction is', W_friction/1e12, 'x 10^12 J')
#print('Work done against initial pressure is', W_pressure/1e12, ' x 10^12 J')

strength = np.zeros(len(time2))
zero_idx = np.where(time2==0)[0][0]
strength[zero_idx:] = -dtau

fig, ax1 = plt.subplots(figsize=(10, 7))
ax2 = ax1.twinx()
ax1.plot(time1, displacement1, 'royalblue', linestyle = 'dashed')
ax1.plot(time2, displacement2, 'royalblue', label = r'displacement, $u$')
ax2.plot(time1, pressure1/1e6, 'firebrick', linestyle = 'dashed')
ax2.plot(time2, pressure2/1e6, 'firebrick', label = r'pressure, $p$')
ax2.plot(time1, shear_stress1/1e6, 'goldenrod', linestyle = 'dashed')
ax2.plot(time2, shear_stress2/1e6, 'goldenrod', label = r'dynamic shear stress, $\tau$')
ax2.plot(time2, strength/1e6, 'k-', label=r'shear strength, $\tau_{str}$')
plt.xlim([-2, 6])
ax1.set_ylabel('(m)', fontsize=15)
ax2.set_ylabel('(MPa)', fontsize=15)
ax1.set_xlabel('time since beginning of collapse(s)',fontsize=15)
fig.legend()
#plt.savefig('caldera_dynamics.eps', format='eps')
plt.show()

'''
plt.figure()
plt.plot(displacement2, tau_qs)
plt.show()

plt.figure()
plt.plot(displacement2, tau)
plt.show()
'''

# Make plots to illustrate that radiation energy is not determined by gravitational potential
plt.figure()
plt.plot(time2, pressure2*A, 'firebrick', label = r'$F_{\Delta p}$')
plt.plot(time2, shear_stress2*C*L, 'goldenrod', label = r'$F_{\Delta \tau}$')
plt.plot(time2, pressure2*A + shear_stress2*C*L, 'royalblue', label = r'$m \ddot{u}$')
plt.xlim([-2,12])
plt.xlabel('time (s)'); plt.ylabel('force (N)')
plt.legend()
plt.show()

p0 = L*2000*g
tau0 = (m_p*g - p0*A)/(C*L)

plt.figure()
plt.plot(time2, (pressure2+p0)*A, 'firebrick', linestyle = 'dashed', label = r'$F_{p}$')
plt.plot(time2, (shear_stress2+tau0)*C*L, 'goldenrod', linestyle = 'dashed', label = r'$F_{\tau}$')
plt.plot(time2, np.repeat(m_p*g, len(time2)), 'k', linestyle = 'dashed', label = r'$F_{g}$')
plt.plot(time2, (pressure2)*A + (shear_stress2)*C*L, 'royalblue', linestyle = 'dashed', label = r'$m \ddot{u}$')
plt.xlim([-2,12])
plt.xlabel('time (s)'); plt.ylabel('force (N)')
plt.legend()
plt.show()


# Get station positions
sta_pos = np.zeros((19, 3))

# Load accelerometer locations (w/ origin NPIT)
accel_pos_xy = np.loadtxt(directory+'/inversion_input/acce_local_xy.txt', delimiter=',', skiprows=1) 
accel_labels = ['HMLE', 'NPT', 'PAUD', 'RSDD', 'UWE']

# Load broadband seismometer locations (w/ origin NPIT)
seism_pos_xy = np.loadtxt(directory+'/inversion_input/seism_local_xy.txt', delimiter=',', skiprows=1) 
seism_labels = ['HLPD', 'JOKA', 'MLOD', 'STCD']

GPS_pos_xy = np.loadtxt(directory+'/inversion_input/gps_local_xy.txt', delimiter=',', skiprows=1) 
GPS_labels = ['69FL', '92YN', 'AHUP', 'BDPK', 'BYRL', 'CNPK', 'CRIM', 'DEVL', 'OUTL', 'PUHI', 'PWRL', 'UWEV', 'V120', 'VSAS']
    
sta_pos[:3, :2] = accel_pos_xy[[0, 2, 3], :]
sta_pos[3:6, :2] = seism_pos_xy[[0, 1, 2],:]
sta_pos[6:17, :2] = GPS_pos_xy[:11, :]
sta_pos[[17,18], :2] = GPS_pos_xy[[12,13], :]

parameters = np.loadtxt(directory+'/inversion_input/parameters.txt', delimiter=',', usecols= 1)
chamber_cent = np.array(parameters[:3])
chamber_cent[2] = -chamber_cent[2] # z positive upwards
piston_cent = np.array(parameters[:3])
chamberParams = [V_c, chamber_cent, alpha] # use updated aspect ratio 
pistonParams = [L, R, piston_cent]         # use updated piston length or depth to top of chamber, and piston radius

# load all GFs
mt_gf_file = 'greens_functions/halfspace_Kilauea/half_1.94_mt/'
sf_gf_file = 'greens_functions/halfspace_Kilauea/half_1.94_sf/'
load_all_gfs.init(mt_gf_file, sf_gf_file, T)

# calculate single force on the crust, accounting for caldera block+magma inertia
dt = 0.05
acc2 = np.gradient(np.gradient(displacement2, dt, edge_order=2), dt)
force_history2 = M * acc2 # force history on the crust

ts_r, ts_z, Mt, Ft = syn.synthetic_general(pressure2, force_history2, time2, sta_pos[0:3], ['HMLE', 'PAUD', 'RSDD'], chamberParams, pistonParams, [mu, rho_p], deriv='VEL', coord='CYLINDRICAL')

moment_1 = Mt[0, 0]
moment_3 = Mt[2, 2]

fig, ax1 = plt.subplots(figsize=(10, 7))
ax2 = ax1.twinx()
ax1.plot(time2, moment_1, 'k--', label = r'$M_{xx} = M_{yy}$')
ax1.plot(time2, moment_3, 'k-.', label = r'$M_{zz}$')
ax2.plot(time2, Ft, 'k', label = r'$F_z$')
plt.xlim([-2, 6])
ax1.set_ylabel(r'(N $\cdot$ m)', fontsize=15)
ax2.set_ylabel(r'(N)', fontsize=15)
ax1.set_xlabel('time since beginning of collapse(s)',fontsize=15)
fig.legend()
#plt.savefig('point_representation.eps', format='eps')
plt.show()

#np.savetxt('MAP_force_history_no_magma_in_SF.txt', (time2, force_history2)) 

# 2. Plot moment tensor beachballs
# Get 6 components of the moment tensor 
'''
from obspy.imaging.beachball import beachball

mt = Mt[:,:,-1]
mxx = mt[0, 0]; mxy = mt[0, 1]; mxz = mt[0, 2]; myy = mt[1, 1]; myz = mt[1, 2]; mzz = mt[2, 2]
fig = beachball([mzz, mxx, myy, mxy, mxz, myz])
'''

# 3. Plot moment and force contributions
sta_labels = ['HMLE', 'PAUD', 'RSDD', 'HLPD', 'MLOD', 'STCD']
distance_labels = ['6.33 km', '7.61 km', '5.68 km', '12.9 km', '14.7 km', '16.2 km']

V_mom_r, V_mom_z, V_for_r, V_for_z = syn.synthetic_general(pressure2, force_history2, time2, sta_pos[0:6], ['HMLE', 'PAUD', 'RSDD', 'HLPD', 'MLOD', 'STCD'], chamberParams, pistonParams, [mu, rho_p], deriv='VEL', comp='YES', coord='CYLINDRICAL')

V_r = V_mom_r + V_for_r
V_z = V_mom_z + V_for_z

# Lowpass filter the velocity
order = 6
dt = time2[1] - time2[0]
fs = 1/dt     # sample rate, Hz
cutoff = 0.2  # desired cutoff frequency of the filter, Hz
	
V_mom_r_lp = bw.butter_lowpass_filter(V_mom_r, cutoff, fs, order)
V_mom_z_lp = bw.butter_lowpass_filter(V_mom_z, cutoff, fs, order)
V_for_r_lp = bw.butter_lowpass_filter(V_for_r, cutoff, fs, order)
V_for_z_lp = bw.butter_lowpass_filter(V_for_z, cutoff, fs, order)
V_r_lp = bw.butter_lowpass_filter(V_r, cutoff, fs, order)
V_z_lp = bw.butter_lowpass_filter(V_z, cutoff, fs, order)

fig, axes = plt.subplots(1, 2, figsize=(10,3), sharex=True, sharey=True)
axes_flat = axes.flatten()
for idx, i in enumerate([2,0,1,3,4,5]):

	ax_l = axes_flat[0]   # left
	ax_r = axes_flat[1] # right

	ax_l.plot(time2, V_mom_r_lp[i]+(7-idx)*5e-3, 'k--')
	ax_l.plot(time2, V_for_r_lp[i]+(7-idx)*5e-3, 'k:')
	ax_l.plot(time2, V_r_lp[i]+(7-idx)*5e-3, 'k-')
	ax_l.set_xlim([3, 35])
	#ax_l.set_title(sta_labels[i]+' '+'radial'+ ' '+'('+distance_labels[i]+')')

	ax_r.plot(time2, V_mom_z_lp[i]+(7-idx)*5e-3, 'k--',label='moment contribution')
	ax_r.plot(time2, V_for_z_lp[i]+(7-idx)*5e-3, 'k:',label='single force contribution')
	ax_r.plot(time2, V_z_lp[i]+(7-idx)*5e-3, 'k-',label='total')
	ax_r.set_xlim([3, 35])
	#ax_r.set_title(sta_labels[i]+' '+'vertical')

	#if idx == 0:
		#ax_r.legend(fontsize=14)

	#if idx == 2:
		#ax_l.set_ylabel('velocity (m/s)')

	#if i == 5:
		#ax_l.set_xlabel('time (s)')
		#ax_r.set_xlabel('time (s)')
fig.tight_layout()
#plt.savefig('mom_for_contribution.eps', format='eps')
plt.show()

# 4. Plot GPS radial and vertical displacements as a function of radial distance
ts_gps_x, ts_gps_y, ts_gps_z = syn.synthetic_general(pressure2, force_history2, time2, sta_pos[6:, :], ['69FL', '92YN', 'AHUP', 'BDPK', 'BYRL', 'CNPK', 'CRIM', 'DEVL', 'OUTL', 'PUHI', 'PWRL', 'V120', 'VSAS'], chamberParams, pistonParams, [mu, rho_p], deriv='DIS', coord='CARTESIAN')
x_offset = ts_gps_x[:,-1] - ts_gps_x[:,0]
y_offset = ts_gps_y[:,-1] - ts_gps_y[:,0]
z_offset = ts_gps_z[:,-1] - ts_gps_z[:,0]

r_offset = (x_offset**2 + y_offset**2)**(1/2)

x_dis = sta_pos[6:, 0]-chamber_cent[0] # translate to origin at chamber centroid 
y_dis = sta_pos[6:, 1]-chamber_cent[1]
r_dis = (x_dis**2 + y_dis**2)**(1/2)

plt.figure()
plt.plot(r_dis/1000, r_offset, '.', label='radial displacement')
plt.plot(r_dis/1000, z_offset, '.', label='vertical displacement')
plt.xlabel('radial distance from chamber ')
plt.legend()
plt.show()

#np.savetxt('disp_vs_radial_dis_ptsrc.txt', (r_dis, r_offset, z_offset)) 
#np.savetxt('MAP_disp_vs_radial_dis_ptsrc.txt', (r_dis, r_offset, z_offset)) 
#np.savetxt('MAP_onlyGPS_disp_vs_radial_dis_ptsrc.txt', (r_dis, r_offset, z_offset)) 


