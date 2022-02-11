#!/usr/bin/env python
# coding: utf-8
# Forward prediction function for waveforms

import sys
import numpy as np
import matplotlib.pyplot as plt
import gc

#import synthetic as syn
import synthetic as syn
import helpers as hp
import ButterWorth as bw
import mogi as mogi
import cmp_Er as cmp_Er

import calcol as calcol


def pred(md_vec, params):
	'''
	The function first calls the matlab function ''calcol_alyt'' to compute stress and pressure history,
	then input them into thehalf space elastodynamic Green's functions to compute synthetic seismograms:

	md_vec = variables that we want to modify
	params = constants or location parameters that are constant

	'''
	# Caldera collapse physics ------------------------------------------------------------------------------------------------------------------
	# load parameters
	
	g = params[0];  dz = np.abs(params[2][2])
	deriv = params[6]; comp = params[7]; coord = params[8]; T = params[10];
	mu = params[11]; rho_c = params[12];

	# load variables
	beta = md_vec[0]; V_c = md_vec[1]; dtau = md_vec[2]; eff_rho_m = md_vec[3]; alpha = md_vec[4];
	rho_p = md_vec[5]; R = md_vec[6];  
	time_shift_HMLE_r = md_vec[7]; time_shift_PAUD_r = md_vec[8]; time_shift_RSDD_r = md_vec[9]; time_shift_HLPD_r = md_vec[10]; time_shift_MLOD_r = md_vec[11]; time_shift_STCD_r = md_vec[12]; 
	time_shift_HMLE_z = md_vec[13]; time_shift_PAUD_z = md_vec[14]; time_shift_RSDD_z = md_vec[15]; time_shift_HLPD_z = md_vec[16]; time_shift_MLOD_z = md_vec[17]; time_shift_STCD_z = md_vec[18]; 

	# form input for calcol_alyt
	ra = (alpha**2 * V_c * 3 / (4*np.pi))**(1/3)
	L = dz - ra 
	consts = [rho_p, eff_rho_m, L, g, R, beta, dtau, V_c, T];
	[time, displacement, pressure, shear_stress, DeltaT, DeltaU, DeltaP, DeltaS] = calcol.calcol_alyt(consts, plot = 'NO')

	#shear_stress = -shear_stress # positive upwards

	# to avoid any misfit due to uncertainties of source time
	dt = time[2] - time[1]

	time_HMLE_r = time + time_shift_HMLE_r; time_PAUD_r = time + time_shift_PAUD_r; time_RSDD_r = time + time_shift_RSDD_r; 
	time_HLPD_r = time + time_shift_HLPD_r; time_MLOD_r = time + time_shift_MLOD_r; time_STCD_r = time + time_shift_STCD_r; 

	time_HMLE_z = time + time_shift_HMLE_z; time_PAUD_z = time + time_shift_PAUD_z; time_RSDD_z = time + time_shift_RSDD_z; 
	time_HLPD_z = time + time_shift_HLPD_z; time_MLOD_z = time + time_shift_MLOD_z; time_STCD_z = time + time_shift_STCD_z; 

	# calculate single force on the crust, accounting for caldera block+magma inertia
	M = np.pi*R**2*L*rho_p + eff_rho_m*V_c
	acc = np.gradient(np.gradient(displacement, dt), dt)
	force_history = M * acc # force history on the crust

	# Compute ground motion --------------------------------------------------------------------------------------------------------------
	sta_pos = params[1]
	chamber_cent = params[2]
	piston_cent = params[3]

	chamberParams = [V_c, chamber_cent, alpha] # use updated aspect ratio 
	pistonParams = [L, R, piston_cent]         # use updated piston length or depth to top of chamber, and piston radius

	ts_r, ts_z, Mt, Ft = syn.synthetic_general(pressure, force_history, time, sta_pos[0:6], ['HMLE', 'PAUD', 'RSDD', 'HLPD', 'MLOD', 'STCD'], chamberParams, pistonParams, [mu, rho_p], deriv=deriv, coord=coord)

	# Compute radiated energy from kinematic model
	v_p = np.sqrt(3*mu/rho_c)
	v_s = np.sqrt(mu/rho_c) # assume nu = 0.25
	Er_m, Er_f, Er_k = cmp_Er.cmp_kinematic_Er(time, Mt, Ft, v_p, v_s, rho_c) 

	HMLE_r = ts_r[0]; PAUD_r = ts_r[1]; RSDD_r = ts_r[2]; HLPD_r = ts_r[3]; MLOD_r = ts_r[4]; STCD_r = ts_r[5];
	HMLE_z = ts_z[0]; PAUD_z = ts_z[1]; RSDD_z = ts_z[2]; HLPD_z = ts_z[3]; MLOD_z = ts_z[4]; STCD_z = ts_z[5];

	## Lowpass filter the velocity
	order = 6
	fs = 1/dt     # sample rate, Hz
	cutoff = 0.2  # desired cutoff frequency of the filter, Hz
	
	HMLE_r_lp = bw.butter_lowpass_filter(HMLE_r, cutoff, fs, order); PAUD_r_lp = bw.butter_lowpass_filter(PAUD_r, cutoff, fs, order); RSDD_r_lp = bw.butter_lowpass_filter(RSDD_r, cutoff, fs, order)
	HLPD_r_lp = bw.butter_lowpass_filter(HLPD_r, cutoff, fs, order); MLOD_r_lp = bw.butter_lowpass_filter(MLOD_r, cutoff, fs, order); STCD_r_lp = bw.butter_lowpass_filter(STCD_r, cutoff, fs, order)

	HMLE_z_lp = bw.butter_lowpass_filter(HMLE_z, cutoff, fs, order); PAUD_z_lp = bw.butter_lowpass_filter(PAUD_z, cutoff, fs, order); RSDD_z_lp = bw.butter_lowpass_filter(RSDD_z, cutoff, fs, order)
	HLPD_z_lp = bw.butter_lowpass_filter(HLPD_z, cutoff, fs, order); MLOD_z_lp = bw.butter_lowpass_filter(MLOD_z, cutoff, fs, order); STCD_z_lp = bw.butter_lowpass_filter(STCD_z, cutoff, fs, order)

	if params[9] == 'YES':
	# GPS offset prediction
		ts_gps_x, ts_gps_y, ts_gps_z = syn.synthetic_general(pressure, force_history, time, sta_pos[6:, :], ['69FL', '92YN', 'AHUP', 'BDPK', 'CNPK', 'DEVL', 'OUTL', 'PUHI', 'PWRL', 'V120', 'VSAS'], chamberParams, pistonParams, [mu, rho_p], deriv='DIS', coord='CARTESIAN')
		x_offset = ts_gps_x[:,-1] - ts_gps_x[:,0]
		y_offset = ts_gps_y[:,-1] - ts_gps_y[:,0]
		z_offset = ts_gps_z[:,-1] - ts_gps_z[:,0]

		gps_offset = np.concatenate((x_offset, y_offset, z_offset))
		return ((time_HMLE_r, time_HMLE_z), (time_PAUD_r, time_PAUD_z), (time_RSDD_r, time_RSDD_z), (time_HLPD_r, time_HLPD_z), (time_MLOD_r, time_MLOD_z), (time_STCD_r, time_STCD_z)), (HMLE_r_lp, HMLE_z_lp), (PAUD_r_lp, PAUD_z_lp), (RSDD_r_lp, RSDD_z_lp), (HLPD_r_lp, HLPD_z_lp), (MLOD_r_lp, MLOD_z_lp), (STCD_r_lp, STCD_z_lp), (DeltaT, DeltaU, DeltaP, DeltaS, Er_m, Er_f, Er_k), gps_offset


	elif params[9] == 'NO':
	# Just waveforms
		return ((time_HMLE_r, time_HMLE_z), (time_PAUD_r, time_PAUD_z), (time_RSDD_r, time_RSDD_z), (time_HLPD_r, time_HLPD_z), (time_MLOD_r, time_MLOD_z), (time_STCD_r, time_STCD_z)), (HMLE_r_lp, HMLE_z_lp), (PAUD_r_lp, PAUD_z_lp), (RSDD_r_lp, RSDD_z_lp), (HLPD_r_lp, HLPD_z_lp), (MLOD_r_lp, MLOD_z_lp), (STCD_r_lp, STCD_z_lp), (DeltaT, DeltaU, DeltaP, DeltaS, Er_m, Er_f, Er_k)
	
