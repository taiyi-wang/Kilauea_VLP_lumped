#!/usr/bin/env python
# coding: utf-8
# Set up globally available data
import os
import numpy as np


def init(lb_time, ub_time, vel_disp_flag, GPS_flag):
	# use global variables to reduce time spent passing variables among parallel processors, which makes parallel processing slower than serial
	global data, data_invCov, data_lndetCov
	# get path of current directory
	directory = os.getcwd()
	# load data -------------------------------------------------------------------------------------------------------------------------------------
	
	if vel_disp_flag == 'VEL':
		# Velocity waveforms
		data_HMLE = np.loadtxt(directory+'/inversion_input/HMLE_avg_vel_last32_stackfirst.txt', delimiter=',', skiprows=1)
		data_PAUD = np.loadtxt(directory+'/inversion_input/PAUD_avg_vel_last32_stackfirst.txt', delimiter=',', skiprows=1)
		data_RSDD = np.loadtxt(directory+'/inversion_input/RSDD_avg_vel_last32_stackfirst.txt', delimiter=',', skiprows=1)
		data_HLPD = np.loadtxt(directory+'/inversion_input/HLPD_avg_vel_last32_stackfirst.txt', delimiter=',', skiprows=1)
		data_MLOD = np.loadtxt(directory+'/inversion_input/MLOD_avg_vel_last32_stackfirst.txt', delimiter=',', skiprows=1)
		data_STCD = np.loadtxt(directory+'/inversion_input/STCD_avg_vel_last32_stackfirst.txt', delimiter=',', skiprows=1)

		# load noise standard deviation in waveforms, assuming white noise
		whNoiseStd = np.loadtxt(directory+'/inversion_input/whNoiseStd_waveform.txt', delimiter=',', skiprows=1)
		std_r_list = whNoiseStd[:, 0]
		std_z_list = whNoiseStd[:, 1]

	elif vel_disp_flag == 'DIS':
		# Displacement waveforms
		data_HMLE = np.loadtxt(directory+'/inversion_input/HMLE_avg_disp_last32_stackfirst.txt', delimiter=',', skiprows=1)
		data_PAUD = np.loadtxt(directory+'/inversion_input/PAUD_avg_disp_last32_stackfirst.txt', delimiter=',', skiprows=1)
		data_RSDD = np.loadtxt(directory+'/inversion_input/RSDD_avg_disp_last32_stackfirst.txt', delimiter=',', skiprows=1)
		data_MLOD = np.loadtxt(directory+'/inversion_input/MLOD_avg_disp_last32_stackfirst.txt', delimiter=',', skiprows=1)
	
	all_station_data = [data_HMLE, data_PAUD, data_RSDD, data_HLPD, data_MLOD, data_STCD]

	if GPS_flag == 'YES':
		# GPS offsets
		GPS = np.loadtxt(directory+'/inversion_input/GPS_avg_offset_and_std_last32.txt', delimiter=',', skiprows=1)
		GPS = np.delete(GPS, [4, 6, 11], 0) # Delete BYRL, CRIM, and UWEV, in that order, for being too close to ring fault
		# reorder into a vector [E, N, U]
		data_gps = np.concatenate((GPS[:, 0], GPS[:, 1], GPS[:, 2])) 
		data_std_gps = np.concatenate((GPS[:, 3], GPS[:, 4], GPS[:,5])) 

		cov_gps = np.diag(data_std_gps * data_std_gps)
		invCov_gps = np.diag(1/(data_std_gps * data_std_gps))

		L = np.linalg.cholesky(cov_gps)
		R = np.transpose(L)
		lndetCov_gps = 2*np.sum(np.log(np.diag(R)))

	data = []
	data_invCov = []
	data_lndetCov = []

	Nsites = len(all_station_data)
	for i in np.arange(Nsites):
		# collect waveform data
		data_time = all_station_data[i][:,0]
		data_z = all_station_data[i][:,3]                                             	# vertical component
		data_r = all_station_data[i][:,1]                                             	# radial component       
		data_idx = np.where(np.logical_and(data_time >= lb_time, data_time <= ub_time)) # Indices of time values within the time window of interest

		data_time = data_time[data_idx]
		data_z = data_z[data_idx]
		data_r = data_r[data_idx]

		data.append([data_time, data_r, data_z])

		# waveform data uncertainty
		# Note that, in objective.py, data and predictions are downsampled to 300 samples per waveform between 0 and 60 s, at 0.2 s periods
		if vel_disp_flag == 'VEL':
			data_std_r = np.repeat(std_r_list[i], 300) 
			data_std_z = np.repeat(std_z_list[i], 300)
		elif vel_disp_flag == 'DIS':
			data_std = np.repeat(1e-3, 300) 

		cov_r = np.diag(data_std_r * data_std_r)
		cov_z = np.diag(data_std_z * data_std_z)
		invCov_r = np.diag(1/(data_std_r * data_std_r))
		invCov_z = np.diag(1/(data_std_z * data_std_z))

		# Cholesky decomposition to avoid infinite or 0 determinant of covariance matrix (upper triangle)
		L_r = np.linalg.cholesky(cov_r)
		L_z = np.linalg.cholesky(cov_z)
		R_r = np.transpose(L_r)
		R_z = np.transpose(L_z)
		lndetCov_r = 2*np.sum(np.log(np.diag(R_r)))
		lndetCov_z = 2*np.sum(np.log(np.diag(R_z)))
		
		data_invCov.append([invCov_r, invCov_z])
		data_lndetCov.append([lndetCov_r, lndetCov_z])

	if GPS_flag == 'YES':
		data.append(data_gps)
		data_invCov.append(invCov_gps)
		data_lndetCov.append(lndetCov_gps)




