#!/usr/bin/env python
# coding: utf-8
# Forward prediction function for waveforms

import sys
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import gc
import time as Ti
import emcee
import os

os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
from multiprocessing import cpu_count

import synthetic as syn
import helpers as hp
import ButterWorth as bw
import pred as pd
import objective as ob
import load_data 
import load_all_gfs

# Set inversion parameters----------------------------------------------------------------------------------------------------------------------------

parallel_flag = 'YES' # use multi-processing to parallelize inversion
new_run_flag = 'YES'   # using new initial samples or using the last sample of previous run, assuming output file from previous run is 'output.h5'
vel_disp_flag = 'VEL' # invert for time series velocity (VEL) or displacment (DIS); when 'VEL', GPS_flag = 'NO' (mandatory).
comp_flag = 'NO'      # whether to get moment/single force components of the predicted waveforms. Should always be 'NO' for inversion.
coord_flag = 'CYLINDRICAL' # whether to get waveforms in ENU ('CARTESIAN') or Radial-vertical-tangential ('CYLINDRICAL')
GPS_flag = 'YES'      # invert for the ENU components of GPS stations ('YES').

Nitr = 500
nwalkers = 100
step_scale = 2 # stretch move scale parameter (default = 2)

mt_depth = 1.94  # moment tensor depth [km], also corresponding to chamber centroid depth. Choices are [1.5, 1.84, 1.94, 2.12]: 
sf_depth = 1.94  # single force depth [km]. Choices are [0.5, 1.0, 1.5, 1.94]

dt = 0.05         # dynamic model output sampling period (s)
#------------------------------------------------------------------------------------------------------------------------------------

# get path of current directory
directory = os.getcwd()

# Load accelerometer locations (w/ origin NPIT)
accel_pos_xy = np.loadtxt(directory+'/inversion_input/acce_local_xy.txt', delimiter=',', skiprows=1) 
accel_labels = ['HMLE', 'NPT', 'PAUD', 'RSDD', 'UWE']

# Load broadband seismometer locations (w/ origin NPIT)
seism_pos_xy = np.loadtxt(directory+'/inversion_input/seism_local_xy.txt', delimiter=',', skiprows=1) 
seism_labels = ['HLPD', 'JOKA', 'MLOD', 'STCD']

# Load GPS station locations, if required (w/ origin NPIT)
if GPS_flag == 'YES':
	# exclude UWE, NPT seismic stations, UWEV GPS station
	sta_pos = np.zeros((17, 3))
    
	GPS_pos_xy = np.loadtxt(directory+'/inversion_input/gps_local_xy.txt', delimiter=',', skiprows=1) 
	GPS_pos_xy = np.delete(GPS_pos_xy, [4, 6, 11], 0) # Delete BYRL, CRIM, and UWEV, in that order, for being too close to ring fault
	GPS_labels = ['69FL', '92YN', 'AHUP', 'BDPK', 'BYRL', 'CNPK', 'CRIM', 'DEVL', 'OUTL', 'PUHI', 'PWRL', 'UWEV', 'V120', 'VSAS']
    
	sta_pos[:3, :2] = accel_pos_xy[[0, 2, 3], :]
	sta_pos[3:6, :2] = seism_pos_xy[[0, 1, 2],:]
	sta_pos[6:17, :2] = GPS_pos_xy[:11, :]

elif GPS_flag == 'NO':
	# exclude UWE, NPT
	sta_pos = np.zeros((6, 3)) 
	sta_pos[:3, :2] = accel_pos_xy[[0, 2, 3], :]
	sta_pos[3:6, :2] = seism_pos_xy[[0, 1, 2],:]


# piston and chamber locations (fixed, except piston length)
parameters = np.loadtxt(directory+'/inversion_input/parameters.txt', delimiter=',', usecols= 1)
chamber_cent = np.array(parameters[:3])
chamber_cent[2] = -mt_depth*1000 # z positive upwards
piston_cent = np.array(parameters[:3])

# Trim data to focus on the period of interest 
if vel_disp_flag == 'VEL':
	lb_time = 0
	ub_time = 60
elif vel_disp_flag == 'DIS':
	lb_time = 0
	ub_time = 30

# load data
load_data.init(lb_time, ub_time, vel_disp_flag, GPS_flag)
# load Green's functions (computed with mu = 3e9 Pa, rho = 3000 kg/m^3, nu = 0.25)

T = np.arange(-20, 60, dt)    # resampled time for dynamic model output
T[np.argmin(abs(T))] = 0      # make sure 0 s exists
mt_gf_file = 'greens_functions/halfspace_Kilauea/half_'+str(mt_depth)+'_mt/'
sf_gf_file = 'greens_functions/halfspace_Kilauea/half_'+str(sf_depth)+'_sf/'
load_all_gfs.init(mt_gf_file, sf_gf_file, T)

g = 9.8   # gravitational constant [m/s^2]
mu = 3e9  # crustal shear modulus [Pa] (should not be changed, because this is the assumed value for computing the Green's functions)
rho_c = 3000 # density of crust outside of ring fault
param_vec = [g, sta_pos, chamber_cent, piston_cent, lb_time, ub_time, vel_disp_flag, comp_flag, coord_flag, GPS_flag, T, mu, rho_c]

# Set up the inversion scheme -------------------------------------------------------------------------------------------------------------------
# 1. log10 compressibility 2. chamber volume 3. shear strength drop 4. effective magma density 5. aspect ratio 6. rock density 7. piston radius 8-13:radial time shift 14-20: vertical time shift (in the order of HMLE, PAUD, RSDD)
#best_fit = [-9.52, 5.21e+09, 1.95e+05,  1.01e+02, 1.1, 2.31e+03, 5.03e+02, 3.43, 3.33, 4.52, 1.48, 1.14, 6.99e-01, 3.12, 1.62, 3.19, -9.39e-01, -7e-01, -5e-01]
best_fit = [-9.64,  8.64e+09, 1.67e+05, 1.24e+01, 1.20e+00, 2.28e+03, 4.34e+02, 3.88, 3.77, 4.76, 1.76, 1.80, 0.58, 3.74, 3.44, 3.88, 0, 0, 0] # within a physically reasonable range
bnds = np.array([[-9.7, -8.88], [2, 7.23], [0.1, 1.3], [0.21, 0.87], [1, 1.4], [2.4, 2.8], [0.5, 1.3], [-3, 3], [-3, 3], [-3, 3], [-3, 3], [-3, 3], [-3, 3], [-3, 3], [-3, 3], [-3, 3], [-3, 3], [-3, 3], [-3, 3]])  # rescaled so that all parameters are of order 1
means = np.mean(bnds, axis = 1)
means[7:18] = best_fit[7:18] # use the best fit time shift to begin with 
std = (bnds[:, 1] - bnds[:, 0])/20 # only used to generate initial distribution of model parameter values

# Set up walkers-------------------------------------------------------------------------------------------------------------------------
if new_run_flag == 'YES':
	# set up new initial walkers
	print('setting up initial walkers')
	ndim = len(means)
	p0 = stats.truncnorm.rvs((bnds[:,0] - means) / std, (bnds[:,1] - means) / std, loc=means, scale=std, size=(nwalkers, ndim))
	print('finished setting up initial walkers')

	# save the input param_vec
	np.savez('input', param_vec=param_vec, bnds=bnds, means=means)

	# Don't forget to clear it in case the file already exists
	filename = "output.h5"
elif new_run_flag == 'NO':
	# Use existing walkers from previous run
	print('reading in last sample of last run')
	reader = emcee.backends.HDFBackend(directory+'/output.h5')                                     
	samples = reader.get_chain(discard=0) #of the shape (iterations, Nwalkers, Nparams)
	p0 = samples[-1]
	ndim = np.shape(p0)[1]     # set number of dimensions to be the same as previous run
	nwalkers = np.shape(p0)[0] # set number of walkers to be the same as previous run
	print('finished reading')

	# Don't forget to clear it in case the file already exists
	filename = "output_cont.h5"

# Set up the backend to save progress-------------------------------------------------------------------------------------------------------------
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

# Run inversion ----------------------------------------------------------------------------------------------------------------------------------
if parallel_flag == 'YES':
	ncpu = cpu_count()
	print("{0} CPUs".format(ncpu))
	with Pool() as pool:
		sampler = emcee.EnsembleSampler(nwalkers, ndim, ob.objective, args=[param_vec, bnds], moves = emcee.moves.StretchMove(a = step_scale), backend=backend, pool=pool)
		start = Ti.time()
		sampler.run_mcmc(p0, Nitr, progress=True)
		end = Ti.time()
		multi_time = end - start
		print("Multiprocessing took {0:.1f} seconds".format(multi_time))
else:

	sampler = emcee.EnsembleSampler(nwalkers, ndim, ob.objective, args=[param_vec, bnds], moves = emcee.moves.StretchMove(a = step_scale), backend=backend)

	start = Ti.time()
	sampler.run_mcmc(p0, Nitr, progress=True)
	end = Ti.time()
	serial_time = end - start
	print("Serial took {0:.1f} seconds".format(serial_time))


# Print out relevant information
log_prob_samples = sampler.get_chain(flat=True)
print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
print(np.shape(log_prob_samples))