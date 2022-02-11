#!/usr/bin/env python
# coding: utf-8

# This function makes diagnostic plots of the emcee results

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.patches import Ellipse
from scipy.stats import gaussian_kde
import corner
import emcee
import os

import load_data 
import load_all_gfs
import pred as pd
import log_prob as lp

# load output to inversion
directory = os.getcwd()
reader = emcee.backends.HDFBackend(directory+'/inversion_output/sf_magma_acc_gps_1e6itr_final/output.h5')         # invert gps static displacement + VLP + observational constraints

burnin = 800
cutoff_logP = -0.6e8 # throw away samples with a log probability below this value, in case some walkers get stuck in less desirable modes
samples = reader.get_chain(discard=burnin) #of the shape (iterations, Nwalkers, Nparams)

samples_flat = reader.get_chain(discard=burnin, flat=True)
log_prob = reader.get_log_prob(discard=burnin)
log_prob_flat = reader.get_log_prob(discard=burnin, flat=True)
log_prior= reader.get_blobs(discard=burnin, flat=True)

samples_flat_p = samples_flat.copy()
samples_flat_p[:, 1] = samples_flat_p[:, 1]/1e9
samples_flat_p[:, 2] = samples_flat_p[:, 2]/1e9

burnin_gps = 100
Nvar= 1000 # number of models used to make plot prediction variance
#samples_gps_flat = reader_gps.get_chain(discard=burnin_gps, flat=True)

choose_idx = np.argwhere(log_prob_flat > cutoff_logP).squeeze() # choose these samples, which have log probability higher than a certain value

MAP_idx = np.argmax(log_prob_flat)

Nvar_idx = np.random.randint(len(choose_idx), size=Nvar)
MAP = samples_flat[MAP_idx,:] 

#MAP[-1] = -0.3
#MAP[-2] = -1
MAP[-3] = 0.5
#MAP[-7] = 0

# load input to inversion
input = np.load(directory+'/inversion_output/sf_magma_acc_gps_1e6itr_final/input.npz', allow_pickle= 'true')
param_vec = input['param_vec']
bnds = input['bnds']
means = input['means']

print('90 % credible interval for')
print('shape',np.shape(samples_flat))

labels = [r'$log_{10} \beta (Pa^{-1})$', r'$V_c (km^3)$', r'$\Delta \tau (MPa)$', r'$\phi \rho_m (kg/m^3)$',  r'$\alpha$', r'$\rho_p (kg/m^3)$', r'$R(km)$']

#Plot correlations -------------------------------------------------------------------------------------------------------------------------------
#fig = corner.corner(samples_flat[:, 0:7], labels=labels, no_fill_contours=True, plot_density=True, plot_contours=True, plot_datapoints=False)
#plt.show()

#Plot PDF of model parameters----------------------------------------------------------------------------------------------------------------------
Nbins = 500 # number of bins
prefer_bnds = [[-9.94, -9.5], [1.7, 7.2], [0.176, 0.24], [0, 0.4], [0.879, 0.93], [2.27, 2.92], [0.3, 0.55]] # narrow the range to where the PDF is

fig, axes = plt.subplots(3, 3, figsize=(10,10), dpi=100)

for i, ax in enumerate(axes.flatten()[0:7]):
	xlb = prefer_bnds[i][0]
	xub = prefer_bnds[i][1]

	# for gps+VLP
	density_PDF = gaussian_kde(samples_flat[choose_idx,i])
	density_PDF.covariance_factor = lambda : 0.05 #Smoothing parameter
	density_PDF._compute_covariance()

	# for gps only
	#density_gps_PDF = gaussian_kde(samples_gps_flat[:,i])
	#density_gps_PDF.covariance_factor = lambda : 0.05 # Smoothing parameter
	#density_gps_PDF._compute_covariance()

	sig = (bnds[i, 1] - bnds[i, 0])*0.1 # standard deviation of priors
	min_param = bnds[i, 0]-sig*3
	max_param = bnds[i, 1]+sig*3

	param_range = np.linspace(xlb, xub, Nbins)
	#param_range_gps = np.linspace(min_param, max_param, Nbins)

	#ax.fill_between(param_range_gps, lp.gaussian_tail(bnds[i, 0], bnds[i, 1], param_range, logscale='FALSE'), color='dodgerblue', alpha=0.4); 
	#ax.fill_between(param_range_gps, density_gps_PDF(param_range), color='orangered', alpha=0.4)
	ax.fill_between(param_range, density_PDF(param_range), color='orangered', alpha=0.4);
	
	ax.set_xlabel(labels[i])
	ax.set_xlim([xlb, xub])
	#ax.hist(samples_flat[choose_idx,i], Nbins, color=[0,0,0, 0.5], density="true", histtype="stepfilled",); ax.set_xlabel(labels[i])
	ax.axvline(x=samples_flat[MAP_idx,i], color = 'r', linestyle='--')

	# compute 90% confidence interval in the most concentrated region of PDFs
	samples_flat_i = samples_flat[:,i]
	samples_flat_i = samples_flat_i[(samples_flat_i>xlb)*(samples_flat_i<xub)]
	print(labels[i], 'is [', np.percentile(samples_flat_i, 5), np.percentile(samples_flat_i, 95), ']')

plt.tight_layout();
#plt.savefig('param_PDFs.eps', format='eps')
plt.show()

# Plot PDF of time shifts 
fig, axes = plt.subplots(4, 3, figsize=(10,10), dpi=100)
labels_time_shift = [r'radial time shift HMLE(s)', r'radial time shift PAUD(s)', r'radial time shift RSDD(s)', r'radial time shift HLPD(s)',  r'radial time shift MLOD(s)', r'radial time shift STCD(s)', r'vertical time shift HMLE(s)', r'vertical time shift PAUD(s)', r'vertical time shift RSDD(s)', r'vertical time shift HLPD(s)' , r'vertical time shift MLOD(s)' , r'vertical time shift STCD (s)']
for i, ax in enumerate(axes.flatten()[0:12]):
	#ax.hist(samples_flat_p[:,i+7], Nbins, density="true", histtype="stepfilled", fc=(255/255,64/255,64/255,0.5)); ax.set_xlabel(labels_time_shift[i])
	density = gaussian_kde(samples_flat[:,i+7])
	density.covariance_factor = lambda : 0.1 #Smoothing parameter
	density._compute_covariance()

	sig = (bnds[i+7, 1] - bnds[i+7, 0])*0.1 # standard deviation of priors
	min_param = bnds[i+7, 0]-sig*3
	max_param = bnds[i+7, 1]+sig*3
	param_range = np.linspace(min_param, max_param, 100)

	ax.fill_between(param_range, lp.gaussian_tail(bnds[i+7, 0], bnds[i+7, 1], param_range, logscale='FALSE'), color='dodgerblue', alpha=0.4); 
	ax.fill_between(param_range, density(param_range), color='orangered', alpha=0.4); ax.set_xlabel(labels_time_shift[i])
	ax.axvline(x=samples_flat[MAP_idx,i+7], color = 'r', linestyle='--')
	print(labels_time_shift[i], 'is [', np.percentile(samples_flat[:,i+7], 5), np.percentile(samples_flat[:,i+7], 95), ']')
plt.tight_layout();
plt.show()

#Plot the log probability--------------------------------------------------------------------------------------------------------------------
nwalkers = np.shape(log_prob)[1]
plt.figure()
for ii in np.arange(1, nwalkers, 1):
	plt.plot(log_prob[:,ii], '.-')
plt.xlabel('number of iterations after burn in')
plt.ylabel('log probability')
plt.ylim([1e4,5.5e4])
#plt.ylim([0,150])
plt.show()

#Plot the MAP waveform with data---------------------------------------------------------------------------------------------------------------------

print('maximum log probability is', np.max(log_prob))

vel_disp_flag = param_vec[6]
GPS_flag = param_vec[9]
lb_time = param_vec[4]
ub_time = param_vec[5]

# load data
load_data.init(lb_time, ub_time, vel_disp_flag, GPS_flag)

data = load_data.data
data_invCov = load_data.data_invCov
data_lndetCov = load_data.data_lndetCov

# load Green's functions
T = param_vec[10] 
mt_gf_file = 'greens_functions/halfspace_Kilauea/half_1.94_mt/'
sf_gf_file = 'greens_functions/halfspace_Kilauea/half_1.94_sf/'
#mt_gf_file = 'greens_functions/twolayer/twolay_1.94_mt/'
#sf_gf_file = 'greens_functions/twolayer/twolay_1.94_sf/'
load_all_gfs.init(mt_gf_file, sf_gf_file, T)

# rescale the MAP model to linear
MAP[0] = 10**MAP[0] # convert compressibility back into linear scale
MAP[1] = MAP[1] * 1e9
MAP[2] = MAP[2] * 1e6
MAP[3] = MAP[3] * 1e3
MAP[5] = MAP[5] * 1e3
MAP[6] = MAP[6] * 1e3

print('MAP model is', MAP)

# Make forward prediction for the best fit model
pred = pd.pred(MAP, param_vec) # in the order [pred_time, pred_HMLE, pred_PAUD, pred_RSDD, (DeltaT, DeltaU, DeltaP)]

# Make forward prediction for an ensemble of models

'''
pred_vars = []
for idx_md, md in enumerate(samples_flat[choose_idx[Nvar_idx],:]): # choose random samples that exclude NaN predictions

	    # convert parameters back into linear scale
	    md[0] = 10**md[0] 
	    md[1] = md[1] * 1e9
	    md[2] = md[2] * 1e6
	    md[3] = md[3] * 1e3
	    md[5] = md[5] * 1e3
	    md[6] = md[6] * 1e3

	    # use the same time shift as MAP model
	    #md[-1] = -0.3
	    #md[-2] = -1
	    md[-3] = 0.5
	    #md[-7] = 0

	    pred_var = pd.pred(md, param_vec)
	    pred_vars.append(pred_var)
'''

if GPS_flag == 'YES':
	DeltaT = pred[-2][0]
	DeltaU = pred[-2][1]
	DeltaP = pred[-2][2]
	DeltaS = pred[-2][3]

	Er_k = pred[-2][-1]
	Er_f = pred[-2][-2]
	Er_m = pred[-2][-3]

elif GPS_flag == 'NO':
	DeltaT = pred[-1][0]
	DeltaU = pred[-1][1]
	DeltaP = pred[-1][2]
	DeltaS = pred[-1][3]

	Er_k = pred[-1][-1]
	Er_f = pred[-1][-2]
	Er_m = pred[-1][-3]


print('event duration is:', DeltaT, 's.')
print('collapse displacement is:', DeltaU, 'm.')
print('co-collapse pressure increase is:', DeltaP/1e6, 'MPa.')
print('total stress drop is:', DeltaS/1e6, 'MPa.')
print('total radiated energy is', Er_k/1e12, '10^12 J')
print('moment tensor radiated energy is', Er_m/1e12, '10^12 J')
print('single force radiated energy is', Er_f/1e12, '10^12 J')

subplot_labels = ['HMLE', 'PAUD', 'RSDD', 'HLPD', 'MLOD', 'STCD']
distance_labels = ['6.33 km', '7.61 km', '5.68 km', '12.9 km', '14.7 km', '16.2 km']

fig, axes = plt.subplots(1, 2, figsize=(10,3), sharex=True, sharey=True)
axes_flat = axes.flatten()

for idx, i in enumerate([2,0,1,3,4,5]): # ordered using distance
	sta_data_time = data[i][0]
	sta_data_r = data[i][1]
	sta_data_z = data[i][2]
	#sta_data_cov = data_cov[i]

	sta_pred_time = pred[0][i]

	sta_pred_r = pred[i+1][0]                  # radial component 
	sta_pred_z = pred[i+1][1]                  # vertical component 

	# Interpolate both data and prediction so that they are not over-sampled
	
	time = np.arange(0, 60, 0.2)
	sta_pred_r_intp = np.interp(time, sta_pred_time[0], sta_pred_r)  
	sta_pred_z_intp = np.interp(time, sta_pred_time[1], sta_pred_z)
	sta_data_r_intp = np.interp(time, sta_data_time, sta_data_r) 
	sta_data_z_intp = np.interp(time, sta_data_time, sta_data_z)

	ax_l = axes[0]
	ax_r = axes[1]

	# Plot variance of predicted waveforms
	'''
	for idx_var, pred_var in enumerate(pred_vars):
		sta_pred_var_time = pred_var[0][i]
		sta_pred_var_r = pred_var[i+1][0]                  # radial component 
		sta_pred_var_z = pred_var[i+1][1]                  # vertical component 

		sta_pred_var_r_intp = np.interp(time, sta_pred_var_time[0], sta_pred_var_r) 
		sta_pred_var_z_intp = np.interp(time, sta_pred_var_time[1], sta_pred_var_z)

		ax_l.plot(sta_pred_var_time[0], sta_pred_var_r+(7-idx)*5e-3, 'r', alpha=0.02)
		ax_r.plot(sta_pred_var_time[1], sta_pred_var_z+(7-idx)*5e-3, 'r', alpha=0.02)
	'''

	# Plot MAP predicted waveforms

	ax_l.plot(sta_pred_time[0], sta_pred_r+(7-idx)*5e-3, 'r')
	ax_l.plot(sta_data_time, sta_data_r+(7-idx)*5e-3, 'k')
	
	ax_r.plot(sta_pred_time[1], sta_pred_z+(7-idx)*5e-3, 'r')
	ax_r.plot(sta_data_time, sta_data_z+(7-idx)*5e-3, 'k')

	sta_pred_intp = np.concatenate((sta_pred_r_intp, sta_pred_z_intp))
	sta_data_intp = np.concatenate((sta_data_r_intp, sta_pred_z_intp))
	
	ax_l.set_xlim([5, 40])
	ax_r.set_xlim([5, 40])


	if idx == 0:
		waveform_pred = sta_pred_intp
		waveform_data = sta_data_intp
	else:
		waveform_pred = np.concatenate((waveform_pred, sta_pred_intp))
		waveform_data = np.concatenate((waveform_data, sta_data_intp))

	# check out the likelihood of best fit model
	sta_data_invCov_r = data_invCov[i][0]
	sta_data_invCov_z = data_invCov[i][1]
	sta_data_lndetCov_r = data_lndetCov[i][0]
	sta_data_lndetCov_z = data_lndetCov[i][1]
	sta_llike = lp.log_likelihood(sta_pred_r_intp, sta_data_r_intp, sta_data_invCov_r, sta_data_lndetCov_r)+lp.log_likelihood(sta_pred_z_intp, sta_data_z_intp, sta_data_invCov_z, sta_data_lndetCov_z)
	print('log likelihood of', subplot_labels[i],'is',sta_llike)
	
	
#plt.subplots_adjust(wspace= 0.1, hspace = 5)
fig.tight_layout()
plt.show()

# Plot the MAP GPS offset with data--------------------------------------------------------------------------------------------------------------------
# load data standard deviation
GPS = np.loadtxt(directory+'/inversion_input/GPS_avg_offset_and_std_last32.txt', delimiter=',', skiprows=1)
GPS = np.delete(GPS, [4, 6, 11], 0) # Delete BYRL, CRIM, and UWEV, in that order, for being too close to ring fault
# reorder into a vector [E, N, U]
data_std_gps_E = GPS[:, 3]
data_std_gps_N = GPS[:, 4]
data_std_gps_U = GPS[:, 5]

if GPS_flag == 'YES':
	GPS_pos_xy = np.loadtxt(directory+'/inversion_input/GPS_local_xy.txt', delimiter=',', skiprows=1) 
	line_xy = np.loadtxt(directory+'/inversion_input/kilauea_topo_xy_new.txt', delimiter=',', skiprows=1)
	GPS_labels = ['69FL', '92YN', 'AHUP', 'BDPK', 'BYRL', 'CNPK', 'CRIM', 'DEVL', 'OUTL', 'PUHI', 'PWRL', 'UWEV', 'V120', 'VSAS']

	GPS_pos_xy = np.delete(GPS_pos_xy, [4, 6, 11], 0) # exclude BYRL, CRIM, UWEV
	GPS_labels = np.delete(GPS_labels, [4, 6, 11], 0)

	data_offset_E = data[-1][:11]
	data_offset_N = data[-1][11:22]
	data_offset_U = data[-1][22:33]

	pos_data_idx = np.argwhere(data_offset_U>=0)
	neg_data_idx = np.argwhere(data_offset_U<0)

	pred_offset_E = pred[-1][:11]
	pred_offset_N = pred[-1][11:22]
	pred_offset_U = pred[-1][22:33]

	#np.savetxt('MAP_gps_pred', [pred_offset_E, pred_offset_N, pred_offset_U])

	data_offset_R = (data_offset_E**2 + data_offset_N**2)**(1/2)
	pred_offset_R = (pred_offset_E**2 + pred_offset_N**2)**(1/2)

	pos_pred_idx = np.argwhere(pred_offset_U>=0)
	neg_pred_idx = np.argwhere(pred_offset_U<0)

	figure, ax = plt.subplots()
	plt.autoscale(False)
	plt.xlim([-7.5, 7.5]);plt.ylim([-7.5, 7.5])
	plt.plot(line_xy[:,0]/1000, line_xy[:,1]/1000, 'k-', linewidth = 0.5)
	plt.quiver(GPS_pos_xy[:,0]/1000, GPS_pos_xy[:,1]/1000, data_offset_E, data_offset_N, color = 'k', scale=0.3)
	plt.quiver(GPS_pos_xy[:,0]/1000, GPS_pos_xy[:,1]/1000, pred_offset_E, pred_offset_N, color = 'r', scale=0.3)

	
	for i, idx in enumerate(neg_data_idx):
		data_U = np.abs(data_offset_U[idx])
		data_U_lb = np.abs(data_offset_U[idx]) - data_std_gps_U[idx]
		data_U_ub = np.abs(data_offset_U[idx]) + data_std_gps_U[idx]

		print(data_U, data_U_lb, data_U_ub);

		c = plt.Circle((GPS_pos_xy[idx,0]/1000, GPS_pos_xy[idx,1]/1000), data_U*65,  color = 'k', linestyle = '--', fill=False)

		if data_U_lb >= 0: # only plot lower bound if it is non-negative
			c_lb = plt.Circle((GPS_pos_xy[idx,0]/1000, GPS_pos_xy[idx,1]/1000), data_U_lb*65,  color = '0.8', linestyle = '-', fill=False)
			ax.add_patch(c_lb); 

		c_ub = plt.Circle((GPS_pos_xy[idx,0]/1000, GPS_pos_xy[idx,1]/1000), data_U_ub*65,  color = '0.8', linestyle = '--', fill=False)
		ax.add_patch(c); ax.add_patch(c_ub);
	
	for i, idx in enumerate(pos_data_idx):
		data_U = np.abs(data_offset_U[idx])
		data_U_lb = np.abs(data_offset_U[idx]) - data_std_gps_U[idx]
		data_U_ub = np.abs(data_offset_U[idx]) + data_std_gps_U[idx]

		print(data_U, data_U_lb, data_U_ub);

		c = plt.Circle((GPS_pos_xy[idx,0]/1000, GPS_pos_xy[idx,1]/1000), data_U*65,  color = 'k', linestyle = '-', fill=False)

		if data_U_lb >= 0: # only plot lower bound if it is non-negative
			c_lb = plt.Circle((GPS_pos_xy[idx,0]/1000, GPS_pos_xy[idx,1]/1000), data_U_lb*65,  color = '0.8', linestyle = '-', fill=False)
			ax.add_patch(c_lb);

		c_ub = plt.Circle((GPS_pos_xy[idx,0]/1000, GPS_pos_xy[idx,1]/1000), data_U_ub*65,  color = '0.8', linestyle = '--', fill=False)
		ax.add_patch(c); ax.add_patch(c_ub);

	for i, idx in enumerate(neg_pred_idx):
		c = plt.Circle((GPS_pos_xy[idx,0]/1000, GPS_pos_xy[idx,1]/1000), np.abs(pred_offset_U[idx])*65,  color = 'r', linestyle = '--', fill=False)
		ax.add_patch(c)

	for i, idx in enumerate(pos_pred_idx):
		c = plt.Circle((GPS_pos_xy[idx,0]/1000, GPS_pos_xy[idx,1]/1000), np.abs(pred_offset_U[idx])*65,  color = 'r', linestyle = '-', fill=False)
		ax.add_patch(c)

	# Plot error ellipses for radial components
	for i in np.arange(0, 11):

		e = Ellipse(xy=(GPS_pos_xy[i,0]+data_offset_E[i], GPS_pos_xy[i,1]+data_offset_N[i]), width=data_std_gps_E[i]*100, height=data_std_gps_N[i]*100, 
                        edgecolor='r', fc='None')
		ax.add_patch(e)

	# legend
	plt.quiver(6, 6, -0.02, 0, color = 'k', scale=0.3)
	c = plt.Circle((6, 6), 0.02*65,  color = 'k', linestyle = '-', fill=False)
	ax.add_patch(c)
	plt.text(6, 6, '2 cm')
	plt.plot(param_vec[2][0]/1000, param_vec[2][1]/1000, '+', markersize = 12)

	plt.xlabel('East (km)');plt.ylabel('North (km)')
	plt.show()


# percentage of variance explained
if GPS_flag == 'YES':
	gps_pred_vec = pred[-1]
	gps_data_vec = data[-1]

	# calculate GPS likelihood
	
	gps_data_invCov = data_invCov[-1]
	gps_data_lndetCov = data_lndetCov[-1]
	gps_llike = lp.log_likelihood(gps_pred_vec, gps_data_vec, gps_data_invCov, gps_data_lndetCov)
	print('GPS log likelihood is', gps_llike)
	

	gps_perc_explained = 1 - np.sum((gps_pred_vec - gps_data_vec)**2)/np.sum(gps_data_vec**2)
	print('MAP model explains', gps_perc_explained*100, r'% of variance in the GPS data')


waveform_perc_explained = 1 - np.sum((waveform_pred - waveform_data)**2)/np.sum((waveform_data)**2)
print('MAP model explains', waveform_perc_explained*100, r'% of variance in the waveform data')


# Autocorrelation analysis-----------------------------------------------------------------------------------------------------------------------
# This is not trustable yet since the plot changes significantly each time I plot the same data
'''
N = np.linspace(1, 99, 30).astype(int)
tau_estimates = np.empty([len(N), nwalkers])
avg_tau = np.empty(len(N))

# Note samples is organized as (Nitr, Nwalkers, Nparams)
# Here we compute the auto-coorelation time for one parameter averaged over all walkers
for i, n in enumerate(N):
	for j in np.arange(1, nwalkers):
		tau_estimates[i, j] = emcee.autocorr.integrated_time(samples[:n,j,0], c=5, tol=50, quiet=True)
    
    # avoid nans when taking the average among walkers
	avg_tau[i] = np.mean(tau_estimates[i, :][~np.isnan(tau_estimates[i, :])])


plt.figure()
plt.loglog(N, avg_tau, "o-")
plt.xlabel('number of samples')
plt.ylabel(r'autocorrelation time $\tau$')
plt.show()
'''
