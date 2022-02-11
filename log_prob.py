#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time as Ti
import matplotlib.pyplot as plt

# normal distribution centered on "mu" with covariance "cov"
def log_likelihood(yhat, y, invCov, lndetCov):
	
	diff = yhat - y
	N = len(y)
	lnlike = -0.5 * (N*np.log(2*np.pi) + lndetCov + np.dot(diff, np.dot(invCov, diff)))
	return lnlike
    
def log_uniform_prior(md_vec, lb, ub): 
	if np.all(md_vec > lb) & np.all(md_vec < ub):
		return 0.0
	return -np.inf

def log_GaussianTailed_prior(md_vec, inner_lb, inner_ub): 
	prior_vec = []
	for ii, param in enumerate(md_vec):
		prior = gaussian_tail(inner_lb[ii], inner_ub[ii], param)
		prior_vec.append(prior)

	prior_sum = np.sum(prior_vec)
	return prior_sum

def gaussian_tail(inner_lb, inner_ub, param, logscale='TRUE'):
	# Compute absolute bounds on the parameters
	width = inner_ub - inner_lb # width of the uniform part
	sig = width * 0.1
	lb = inner_lb - 3 * sig 
	ub = inner_ub + 3 * sig
	height = 1./(width + sig * np.sqrt(2*np.pi))

	# set up the empty prior vector for one paramter
	prior = np.zeros_like(param)

	if  np.all(param < lb) or np.all(param > ub):
		prior = -np.inf*np.ones_like(prior)
	else:
		mu_l = inner_lb
		mu_u = inner_ub

		if isinstance(param, float): # when input paramter is a scalar
			if (param >= inner_lb) & (param <= inner_ub):
				prior = np.log(height)
			elif param < inner_lb:
				prior = np.log(height)-0.5*np.sum(((param - mu_l)/sig)**2)
			elif param > inner_ub:
				prior = np.log(height)-0.5*np.sum(((param - mu_u)/sig)**2)
		else: # when input a range of parameter value, for plotting
			prior = np.empty_like(param)
			for ii, param_value in enumerate(param):
				if (param_value >= inner_lb) & (param_value <= inner_ub):
					prior[ii] = np.log(height)
				elif param_value < inner_lb:
					prior[ii] = np.log(height)-0.5*np.sum(((param_value - mu_l)/sig)**2)
				elif param_value > inner_ub:
					prior[ii] = np.log(height)-0.5*np.sum(((param_value - mu_u)/sig)**2)
		if logscale == 'FALSE':
			prior = np.exp(prior)

	return prior

			