#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt

def calcol_alyt(consts, plot = 'NO'):
	'''
	Analytical solution to caldera collpase with static-dynamic friction.
	Taiyi Wang July 30, 2021

	features in addition to Kumagai's model:
	1. magma inertia 2. radiation damping 3. zero magma outflow during collapse
	
	Input:
	consts = [piston density (kg/m^3), effective magma density (kg/m^3), piston length (m), gravitational constant (m/s^2), piston radius (m),
	total compressibility (Pa^-1), shear stress change (Pa), chamber volume (m^3) ]

	Output:
	T  = time, evenly sampled at 0.25 s
	Ut = displacement time series (m)
	Pt = pressure change time series (Pa)
	St = dynamic shear stress change time series (Pa)
	'''
    
	# Unpack constants
	rho_p = consts[0]; eff_rho_m = consts[1]; L = consts[2]; g = consts[3]; R = consts[4];
	beta = consts[5]; dtau = consts[6]; V_c = consts[7]; T = consts[8];
	pi = np.pi

	# Intermediate constants
	A = pi * R**2              # cross-sectional area of piston
	C = 2 * pi * R             # circumference of piston
	V_p = A * L                # volume of piston
	m_p = rho_p * V_p          # mass of piston
	M = m_p + V_c*eff_rho_m    # effective mass for the whole system

	# Characteristic quantities
	l_star = beta*V_c*M*g/(pi**2 * R**4)
	t_star = (beta*V_c*M/(pi**2 * R**4))**(1/2)
	p_star = M*g/(pi*R**2)

	# Dimensionless results
	t_max_hat = pi
	t_hat = np.arange(0, t_max_hat, 0.01)

	pi_0 = 2*pi*R*L/(M*g)*dtau
	
	u_hat = -pi_0*np.cos(t_hat)+pi_0                                 # displacement
	p_hat = u_hat                                                    # perturbation pressure 

	# Dimensional results
	t = t_hat * t_star
	p = p_hat * p_star
	u = u_hat * l_star
	
	s = np.repeat(-dtau, len(t_hat))
	s[0] = 0; s[-1] = -2*dtau

	# Make the result smooth functions through time as Green's function inputs
	
	tn = np.concatenate((np.linspace(-25,-1e-3,10), t, np.linspace(t[-1]+1e-3,t[-1]+65,30))) # time range has to contain resampling period
	ut = np.concatenate((np.repeat(u[0], 10), u, np.repeat(u[-1], 30)))
	pt = np.concatenate((np.repeat(p[0], 10), p, np.repeat(p[-1], 30)))
	st = np.concatenate((np.repeat(s[0], 10), s, np.repeat(s[-1], 30))) # Note the dynamic shear stress could also be calculated as (m*g - pressure*(pi*R**2) - M * a)/(2 * pi * R * L)

	Ut = np.interp(T, tn, ut);
	Pt = np.interp(T, tn, pt);
	St = np.interp(T, tn, st);

	# Compute static changes
	delta_t = pi
	delta_u = 2*pi_0
	delta_p = delta_u

	DeltaT = delta_t * t_star
	DeltaU = delta_u * l_star
	DeltaP = delta_p * p_star
	DeltaS = s[-1]   # Static drop in stress

	if plot == 'YES':
		plt.figure()
		plt.plot(t_hat, u_hat)
		plt.xlabel(r'$\hat{t}$')
		plt.ylabel(r'$\hat{u}$')

	return T, Ut, Pt, St, DeltaT, DeltaU, DeltaP, DeltaS


