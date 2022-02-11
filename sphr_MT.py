#!/usr/bin/env python
# coding: utf-8
from sympy import *
import numpy as np

def sphr_MT(a1, a2, a3, dp, consts = None):
	'''
	This function computes the moment tensor associated with a spheroid. 

	Input:
	a1, a2, a3 = length of two semi-minor axes and one semi-major axis (m)
	dp         = pressure change (Pa); either scalar or a dim = Ntimes list
	consts     = [poisson's ratio, crustal shear modulus (Pa)]

	Output:
	M = a 3 X Ntimes array with each column being the 3 components of the moment tensor at 1 time step
	
	Notes:
	1. For a test case, when the spheroid is a sphere
	S1111 = S2222 = S3333 = (7 - 5 * nu)/(15 * (1 - nu));
	S1122 = S2211 = S1133 = S2233 = S3311 = S3322 = (5 * nu - 1)/(15 * (1 -nu));
	Full space spherical point source moment
	m = 2*((1-nu)/(1-(2*nu)))*(pi*dp*(a1**3));
	2. The 'Integral' + 'evalf' is a workaround for a persistent bug that exists in current versions of sympy.integrate. 

	% Taiyi Wang 02/08/2021
	'''
	
	if consts == None:
		nu = 0.25
		mu = 3e9
	else:
		nu, mu = consts
	
	K = (2*mu*(1+nu))/(3*(1-2*nu))
	l = K - (2 * mu/3)  

	# Integrals in Eqn. 7.52
	Delta = lambda s: ((a1**2+s)**Rational(1, 2))*((a2**2+s)**Rational(1, 2))*((a3**2+s)**Rational(1, 2))
    
    # symbolically compute the integrals to ensure accuracy. The integrations converge very slowly as a function of s.
    # Therefore numerical integration can hardly generate accurate results.
    # Closed form analytical solutions only exist when a1, a2, a3 are known.
    # One way to check whether the integrals are correct is to use Eqn. 3.10 - 3.13 from Eshelby, 1957.

	s = Symbol('s')

	I1 = re(N(2*np.pi*a1*a2*a3*Integral(1/((a1**2 + s) * Delta(s)), (s, 0, oo)))).evalf()
	I2 = re(N(2*np.pi*a1*a2*a3*Integral(1/((a2**2 + s) * Delta(s)), (s, 0, oo)))).evalf()
	I3 = re(N(2*np.pi*a1*a2*a3*Integral(1/((a3**2 + s) * Delta(s)), (s, 0, oo)))).evalf()

	I11 = re(N(2*np.pi*a1*a2*a3*Integral(1/(((a1**2 + s)**2) * Delta(s)), (s, 0, oo)))).evalf()
	I22 = re(N(2*np.pi*a1*a2*a3*Integral(1/(((a2**2 + s)**2) * Delta(s)), (s, 0, oo)))).evalf()
	I33 = re(N(2*np.pi*a1*a2*a3*Integral(1/(((a3**2 + s)**2) * Delta(s)), (s, 0, oo)))).evalf()

	I12 = re(N(2*np.pi*a1*a2*a3*Integral(1/(((a1**2 + s)*(a2**2 + s)) * Delta(s)), (s, 0, oo)))).evalf()
	I13 = re(N(2*np.pi*a1*a2*a3*Integral(1/(((a1**2 + s)*(a3**2 + s)) * Delta(s)), (s, 0, oo)))).evalf()
	I23 = re(N(2*np.pi*a1*a2*a3*Integral(1/(((a2**2 + s)*(a3**2 + s)) * Delta(s)), (s, 0, oo)))).evalf()

	I21 = re(N(2*np.pi*a1*a2*a3*Integral(1/(((a2**2 + s)*(a1**2 + s)) * Delta(s)), (s, 0, oo)))).evalf()
	I31 = re(N(2*np.pi*a1*a2*a3*Integral(1/(((a3**2 + s)*(a1**2 + s)) * Delta(s)), (s, 0, oo)))).evalf()
	I32 = re(N(2*np.pi*a1*a2*a3*Integral(1/(((a3**2 + s)*(a2**2 + s)) * Delta(s)), (s, 0, oo)))).evalf()

	# The shape tensor 
    # components:
	S1111 = (3/(8*np.pi*(1-nu)))*(a1**2)*I11 + ((1-2*nu)/(8*np.pi*(1-nu)))*I1
	S2222 = (3/(8*np.pi*(1-nu)))*(a2**2)*I22 + ((1-2*nu)/(8*np.pi*(1-nu)))*I2
	S3333 = (3/(8*np.pi*(1-nu)))*(a3**2)*I33 + ((1-2*nu)/(8*np.pi*(1-nu)))*I3

	S1122 = (1/(8*np.pi*(1-nu)))*(a2**2)*I12 - ((1-2*nu)/(8*np.pi*(1-nu)))*I1
	S1133 = (1/(8*np.pi*(1-nu)))*(a3**2)*I13 - ((1-2*nu)/(8*np.pi*(1-nu)))*I1
	S2233 = (1/(8*np.pi*(1-nu)))*(a3**2)*I23 - ((1-2*nu)/(8*np.pi*(1-nu)))*I2

	S2211 = (1/(8*np.pi*(1-nu)))*(a1**2)*I21 - ((1-2*nu)/(8*np.pi*(1-nu)))*I2
	S3311 = (1/(8*np.pi*(1-nu)))*(a1**2)*I31 - ((1-2*nu)/(8*np.pi*(1-nu)))*I3
	S3322 = (1/(8*np.pi*(1-nu)))*(a2**2)*I32 - ((1-2*nu)/(8*np.pi*(1-nu)))*I3

	# form the tensor
	S = np.array([[S1111-1, S1122, S1133], [S2211, S2222-1, S2233], [S3311, S3322, S3333-1]], dtype = 'float')

	# Compute the transformation strain
	e_T = np.dot(np.linalg.inv(S), (((-1)/(3*K))*np.array([dp, dp, dp], dtype = 'float')))

	# Compute corresponding momemt tensor
	V = (4/3)*np.pi*a1*a2*a3
	sigma = ((2*mu)*(e_T))+ (l*sum(e_T))
	M = sigma * V # tension positive
	M = np.array([M[0], M[1], M[2]])

	return M



