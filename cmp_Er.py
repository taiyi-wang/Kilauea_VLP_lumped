#!/usr/bin/env python
# coding: utf-8
import numpy as np
import time as Ti
import matplotlib.pyplot as plt

def cmp_kinematic_Er(t, Mt, Ft, vp, vs, rho):
	''' 
	Compute kinematic radiated energy from time dependent single force and moment tensor
	Input:
	t      = times (1, Ntimes)
	Mt     = time dependent moment tensor (3, 3, Ntimes)
	Ft     = time dependent single force in z (1, Ntimes)
	vp, vs = p and s wave velocities (m/s)
	rho    = crustal density (kg/m^3)

	Output:
	Er_m   = radiated energy from Eshelby moment tensor (J)
	Er_f   = radiated energy from single force (J)
	Er     = total radiated energy from the kinematic, moment tensor + single force model (J)
	'''
    
    # Diagonal elements of the time dependent moment tensor (assuming the semi-axes are aligned with the Cartesian coordinate system)
	M1 = Mt[0, 0, :]
	M2 = Mt[1, 1, :]
	M3 = Mt[2, 2, :]

	# derivatives
	pi = np.pi
	ddM1 = np.gradient(np.gradient(M1, t), t) 
	ddM2 = np.gradient(np.gradient(M2, t), t) 
	ddM3 = np.gradient(np.gradient(M3, t), t) 
	dFt = np.gradient(Ft, t)

	# radiated energy 
	Er_m_p = 1/(60*pi*rho*vp**5)*np.trapz(3*ddM1**2 + 3*ddM2**2 + 3*ddM3**2 + 2*ddM1*ddM2 + 2*ddM2*ddM3 + 2*ddM1*ddM3)  # Eshelby moment tensor p-wave
	Er_m_s = 1/(30*pi*rho*vs**5)*np.trapz(ddM1**2 + ddM2**2 + ddM3**2 - ddM1*ddM2 - ddM1*ddM3 - ddM2*ddM3)           # Eshelby moment tensor s-wave
	Er_m  = Er_m_p + Er_m_s

	Er_f_p = 1/(12*pi*rho*vp**3)*np.trapz(dFt**2) # vertical single force p-wave
	Er_f_s = 1/(6*pi*rho*vs**3)*np.trapz(dFt**2)  # vertical single force s-wave
	Er_f = Er_f_p + Er_f_s

	Er = Er_m + Er_f # total radiated energy

	return Er_m, Er_f, Er







