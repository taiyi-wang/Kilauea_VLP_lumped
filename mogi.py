#!/usr/bin/env python
# coding: utf-8

import numpy as np

def mogi(dp, a, nu, mu, x, y, d):
	r =  (x**2 + y**2)**0.5
	theta = np.arctan2(y, x)
	uz = (1-nu)*dp*a**3/mu*d/(r**2 + d**2)**(3/2)
	ur = (1-nu)*dp*a**3/mu*r/(r**2 + d**2)**(3/2)

	ux = ur*np.cos(theta)
	uy = ur*np.sin(theta)

	return ux, uy, uz

