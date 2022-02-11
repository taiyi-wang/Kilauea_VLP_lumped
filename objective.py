#!/usr/bin/env python
# coding: utf-8
# objective function for inversion

import sys
import numpy as np
import matplotlib.pyplot as plt
import gc
import pred as pd
import log_prob as lp
import load_data
import time as Ti

def objective(md_vec, param_vec, bnds):
    data = load_data.data
    data_invCov = load_data.data_invCov
    data_lndetCov = load_data.data_lndetCov
    
    n_sta = 6 # Number of seismic stations (3 accelerometer + 3 broadband)

    inner_lb = bnds[:,0] 
    inner_ub = bnds[:,1] 

    # compute log prior
    lprior = lp.log_GaussianTailed_prior(md_vec, inner_lb, inner_ub)

    # Convert input parameters back to linear scale
    md_vec[0] = 10**md_vec[0]
    md_vec[1] = md_vec[1]*1e9
    md_vec[2] = md_vec[2]*1e6
    md_vec[3] = md_vec[3]*1e3
    md_vec[5] = md_vec[5]*1e3
    md_vec[6] = md_vec[6]*1e3

    if not np.isfinite(lprior):
        lposterior = -np.inf
    else:
        # compute collapse duration from input
        pi = np.pi
        g = param_vec[0]; dz = np.abs(param_vec[2][2])
        beta = md_vec[0]; V_c = md_vec[1]; dtau = md_vec[2]; eff_rho_m = md_vec[3]; alpha = md_vec[4];
        rho_p = md_vec[5]; R = md_vec[6];

        ra = (alpha**2 * V_c * 3 / (4*pi))**(1/3)
        L = dz - ra                

        A = pi * R**2              # cross-sectional area of piston
        C = 2 * pi * R             # circumference of piston
        V_p = A * L                # volume of piston
        m_p = rho_p * V_p          # mass of piston
        M = m_p + V_c*eff_rho_m    # effective mass for the whole system

        pi_0 = 2*pi*R*L/(M*g)*dtau

        l_star = beta*V_c*M*g/(pi**2 * R**4)
        t_star = (beta*V_c*M/(pi**2 * R**4))**(1/2)
        p_star = M*g/(pi*R**2)

        delta_t = pi
        delta_u = 2*pi_0
        delta_p = delta_u

        DeltaT = delta_t * t_star
        DeltaU = delta_u * l_star
        DeltaP = delta_p * p_star

        if (DeltaT > 8) or (DeltaT < 2) or (DeltaU > 5) or (DeltaU < 2) or (DeltaP < 0.5e6) or (DeltaP > 4e6):
        # exclude cases with unrealistic duration, displacement, pressure change
            lprior = -1e8
    
        # Only compute likelihood if the proposal is within bounds
        pred = pd.pred(md_vec, param_vec)
      
        # loop through all seismic stations
        # factors adjusting for stations at various distances
        amp_corr_factor = [1, 1, 1, 1, 1, 1]

        for i in np.arange(0, n_sta, 1): 

        # in the order: HMLE, PAUD, RSDD, HLPD, MLOD, STCD
            acf = amp_corr_factor[i] # amplitude correction factor
        
            sta_data_time = data[i][0]
            sta_data_r  = data[i][1]
            sta_data_z  = data[i][2]
            sta_data_invCov_r = data_invCov[i][0]
            sta_data_invCov_z = data_invCov[i][1]
            sta_data_lndetCov_r = data_lndetCov[i][0]
            sta_data_lndetCov_z = data_lndetCov[i][1]

            sta_pred_time = pred[0][i]

            sta_pred_r = pred[i+1][0]                  # radial component 
            sta_pred_z = pred[i+1][1]                  # vertical component 

            # Interpolate both data and prediction so that they are not over-sampled
            time = np.arange(0, 60, 0.2)
            sta_pred_r = np.interp(time, sta_pred_time[0], sta_pred_r) 
            sta_pred_z = np.interp(time, sta_pred_time[1], sta_pred_z)
            sta_data_r = np.interp(time, sta_data_time, sta_data_r) 
            sta_data_z = np.interp(time, sta_data_time, sta_data_z)
            
            sta_llike = lp.log_likelihood(sta_pred_r, sta_data_r, sta_data_invCov_r, sta_data_lndetCov_r)+lp.log_likelihood(sta_pred_z, sta_data_z, sta_data_invCov_z, sta_data_lndetCov_z)

            if i == 0:
                seism_llike = sta_llike*acf
            else:
                seism_llike += sta_llike*acf

        if param_vec[9] == 'YES':
        # If invert for GPS data
            gps_data_vec = data[-1]
            gps_data_invCov = data_invCov[-1]
            gps_data_lndetCov = data_lndetCov[-1]
            gps_pred_vec = pred[-1]

            gps_llike = lp.log_likelihood(gps_pred_vec, gps_data_vec, gps_data_invCov, gps_data_lndetCov)

            lposterior = lprior+seism_llike+gps_llike*15
        elif param_vec[9] == 'NO':
            lposterior = lprior+seism_llike
        
       
    return lposterior

    
        


