#!/usr/bin/env python
# coding: utf-8

import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt
import gc

import helpers as hp
import source_setup as ss
import bodywaves_functions as bw
import surfacewaves_functions as sw
import sphr_MT as spmt
import load_all_gfs

import time as Ti


def translate(new_origin, pos, chamberCent, pistonCent):
    '''
    transforms the original position vectors with new origin
    
    ---INPUTS---
    new_origin  : (3)             : position of new origin in old coordinates
    pos         : (# stations, 3) : station positions in old coordinates
    chamberCent : (3)             : chamber centroid in old coordinates
    pistonCent  : (3)             : piston centroid in old coordinates
    ---RETURNS---
    new_pos     : (# stations, 3) : stations positions w.r.t. new origin
    new_chamber : (3)             : chamber centroid w.r.t. new origin
    new_piston  : (3)             : piston centroid w.r.t. new origin
    '''
    nn = pos.shape[0]
    
    new_pos = np.zeros((nn, 3))
    for ii in range(nn):
        new_pos[ii] = pos[ii] - new_origin
        
    new_chamber = chamberCent - new_origin
    new_piston = pistonCent - new_origin
    
    return new_pos, new_chamber, new_piston


def cylindrical(original):
    '''
    transforms from Cartesian to cylindrical (r, phi, z)
    
    NB: +z downwards
    
    ---INPUTS---
    original : (# stations, 3)
    ---RETURNS---
    new      : (# stations, 3)
    '''
    nn = original.shape[0]
    
    new = np.zeros((nn, 3))
    phis = np.zeros(nn)
    for ii in range(nn):
        if original[ii][0] < 0:
            phi = np.arctan(original[ii][1] / original[ii][0]) + np.pi
        else:
            phi = np.arctan(original[ii][1] / original[ii][0])
        new[ii] = np.array([np.linalg.norm(original[ii][:2]), phi, -original[ii][2]])
    return new


def cartesian(radial, pos_cyl):
    '''
    convert radial displacement into cartesian x, y
    '''
    nn = radial.shape[0]
    
    x = np.zeros(radial.shape)
    y = np.zeros(radial.shape)
    for ii in range(nn):
        x[ii] = radial[ii] * np.cos(pos_cyl[ii,1])
        y[ii] = radial[ii] * np.sin(pos_cyl[ii,1])
    return x, y

def radial(Dx, Dy, Dz, pos):

    nn = Dx.shape[0]
   
    Dr = np.zeros(Dx.shape)
    Dp = np.zeros(Dy.shape)

    for ii in range(nn):
        Dr[ii], Dp[ii] = np.real(hp.cartesian_to_cylindrical(Dx[ii], Dy[ii], Dz[ii], pos[ii]))[:2]

    return Dr, Dp, Dz

def synthetic_general(pressure, force_history, time_orig, stationPos, stations, chamberParams, pistonParams, mediumParams, deriv='VEL', comp='NO', coord='CARTESIAN'):
    '''
    calculates the full point source synthetic seismograms for a piston-chamber system at given station positions
    using either numerical halfspace Green's functions calculated using Zhu fk code (DEFAULT) or load in other GFs

    NOTE: assumes just vertical force

    NB: all position vectors must be given in (x, y, z) and in units of m
        +x : east
        +y : north
        +z : upwards
    
        () indicate numpy arrays
        [] indicate lists

    ---INPUTS---
    pressure      : (# time points) : chamber pressure history
    force_history : (# time points) : single force history (piston + magma inertia)
    time_orig     : (# time points) : time array (assumes equal time-stepping)
    stationPos    : (# stations, 3) : positions of accelerometer/seismometer stations
    stations      : [# stations]    : station labels which will be used to load relevant Green's functions
    chamberParams : [1, (3)]        : [chamber volume (m^3), centroid position vector]
    pistonParams  : [1, 1, (3)]     : [(piston height (m), piston radius (m)),
                                   piston source position vector]
    mediumParams  : [2]             : [shear modulus (Pa), rock density (kg/m^3)] (assumes Poisson ratio = 1/4)
    deriv         : string          : seismogram time derivative to return
                                  (options: 'ACC' acceleration; 'VEL' velocity; 'DIS' displacement)
    comp          : string          : option to return moment and single force contributions (only work when output is velocity seismogram at cylindrical coordinates)
    coord         : string          : coordinate system for the synthetic seimograms
                                  (options: 'CARTESIAN' and 'CYLINDRICAL')
    mt_gf_file    : string          : path to directory where MT Green's functions are stored
    sf_gf_file    : string          : path to directory where SF Green's functions are stored
    INTERPOLATE   : bool            : if True, will interpolate loaded Green's functions
    SAVE          : bool            : if True, will save final Green's functions in savefile directory
    mt_savefile   : string          : path to directory where final MT Green's functions are saved
    sf_savefile   : string          : path to directory where final SF Green's functions are saved
    ---RETURNS---
    seismo_x, seismo_y, seismo_z : (# stations, # time points) : chosen deriv applied to synthetic seismograms
    '''

    chamber_vol, chamber_centOLD, chamber_aspc = chamberParams
    piston_height, piston_radius, piston_posOLD = pistonParams

    mu, rho_rock = mediumParams

    dt = time_orig[2]-time_orig[1]
    
    shear_area = 2 * np.pi * piston_radius * piston_height
    cross_section = np.pi * piston_radius**2

    # transforming to origin over chamber centroid (at surface)
    chamber_origin = np.array([chamber_centOLD[0], chamber_centOLD[1], 0])

    #stationPos = np.array([[-2783.5, -12842.5, 0],
    #                       [29042.1, 2385.3, 0],
    #                       [-11025.4, 8958.1, 0]])
    pos, chamber_cent, piston_cent = translate(chamber_origin, stationPos, chamber_centOLD, piston_posOLD)

    # converting position vectors into cylindrical coordinates (to use in SW calc)
    # +z downwards (just for SW calculations)
    pos_cyl = cylindrical(pos)
    
    # setting up moment source
    shift = 0
    
    ra = (chamber_aspc**2 * chamber_vol * 3 / (4*np.pi))**(1/3) # semi-major axis length
    rb = ra/chamber_aspc                                         # semi-minor axis length

    moment_tensor = spmt.sphr_MT(rb, rb, ra, pressure, [0.25, mu])

    moment_tensor = hp.diag(moment_tensor)
    
    general_MT = ss.cushioned_general_MT(moment_tensor, cushion=shift)
    gc.collect()
    general_MT_rate = np.gradient(general_MT, dt, axis=-1)

    gc.collect()

    NN = np.ma.size(general_MT, axis=2)

    general_MT_hat = np.fft.fft(general_MT_rate, axis=2) * dt
    ind = np.argwhere(time_orig == 0)[0,0]
    nn = len(stations)
    time = time_orig #dt * np.arange(NN)

    tt = len(time)

    colors = ['#F0E442', '#E69F00', '#56B4E9', '#009E73', '#000000']

    z_mom = np.zeros((nn,tt), dtype='complex')
    r_mom = np.zeros((nn,tt), dtype='complex')
    tr_mom = np.zeros((nn,tt), dtype='complex')

    gfs_mom = load_all_gfs.gfs_mom
    gfs_sf = load_all_gfs.gfs_sf

    for stat, ii in zip(stations, np.arange(nn)):
        gf_time, gfs = gfs_mom[stat]
        gfs_hat = []
        
        for gg in gfs:
            gf_hat = np.fft.fft(gg, axis=0) * dt
            gfs_hat.append(gf_hat)
        # ['Mxx.txt', '2Mxy.txt', '2Mxz.txt', 'Myy.txt', '2Myz.txt', 'Mzz.txt']
        #plt.plot(np.fft.ifft(general_MT_hat[0,0] * gfs_hat[0][:,0], axis=-1) / dt, color=colors[ii], label=stat)
        #plt.plot(np.fft.ifft(general_MT_hat[1,1] * gfs_hat[3][:,0], axis=-1) / dt, color=colors[ii], linestyle='--')
        #plt.plot(np.fft.ifft(general_MT_hat[2,2] * gfs_hat[5][:,0], axis=-1) / dt, color=colors[ii], linestyle=':')
        z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,0] * gfs_hat[0][:,0], axis=-1) / dt, x=time, initial=0)
        z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,1] * gfs_hat[1][:,0], axis=-1) / dt, x=time, initial=0)
        z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,2] * gfs_hat[2][:,0], axis=-1) / dt, x=time, initial=0)
        z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[1,1] * gfs_hat[3][:,0], axis=-1) / dt, x=time, initial=0)
        z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[1,2] * gfs_hat[4][:,0], axis=-1) / dt, x=time, initial=0)
        z_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[2,2] * gfs_hat[5][:,0], axis=-1) / dt, x=time, initial=0)
        gc.collect()

        r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,0] * gfs_hat[0][:,1], axis=-1) / dt, x=time, initial=0)
        r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,1] * gfs_hat[1][:,1], axis=-1) / dt, x=time, initial=0)
        r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,2] * gfs_hat[2][:,1], axis=-1) / dt, x=time, initial=0)
        r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[1,1] * gfs_hat[3][:,1], axis=-1) / dt, x=time, initial=0)
        r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[1,2] * gfs_hat[4][:,1], axis=-1) / dt, x=time, initial=0)
        r_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[2,2] * gfs_hat[5][:,1], axis=-1) / dt, x=time, initial=0)
        gc.collect()

        tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,0] * gfs_hat[0][:,2], axis=-1) / dt, x=time, initial=0)
        tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,1] * gfs_hat[1][:,2], axis=-1) / dt, x=time, initial=0)
        tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[0,2] * gfs_hat[2][:,2], axis=-1) / dt, x=time, initial=0)
        tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[1,1] * gfs_hat[3][:,2], axis=-1) / dt, x=time, initial=0)
        tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[1,2] * gfs_hat[4][:,2], axis=-1) / dt, x=time, initial=0)
        tr_mom[ii] += si.cumtrapz(np.fft.ifft(general_MT_hat[2,2] * gfs_hat[5][:,2], axis=-1) / dt, x=time, initial=0)

    #plt.plot(time, r_mom[0], label=stat)
    #plt.legend()
    #plt.show()
    z_mom = np.real(z_mom)
    r_mom = np.real(r_mom)
    tr_mom = np.real(tr_mom)
    
    x_mom, y_mom = cartesian(r_mom, pos_cyl)

    #XXX NEED TO ADD IN CONVERSION OF TRANSVERSE CONTRIBUTIONS
    
    seismo_x = x_mom.copy()
    seismo_y = y_mom.copy()
    seismo_z = z_mom.copy()
        
    # calculating force contributions
    #np.savetxt('force_history_no_magma.txt', (time, force_history)) 
    force = ss.moment_density(np.array([force_history]), 1, cushion=shift)[0]

    gc.collect()
    
    force_rate = np.gradient(force, dt, axis=-1)
    gc.collect() 
    force_hat = np.fft.fft(force_rate) * dt
        
    z_for = np.zeros((nn,tt), dtype='complex')
    r_for = np.zeros((nn,tt), dtype='complex')
    tr_for = np.zeros((nn,tt), dtype='complex')
    for stat, ii in zip(stations, np.arange(nn)):
        gf_time, gfs = gfs_sf[stat]
        gfs_hat = []
        
        for gg in gfs:

            gf_hat = np.fft.fft(gg, axis=0) * dt
            gfs_hat.append(gf_hat)
        # ['horizontal_force.txt', 'vertical_force.txt']
        #plt.plot(np.fft.ifft(force_hat * gfs_hat[1][:,0], axis=-1) / dt, color=colors[ii], label=stat)
        #plt.plot(np.fft.ifft(force_hat * gfs_hat[1][:,1], axis=-1) / dt, color=colors[ii], linestyle='--')
        #plt.plot(np.fft.ifft(force_hat * gfs_hat[1][:,2], axis=-1) / dt, color=colors[ii], linestyle=':')
        z_for[ii] += si.cumtrapz(np.fft.ifft(force_hat * gfs_hat[1][:,0], axis=-1) / dt, x=time, initial=0)
        r_for[ii] += si.cumtrapz(np.fft.ifft(force_hat * gfs_hat[1][:,1], axis=-1) / dt, x=time, initial=0)
        tr_for[ii] += si.cumtrapz(np.fft.ifft(force_hat * gfs_hat[1][:,2], axis=-1) / dt, x=time, initial=0)
        gc.collect()
        #plt.plot(z_for[ii])
    #plt.show()

    z_for = np.real(z_for)
    r_for = np.real(r_for)
    tr_for = np.real(tr_for)

    x_for, y_for = cartesian(r_for, pos_cyl)
    #XXX NEED TO ADD IN CONVERSION OF TRANSVERSE CONTRIBUTIONS
    
    seismo_x += x_for
    gc.collect()
    seismo_y += y_for
    gc.collect()
    seismo_z += z_for
    gc.collect()

    if deriv == 'ACC':
        A_seismo_x = np.gradient(np.gradient(seismo_x[:,shift:NN], dt, axis=1), dt, axis=1)
        A_seismo_y = np.gradient(np.gradient(seismo_y[:,shift:NN], dt, axis=1), dt, axis=1)
        A_seismo_z = np.gradient(np.gradient(seismo_z[:,shift:NN], dt, axis=1), dt, axis=1)

        if coord == 'CYLINDRICAL':
            A_seismo_r = radial(A_seismo_x, A_seismo_y, A_seismo_z, pos)[0]
            return A_seismo_r, A_seismo_z, moment_tensor, force_history
        else:
            return A_seismo_x, A_seismo_y, A_seismo_z, moment_tensor, force_history
            
    elif deriv == 'VEL':
        V_seismo_x = np.gradient(seismo_x[:,shift:NN], dt, axis=1)
        V_seismo_y = np.gradient(seismo_y[:,shift:NN], dt, axis=1)
        V_seismo_z = np.gradient(seismo_z[:,shift:NN], dt, axis=1)

        if coord == 'CYLINDRICAL':
            if comp== 'YES':

                V_mom_x = np.gradient(x_mom[:,shift:NN], dt, axis=1)
                V_mom_y = np.gradient(y_mom[:,shift:NN], dt, axis=1)
                V_mom_z = np.gradient(z_mom[:,shift:NN], dt, axis=1)

                V_for_x = np.gradient(x_for[:,shift:NN], dt, axis=1)
                V_for_y = np.gradient(y_for[:,shift:NN], dt, axis=1)
                V_for_z = np.gradient(z_for[:,shift:NN], dt, axis=1)

                V_mom_r = radial(V_mom_x, V_mom_y, V_mom_z, pos)[0]
                V_for_r = radial(V_for_x, V_for_y, V_for_z, pos)[0]
                
                return V_mom_r, V_mom_z, V_for_r, V_for_z

            else:
                V_seismo_r = radial(V_seismo_x, V_seismo_y, V_seismo_z, pos)[0]
                return V_seismo_r, V_seismo_z, moment_tensor, force_history
        else:
            return V_seismo_x, V_seismo_y, V_seismo_z, moment_tensor, force_history
    else:
        seismo_x = seismo_x[:,shift:NN]
        seismo_y = seismo_y[:,shift:NN]
        seismo_z = seismo_z[:,shift:NN]

        if coord == 'CYLINDRICAL':
            #seismo_r = r_for + r_mom 
            seismo_r = radial(seismo_x, seismo_y, seismo_z, pos)[0]
            #plt.plot(time, seismo_r[0])
            #plt.show()
            return seismo_r, seismo_z
        else:
            return seismo_x, seismo_y, seismo_z

def synthetic_mixed_analytical(pressure, shear_force, dt, stationPos, chamberParams, pistonParams, mediumParams, deriv='ACC', comp='NO', coord='CARTESIAN'):
    '''
    calculates the full point source synthetic seismograms for a piston-chamber system at given station positions
    using closed-form analytical expressions for whole-space body waves and half-space Rayleigh surface waves 
    from Aki&Richards (2002)

    NB: all position vectors must be given in (x, y, z) and in units of m
        +x : east
        +y : north
        +z : upwards
    
    () indicate numpy arrays
    [] indicate lists

    ---INPUTS---
    pressure      : (# time points) : chamber pressure history
    shear_force   : (# time points) : shear force history (between piston and wall)
    dt            : (dt)            : time step size (assumes equal time-stepping)
    stationPos    : (# stations, 3) : positions of accelerometer/seismometer stations
    chamberParams : [1, (3)]        : [chamber volume (m^3), centroid position vector]
    pistonParams  : [1, 1, (3)]     : [(piston height (m), piston radius (m)),
                                   piston source position vector]
    mediumParams  : [2]             : [shear modulus (Pa), rock density (kg/m^3)] (assumes Poisson ratio = 1/4)

    deriv         : string          : seismogram time derivative to return
                                  (options: 'ACC' acceleration; 'VEL' velocity; 'DIS' displacement)
    comp          : string          : whether to return all components of the seismogram - body and surface wave contributions
                                  (options: 'YES' return all contributions; 'NO' return the total seismogram)
    coord         : string          : coordinate system for the synthetic seimograms
                                  (options: 'CARTESIAN' and 'CYLINDRICAL')
    ---RETURNS---
    seismo_x, seismo_y, seismo_z : (# stations, # time points) : chosen deriv applied to synthetic seismograms
    '''
    chamber_vol, chamber_centOLD, chamber_aspc = chamberParams
    piston_height, piston_radius, piston_posOLD = pistonParams
    mu, rho_rock = mediumParams
    lame = mu # poisson ratio = 1/4
    
    shear_area = 2 * np.pi * piston_radius * piston_height
    cross_section = np.pi * piston_radius**2
    
    # transforming to origin over chamber centroid (at surface)
    chamber_origin = np.array([chamber_centOLD[0], chamber_centOLD[1], 0])
    pos, chamber_cent, piston_cent = translate(chamber_origin, stationPos, chamber_centOLD, piston_posOLD)
    
    # converting position vectors into cylindrical coordinates (to use in SW calc)
    # +z downwards (just for SW calculations)
    pos_cyl = cylindrical(pos)
    
    # storing coordinates for separation vectors between point-source and seismometers
    # separation: (# receivers, # sources)
    # gamma: (# receivers, # sources, 3)
    separationCH, gammaCH = hp.separation_distances_vectors(pos, [chamber_cent])
    separationPI, gammaPI = hp.separation_distances_vectors(pos, [piston_cent])
    gc.collect()
    
    # setting up moment source
    shift = 4000
    
    '''
    moment = ss.moment_density(np.array([pressure]), 0.75 * chamber_vol, cushion=shift)[0]
    moment_tensor = np.zeros((3, 3, len(pressure)))

    for ii in range(3):
        moment_tensor[ii, ii] += ((lame + 2*mu) / mu) * 0.75 * chamber_vol * pressure
    '''
    
    ra = int((chamber_aspc**2 * chamber_vol * 3 / (4*np.pi))**(1/3)) # semi-major axis length
    rb = int(ra/chamber_aspc)                                     # semi-minor axis length

    moment_tensor = spmt.sphr_MT(rb, rb, ra, pressure, [0.25, mu])
    moment_tensor = hp.diag(moment_tensor)

    general_MT = ss.cushioned_general_MT(moment_tensor, cushion=shift)
    gc.collect()
    
    # calculating moment contributions
    x_bwm, y_bwm, z_bwm = bw.displacement_moment([general_MT], separationCH, gammaCH, dt, [rho_rock, lame, mu])

    seismo_x = x_bwm.copy()
    seismo_y = y_bwm.copy()
    seismo_z = z_bwm.copy()
        
    r_mom, z_mom = sw.rayleigh_displacement_moment([general_MT], pos_cyl, np.array([-chamber_cent[2]]), dt, [rho_rock, lame, mu])
        
    seismo_z += z_mom
    gc.collect()
    x_mom, y_mom = cartesian(r_mom, pos_cyl)
    gc.collect()
    seismo_x += x_mom
    gc.collect()
    seismo_y += y_mom
    gc.collect()
        
    
    # calculating force contributions
    force_history = (shear_area * shear_force) - (cross_section * pressure)

    force = ss.moment_density(np.array([force_history]), 1, cushion=shift)[0]
    gc.collect()
        
    x_bwf, y_bwf, z_bwf = bw.displacement_force([force], separationPI, gammaPI, dt, [rho_rock, lame, mu])
    seismo_x += x_bwf
    gc.collect()
    seismo_y += y_bwf
    gc.collect()
    seismo_z += z_bwf
    gc.collect()
        
    r_swf, z_swf = sw.rayleigh_displacement_force([-force], pos_cyl, np.array([-piston_cent[2]]),
                                                      dt, [rho_rock, lame, mu])
    seismo_z += z_swf
    gc.collect()
    x_swf, y_swf = cartesian(r_swf, pos_cyl)
    seismo_x += x_swf
    gc.collect()
    seismo_y += y_swf
    gc.collect()
    
        
    if deriv == 'ACC':
        A_seismo_x = np.gradient(np.gradient(seismo_x[:,0,shift:], dt, axis=1), dt, axis=1)
        A_seismo_y = np.gradient(np.gradient(seismo_y[:,0,shift:], dt, axis=1), dt, axis=1)
        A_seismo_z = np.gradient(np.gradient(seismo_z[:,0,shift:], dt, axis=1), dt, axis=1)

        if comp == 'YES':
            Ax_bwm = np.gradient(np.gradient(x_bwm[:,0,shift:], dt, axis=1), dt, axis=1)
            Ax_bwf = np.gradient(np.gradient(x_bwf[:,0,shift:], dt, axis=1), dt, axis=1)
            Ax_mom = np.gradient(np.gradient(x_mom[:,0,shift:], dt, axis=1), dt, axis=1)
            Ax_swf = np.gradient(np.gradient(x_swf[:,0,shift:], dt, axis=1), dt, axis=1)

            Ay_bwm = np.gradient(np.gradient(y_bwm[:,0,shift:], dt, axis=1), dt, axis=1)
            Ay_bwf = np.gradient(np.gradient(y_bwf[:,0,shift:], dt, axis=1), dt, axis=1)
            Ay_mom = np.gradient(np.gradient(y_mom[:,0,shift:], dt, axis=1), dt, axis=1)
            Ay_swf = np.gradient(np.gradient(y_swf[:,0,shift:], dt, axis=1), dt, axis=1)

            Az_bwm = np.gradient(np.gradient(z_bwm[:,0,shift:], dt, axis=1), dt, axis=1)
            Az_bwf = np.gradient(np.gradient(z_bwf[:,0,shift:], dt, axis=1), dt, axis=1)
            Az_mom = np.gradient(np.gradient(z_mom[:,0,shift:], dt, axis=1), dt, axis=1)
            Az_swf = np.gradient(np.gradient(z_swf[:,0,shift:], dt, axis=1), dt, axis=1)

            if coord == 'CYLINDRICAL':
                
                A_seismo_r = radial(A_seismo_x, A_seismo_y, A_seismo_z, pos)[0]
                    
                Ar_bwm = radial(Ax_bwm, Ay_bwm, Az_bwm, pos)[0]
                Ar_bwf = radial(Ax_bwf, Ay_bwf, Az_bwf, pos)[0]
                Ar_mom = radial(Ax_mom, Ay_mom, Az_mom, pos)[0]
                Ar_swf = radial(Ax_swf, Ay_swf, Az_swf, pos)[0]

                return (A_seismo_r, Ar_bwm, Ar_bwf, Ar_mom, Ar_swf), (A_seismo_z, Az_bwm, Az_bwf, Az_mom, Az_swf)
            else:

                return (A_seismo_x, Ax_bwm, Ax_bwf, Ax_mom, Ax_swf), (A_seismo_y, Ay_bwm, Ay_bwf, Ay_mom, Ay_swf), (A_seismo_z, Az_bwm, Az_bwf, Az_mom, Az_swf)
        else:
            if coord == 'CYLINDRICAL':
                A_seismo_r = radial(A_seismo_x, A_seismo_y, A_seismo_z, pos)[0]
                return A_seismo_r, A_seismo_z
            else:
                return A_seismo_x, A_seismo_y, A_seismo_z
            
        return np.gradient(seismo_x[:,0,shift:], dt, axis=1), np.gradient(seismo_y[:,0,shift:], dt, axis=1), np.gradient(seismo_z[:,0,shift:], dt, axis=1)
    elif deriv == 'VEL':
        V_seismo_x = np.gradient(seismo_x[:,0,shift:], dt, axis=1)
        V_seismo_y = np.gradient(seismo_y[:,0,shift:], dt, axis=1)
        V_seismo_z = np.gradient(seismo_z[:,0,shift:], dt, axis=1)

        if comp == 'YES':
            Vx_bwm = np.gradient(x_bwm[:,0,shift:], dt, axis=1)
            Vx_bwf = np.gradient(x_bwf[:,0,shift:], dt, axis=1)
            Vx_mom = np.gradient(x_mom[:,0,shift:], dt, axis=1)
            Vx_swf = np.gradient(x_swf[:,0,shift:], dt, axis=1)

            Vy_bwm = np.gradient(y_bwm[:,0,shift:], dt, axis=1)
            Vy_bwf = np.gradient(y_bwf[:,0,shift:], dt, axis=1)
            Vy_mom = np.gradient(y_mom[:,0,shift:], dt, axis=1)
            Vy_swf = np.gradient(y_swf[:,0,shift:], dt, axis=1)

            Vz_bwm = np.gradient(z_bwm[:,0,shift:], dt, axis=1)
            Vz_bwf = np.gradient(z_bwf[:,0,shift:], dt, axis=1)
            Vz_mom = np.gradient(z_mom[:,0,shift:], dt, axis=1)
            Vz_swf = np.gradient(z_swf[:,0,shift:], dt, axis=1)

            if coord == 'CYLINDRICAL':
                V_seismo_r = radial(V_seismo_x, V_seismo_y, V_seismo_z, pos)[0]
                    
                Vr_bwm = radial(Vx_bwm, Vy_bwm, Vz_bwm, pos)[0]
                Vr_bwf = radial(Vx_bwf, Vy_bwf, Vz_bwf, pos)[0]
                Vr_mom = radial(Vx_mom, Vy_mom, Vz_mom, pos)[0]
                Vr_swf = radial(Vx_swf, Vy_swf, Vz_swf, pos)[0]
                return (V_seismo_r, Vr_bwm, Vr_bwf, Vr_mom, Vr_swf), (V_seismo_z, Vz_bwm, Vz_bwf, Vz_mom, Vz_swf)
            else:
                return (V_seismo_x, Vx_bwm, Vx_bwf, Vx_mom, Vx_swf), (V_seismo_y, Vy_bwm, Vy_bwf, Vy_mom, Vy_swf), (V_seismo_z, Vz_bwm, Vz_bwf, Vz_mom, Vz_swf)
        else:
            if coord == 'CYLINDRICAL':
                V_seismo_r = radial(V_seismo_x, V_seismo_y, V_seismo_z, pos)[0]
                return V_seismo_r, V_seismo_z, moment_tensor, force_history
            else:
                return V_seismo_x, V_seismo_y, V_seismo_z
    else:
        seismo_x = seismo_x[:,0,shift:]
        seismo_y = seismo_y[:,0,shift:]
        seismo_z = seismo_z[:,0,shift:]

        if comp == 'YES':
            x_bwm = x_bwm[:,0,shift:]
            x_bwf = x_bwf[:,0,shift:]
            x_mom = x_mom[:,0,shift:]
            x_swf = x_swf[:,0,shift:]

            y_bwm = y_bwm[:,0,shift:]
            y_bwf = y_bwf[:,0,shift:]
            y_mom = y_mom[:,0,shift:]
            y_swf = y_swf[:,0,shift:]

            z_bwm = z_bwm[:,0,shift:]
            z_bwf = z_bwf[:,0,shift:]
            z_mom = z_mom[:,0,shift:]
            z_swf = z_swf[:,0,shift:]


            if coord == 'CYLINDRICAL':
                seismo_r = radial(seismo_x, seismo_y, seismo_z, pos)[0]

                r_bwm = radial(x_bwm, y_bwm, z_bwm, pos)[0]
                r_bwf = radial(x_bwf, y_bwf, z_bwf, pos)[0]
                r_mom = radial(x_mom, y_mom, z_mom, pos)[0]
                r_swf = radial(x_swf, y_swf, z_swf, pos)[0]

                return (seismo_r, r_bwm, r_bwf, r_mom, r_swf), (seismo_z, z_bwm, z_bwf, z_mom, z_swf)
            else:
                return (seismo_x, x_bwm, x_bwf,  x_mom, x_swf), (seismo_y, y_bwm, y_bwf, y_mom, y_swf), (seismo_z, z_bwm, z_bwf, z_mom, z_swf)
        else:
            if coord == 'CYLINDRICAL':

                seismo_r = radial(seismo_x, seismo_y, seismo_z, pos)[0]

                return seismo_r, seismo_z
            else:
                return seismo_x, seismo_y, seismo_z
            



    # In[ ]:




