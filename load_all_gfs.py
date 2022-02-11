#!/usr/bin/env python
# coding: utf-8
# Set up globally available Green's functions
import os
import numpy as np
import load_gfs as gf

def init(mt_gf_file, sf_gf_file, time):
    # use global variables to reduce time spent passing variables among parallel processors, which makes parallel processing slower than serial
    global gfs_mom, gfs_sf
    # get path of current directory
    directory = os.getcwd()

    # load accelerometer+broad band+GPS Green's functions ----------------------------------------------------------------------------------------------
    acc_sta = ['HMLE', 'PAUD', 'RSDD']
    bd_sta = ['HLPD', 'MLOD', 'STCD']
    gps_sta = ['69FL', '92YN', 'AHUP', 'BDPK', 'BYRL', 'CNPK', 'CRIM', 'DEVL', 'OUTL', 'PUHI', 'PWRL', 'V120', 'VSAS']
    all_sta = acc_sta+bd_sta+gps_sta
    
    # Load gfs for moment tensor and single force into respective dictionaries
    gfs_mom = {}
    gfs_sf = {}
    for ii, stat in enumerate(all_sta):
        mom_gf_time, mom_gfs = gf.load_gfs(mt_gf_file+stat+'/', 0, time, INTERPOLATE=True, SAVE=False, save_file=mt_gf_file, PLOT=False)
        gfs_mom[stat] = [mom_gf_time, mom_gfs]

        sf_gf_time, sf_gfs = gf.load_gfs(sf_gf_file+stat+'/', 1, time, INTERPOLATE=True, SAVE=False, save_file=sf_gf_file, PLOT=False)
        gfs_sf[stat] = [sf_gf_time, sf_gfs]
        