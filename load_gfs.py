import numpy as np
import gc
import matplotlib.pyplot as plt
import scipy.interpolate as si

def load_gfs(directory, srctype, time, INTERPOLATE=False, SAVE=False, save_file='gf', PLOT=False):
    '''
    loads in Green's functions and can interpolate in time to get compatible
    array dimensions with desired time array
    can also save the Green's functions to a new file 
    
    NB: if don't want to interpolate each time, can save the resulting interpolated 
        Green's functions and just load them in next time without alteration
    
    original Green's functions must be stored such that the columns correspond below:
            time    vertical    radial   transverse

    --INPUTS--
    directory   : string            : path to folder holding Green's function files
    srctype     : (1)               : source type (0: moment tensor, 1: single force)
    time        : (# time points)   : desired time array
    INTERPOLATE : bool              : if True, interpolate to get values at desired time
    SAVE        : bool              : if True, saves Green's functions to save_file
    save_file   : string            : path to where Green's functions will be saved
    --RETURNS--
    gf_time     : (# time points)        : desired time array
    gfs         : [ (# time points, 3) ] : list of final Green's functions (ver, rad, tra)
                                            if single force, 2 arrays
                                            if moment tensor, 6 arrays
    '''
    if srctype == 0:
        components = ['Mxx.txt', '2Mxy.txt', '2Mxz.txt', 'Myy.txt', '2Myz.txt', 'Mzz.txt']
    elif srctype == 1:
        components = ['horizontal_force.txt', 'vertical_force.txt']
    colors = ['#F0E442', '#E69F00', '#56B4E9', '#009E73', '#000000', '#E50000']
    
    tt = len(time)
    dt = time[2] - time[1]
    desired_omega = np.fft.fftfreq(tt, dt) * (2 * np.pi)
    ind = np.argwhere(desired_omega < 0)[0,0]
    sorted_desired_omega = np.concatenate((desired_omega[ind:], desired_omega[:ind]))

    
    gfs = []
    gfs_hat = []
    gf_time = np.loadtxt(directory+components[0], usecols = 0)
    gf_dt = gf_time[1] - gf_time[0]
    for com in components:
        gf = np.loadtxt(directory+com, usecols = (1,2,3))
        gfs.append(gf)
        gfs_hat.append(np.fft.fft(gf, axis=0) * gf_dt)
    gc.collect()

    gf_omega = np.fft.fftfreq(len(gf_time), gf_dt) * (2 * np.pi)


    gf_ind = np.argwhere(gf_omega < 0)[0,0]
    sorted_gf_omega = np.concatenate((gf_omega[gf_ind:], gf_omega[:gf_ind]))
    
    if PLOT:
        for func, lab, col in zip(gfs_hat, components, colors):
            plt.plot(gf_omega, np.abs(func[:,1]), color=col, label=lab)
            #plt.plot(gf_time, func[:,1], color=col, label=lab)
        #plt.show()
    new_gfs = []
    for func, lab, col in zip(gfs_hat, components, colors):
        sorted_func = np.concatenate((func[gf_ind:], func[:gf_ind]))
        smooth = si.interp1d(sorted_gf_omega, sorted_func, axis=0, kind='cubic')
        sorted_gf_hat_sm = smooth(sorted_desired_omega)
        if PLOT:
            #plt.plot(sorted_gf_omega[:-1], np.abs(smooth(sorted_gf_omega[:-1])[:,1]), color=col)
            plt.plot(sorted_desired_omega, np.abs(sorted_gf_hat_sm[:,1]), '.', color=col)
        gf_hat_sm = np.concatenate((sorted_gf_hat_sm[-ind:], sorted_gf_hat_sm[:-ind]))
        new_gfs.append(np.fft.ifft(gf_hat_sm, axis=0) / dt)
    if PLOT:
        plt.legend()
        plt.show()

    return time, new_gfs
    '''
    if INTERPOLATE:
        new_gfs_endpad = []
        if time[-1] > gf_time[-1]:
            index = np.argwhere(time > gf_time[-1])[0,0]
            new_gf_time_endpad = np.concatenate((gf_time, time[index:]))
            for func in gfs:
                padding = np.tile(func[-1], (len(time[index:]), 1))
                new_gf = np.concatenate((func, padding))
                new_gfs_endpad.append(new_gf)
        else:
            new_gfs_endpad = gfs
            new_gf_time_endpad = gf_time
        new_gfs = []
        if time[0] < gf_time[0]:
            index = np.argwhere(time < gf_time[0])[-1,0]
            new_gf_time = np.concatenate((time[:index], new_gf_time_endpad))
            for func in new_gfs_endpad:
                padding = np.tile(func[0], (len(time[:index]), 1))
                new_gf = np.concatenate((padding, func))
                new_gfs.append(new_gf)
        else:
            new_gfs = new_gfs_endpad
            new_gf_time = new_gf_time_endpad
        gfs = new_gfs
        gf_time = new_gf_time
        gc.collect()
            
        zer = np.argwhere(time > 0)[0,0] - 1

        gfs_smoothed = []
        for func in new_gfs:
            smooth = si.interp1d(new_gf_time, func, axis=0, kind='cubic')
            gf_sm = smooth(time)
            gfs_smoothed.append(np.concatenate((gf_sm[zer:], gf_sm[:zer]), axis=0))
        gfs = gfs_smoothed
        dt = time[2]-time[1]
        gf_time = dt * np.arange(len(time))

    if SAVE:
        for func, lab in zip(gfs, components):
            extra = np.array([gf_time])
            combined = np.concatenate((np.array([gf_time]).transpose(), func), axis=1)
            np.savetxt(save_file+'interpolated_'+lab, combined, 
                    header="time, vertical, radial, transverse")

    if PLOT:
        for func, lab, col in zip(gfs, components, colors):
            omega = np.fft.fftfreq(len(func[:,1]), time[1] - time[0]) * (2 * np.pi)
            plt.plot(omega, np.abs(np.fft.fft(func[:,1])), color=col, linestyle=':')
            #plt.plot(gf_time, func[:,1], color=col, linestyle=':')
        plt.legend()
        plt.show()
    return gf_time, gfs
    '''

'''
time = np.linspace(-10, 60, 10000)


direc = '/Users/kcoppess/muspelheim/synthetic-seismograms/halfspace-greens/test/interpolated_'
#direc = '/Users/kcoppess/muspelheim/synthetic-seismograms/halfspace-greens/half_2.17729975_chamber/HMLE/'
#direc = '/Users/kcoppess/muspelheim/synthetic-seismograms/halfspace-greens/half_0.55654187_piston/HMLE/'
#time, gfs = load_gf(direc, 0, time, INTERPOLATE=True, SAVE=True, 
#        save_file='/Users/kcoppess/muspelheim/synthetic-seismograms/halfspace-greens/test/', PLOT=True)

time, gfs = load_gf(direc, 0, time, PLOT=True)
'''
