import numpy as np
from ctypes import *
import pandas as pd

#     SUBROUTINE evolv2_global(z,zpars,acclim,alphain,qHG,qGB,kick_in)


libc = cdll.LoadLibrary("/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/src/COSMIC/cosmic/src/evolv2_bhms.so")
np.set_printoptions(suppress=True)

def evolv2(m1, m2, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, qGB, logZ):
    tb = 10**logtb
    metallicity = 10**logZ
    sigma = 0

    z = byref(c_double(metallicity))
    zpars = np.zeros(20).ctypes.data_as(POINTER(c_double))
    qkstar2 = byref(c_double(qHG))
    sigma = byref(c_double(0.0))
    qkstar3 = byref(c_double(qGB))
    natal_kick = np.zeros((2,5))
    natal_kick[0,0] = vk1
    natal_kick[0,1] = phi1
    natal_kick[0,2] = theta1
    natal_kick[0,3] = omega1
    natal_kick[0,4] = 3
    natal_kick[1,0] = vk2
    natal_kick[1,1] = phi2
    natal_kick[1,2] = theta2
    natal_kick[1,3] = omega2
    natal_kick[1,4] = 3
    natal_kick = natal_kick.T.flatten().ctypes.data_as(POINTER(c_double))
    alpha = np.zeros(2)
    alpha[0] = alpha_1
    alpha[1] = alpha_2
    acc_lim = np.zeros(2)
    acc_lim[0] = acc_lim_1
    acc_lim[1] = acc_lim_2
    acc_lim = acc_lim.ctypes.data_as(POINTER(c_double))
    alpha = alpha.flatten().ctypes.data_as(POINTER(c_double))
    libc.evolv2_global_(z,zpars,acc_lim,alpha,qkstar2,qkstar3,natal_kick)

    mass = np.array([m1,m2]).ctypes.data_as(POINTER(c_double))
    mass0 = np.array([m1,m2]).ctypes.data_as(POINTER(c_double))
    epoch = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    ospin = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    tb = byref(c_double(tb))
    ecc = byref(c_double(e))
    tphysf = byref(c_double(13700.0))
    dtp = byref(c_double(0.0))
    rad = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    lumin = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    massc = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    radc = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    menv = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    renv = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    B_0 = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    bacc = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    tacc = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    tms = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    bhspin = np.array([0.0,0.0]).ctypes.data_as(POINTER(c_double))
    tphys = byref(c_double(0.0))
    bkick = np.zeros(20).ctypes.data_as(POINTER(c_double))
    kick_info = np.zeros(34).ctypes.data_as(POINTER(c_double)) # Fortran treat n-D array differently than numpy
    bpp_index_out = byref(c_int64(0))
    bcm_index_out = byref(c_int64(0))
    kick_info_out = np.zeros(34).ctypes.data_as(POINTER(c_double))
    t_merge = byref(c_double(0.0))
    m_merge = np.array([0.0,0.0])
    bpp_out=np.zeros([1000,43]).flatten().ctypes.data_as(POINTER(c_double))
    kstar = np.array([1,1]).ctypes.data_as(POINTER(c_double))
    libc.evolv2_(kstar,mass,tb,ecc,z,tphysf,
    dtp,mass0,rad,lumin,massc,radc,
    menv,renv,ospin,B_0,bacc,tacc,epoch,tms,
    bhspin,tphys,zpars,bkick,kick_info,
    bpp_index_out,bcm_index_out,kick_info_out,
    t_merge,m_merge.ctypes.data_as(POINTER(c_double)),bpp_out)
    
    bpp = bpp_out._arr.reshape(43,1000)[:,0:bpp_index_out._obj.value].T
    
    bpp = pd.DataFrame(bpp,
                       columns=['tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2',
                                'sep', 'porb', 'ecc', 'RRLO_1', 'RRLO_2', 'evol_type',
                                'aj_1', 'aj_2', 'tms_1', 'tms_2', 'massc_1', 'massc_2',
                                'rad_1', 'rad_2', 'mass0_1', 'mass0_2', 'lum_1', 'lum_2',
                                'teff_1', 'teff_2', 'radc_1', 'radc_2', 'menv_1', 'menv_2',
                                'renv_1', 'renv_2', 'omega_spin_1', 'omega_spin_2', 'B_1',
                                'B_2', 'bacc_1', 'bacc_2', 'tacc_1', 'tacc_2', 'epoch_1',
                                'epoch_2', 'bhspin_1', 'bhspin_2'])
    
    return t_merge._obj.value,np.sort(m_merge)[::-1],kick_info_out._arr.reshape(17,2).T,bpp

def evolv2_same_alphas(m1, m2, logtb, e, alpha, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, qGB, logZ):
    alpha_1 = alpha
    alpha_2 = alpha
    return evolv2(m1, m2, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, qGB, logZ)

def evolv2_fixed_kicks(m1, m2, logtb, e, alpha_1, alpha_2, acc_lim_1, acc_lim_2, qHG, qGB, logZ):
    vk1 = 0.0
    theta1 = 0.0
    phi1 = 0.0
    omega1 = 0.0
    vk2 = 0.0
    theta2 = 0.0
    phi2 = 0.0
    omega2 = 0.0
    return evolv2(m1, m2, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, qGB, logZ)

def evolv2_fixed_kicks_same_alphas(m1, m2, logtb, e, alpha, acc_lim_1, acc_lim_2, qHG, qGB, logZ):
    alpha_1 = alpha
    alpha_2 = alpha
    return evolv2_fixed_kicks(m1, m2, logtb, e, alpha_1, alpha_2, acc_lim_1, acc_lim_2, qHG, qGB, logZ)

def evolv2_lowmass_secondary(m1, m2, logtb, e, alpha_1, alpha_2, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, qGB, logZ):
    vk1 = 0.0
    theta1 = 0.0
    phi1 = 0.0
    omega1 = 0.0
    return evolv2(m1, m2, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, qGB, logZ)

def evolv2_lowmass_secondary_new(m1, m2, logtb, e, alpha_1, alpha_2, vk1, vk2, theta2, phi2, acc_lim_1, acc_lim_2, qHG, qGB, logZ):
    theta1 = 0.0
    phi1 = 0.0
    omega1 = 0.0
    omega2 = 0.0
    return evolv2(m1, m2, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, qGB, logZ)

def evolv2_lowmass_secondary_new_fixed_alphas(m1, m2, logtb, e, vk1, vk2, theta2, phi2, acc_lim_1, acc_lim_2, qHG, qGB, logZ):
    alpha_1 = 1.0
    alpha_2 = 1.0
    theta1 = 0.0
    phi1 = 0.0
    omega1 = 0.0
    omega2 = 0.0
    return evolv2(m1, m2, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, qGB, logZ)

def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

