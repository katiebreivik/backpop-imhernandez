import numpy as np
from ctypes import *
import pandas as pd

from scipy.stats import gaussian_kde

from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d

from astropy.cosmology import Planck15, FlatLambdaCDM, z_at_value
from astropy import units as u
import astropy.constants as constants
from pesummary.io import read
import arviz as az
#     SUBROUTINE evolv2_global(z,zpars,acclim,alphain,qHG,qGB,kick_in)

libc = cdll.LoadLibrary("/hildafs/home/magana/tmp_ondemand_hildafs_phy230014p_symlink/magana/src/COSMIC/cosmic/src/evolv2_bhms.so")
np.set_printoptions(suppress=True)

def evolv2(m1, q, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, qGB, logZ):
    m2 = m1*q
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

def evolv2_fixed_kicks(m1, q, logtb, e, alpha_1, alpha_2, acc_lim_1, acc_lim_2, qHG, qGB, logZ):
    vk1 = 0.0
    theta1 = 0.0
    phi1 = 0.0
    omega1 = 0.0
    vk2 = 0.0
    theta2 = 0.0
    phi2 = 0.0
    omega2 = 0.0
    return evolv2(m1, q, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, qGB, logZ)

def evolv2_fixed_kicks_minimal(m1, q, logtb, e, alpha, acc_lim, qHG, qGB, logZ):
    alpha_1 = alpha
    alpha_2 = alpha
    acc_lim_1 = acc_lim
    acc_lim_2 = acc_lim
    vk1 = 0.0
    theta1 = 0.0
    phi1 = 0.0
    omega1 = 0.0
    vk2 = 0.0
    theta2 = 0.0
    phi2 = 0.0
    omega2 = 0.0
    return evolv2(m1, q, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, qGB, logZ)

def evolv2_lowmass_secondary(m1, q, logtb, e, alpha_1, alpha_2, vk2, theta2, phi2, acc_lim_1, acc_lim_2, qHG, qGB, logZ):
    vk1 = 0.0
    theta1 = 0.0
    phi1 = 0.0
    omega1 = 0.0
    omega2 = 0.0
    return evolv2(m1, q, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, qGB, logZ)

def evolv2_lowmass_secondary_minimal(m1, q, logtb, e, alpha, vk2, theta2, phi2, acc_lim, qHG, qGB, logZ):
    vk1 = 0.0
    alpha_1 = alpha
    alpha_2 = alpha
    acc_lim_1 = acc_lim
    acc_lim_2 = acc_lim
    theta1 = 0.0
    phi1 = 0.0
    omega1 = 0.0
    omega2 = 0.0
    return evolv2(m1, q, logtb, e, alpha_1, alpha_2, vk1, theta1, phi1, omega1, vk2, theta2, phi2, omega2, acc_lim_1, acc_lim_2, qHG, qGB, logZ)

def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

m1lo = 1.0
tblo = 1.0
elo = 0.0
alphalo_1 = 0.1
alphalo_2 = 0.1
vklo = 0.0
thetalo = 0.0
philo = -90.0
omegalo = 0.0
acc_limlo_1 = 0.0
acc_limlo_2 = 0.0
qc_kstar2lo = 0.5
qc_kstar3lo = 0.5
Zlo = 0.0001

m1hi = 150.0
m2hi = 150.0
tbhi = 5000.0
ehi = 0.9
alphahi_1 = 20.0
alphahi_2 = 20.0
vkhi = 500.0
thetahi = 360.0
phihi = 90.0
omegahi = 360
acc_limhi_1 = 1.0
acc_limhi_2 = 1.0
qc_kstar2hi = 10.0
qc_kstar3hi = 10.0
Zhi = 0.03

qlo = 0.01
qhi = 1


labels_dict = {"backpop" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$e$',r'$\alpha_1$',r'$\alpha_2$',
                            r'$v_1$',r'$\theta_1$',r'$\phi_1$',r'$\omega_1$',r'$v_2$',r'$\theta_2$',
                            r'$\phi_2$',r'$\omega_2$',r'$f_{\rm lim,1}$', r'$f_{\rm lim,2}$',
                            r'$q_{\rm HG}$', r'$q_{\rm GB}$', r'$\log_{10}Z$'],

               "backpop_fixed_kicks" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$e$',r'$\alpha_1$',r'$\alpha_2$',
                                        r'$f_{\rm lim,1}$',r'$f_{\rm lim,2}$',r'$q_{\rm HG}$',r'$q_{\rm GB}$',r'$\log_{10}Z$'],
               
               "backpop_fixed_kicks_minimal" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$e$',r'$\alpha$',
                                                r'$f_{\rm lim}$',r'$q_{\rm HG}$',r'$q_{\rm GB}$',r'$\log_{10}Z$'],
               
               "backpop_lowmass_secondary" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$e$',r'$\alpha_1$',r'$\alpha_2$',
                                              r'$v_2$',r'$\theta_2$',r'$\phi_2$',r'$f_{\rm lim,1}$',
                                              r'$f_{\rm lim,2}$',r'$q_{\rm HG}$',r'$q_{\rm GB}$',r'$\log_{10}Z$'],

               "backpop_lowmass_secondary_minimal" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$e$',r'$\alpha$',
                                              r'$v_2$',r'$\theta_2$',r'$\phi_2$',r'$f_{\rm lim}$',
                                              r'$q_{\rm HG}$',r'$q_{\rm GB}$',r'$\log_{10}Z$']
              }

def get_backpop_config(config_name):
    if (config_name == "backpop"):
        lower_bound = np.array([m1lo, qlo, np.log10(tblo), elo, alphalo_1, alphalo_2, vklo, thetalo, philo, omegalo, vklo, thetalo, philo, omegalo, acc_limlo_1, acc_limlo_2, qc_kstar2lo, qc_kstar3lo, np.log10(Zlo)])
        upper_bound = np.array([m1hi, qhi, np.log10(tbhi), ehi, alphahi_1, alphahi_2, vkhi, thetahi, phihi, omegahi, vkhi, thetahi, phihi, omegahi, acc_limhi_1, acc_limhi_2, qc_kstar2hi, qc_kstar3hi, np.log10(Zhi)])

        evolution = evolv2

    if (config_name == "backpop_fixed_kicks"):
        lower_bound = np.array([m1lo, qlo, np.log10(tblo), elo, alphalo_1, alphalo_2, acc_limlo_1, acc_limlo_2, qc_kstar2lo, qc_kstar3lo, np.log10(Zlo)])
        upper_bound = np.array([m1hi, qhi, np.log10(tbhi), ehi, alphahi_1, alphahi_2, acc_limhi_1, acc_limhi_2, qc_kstar2hi, qc_kstar3hi, np.log10(Zhi)])

        evolution = evolv2_fixed_kicks
        
    if (config_name == "backpop_fixed_kicks_minimal"):
        lower_bound = np.array([m1lo, qlo, np.log10(tblo), elo, alphalo_1, acc_limlo_1, qc_kstar2lo, qc_kstar3lo, np.log10(Zlo)])
        upper_bound = np.array([m1hi, qhi, np.log10(tbhi), ehi, alphahi_1, acc_limhi_1, qc_kstar2hi, qc_kstar3hi, np.log10(Zhi)])

        evolution = evolv2_fixed_kicks_minimal

    if (config_name == "backpop_lowmass_secondary"):
        lower_bound = np.array([m1lo, qlo, np.log10(tblo), elo, alphalo_1, alphalo_2, vklo, thetalo, philo, acc_limlo_1, acc_limlo_2, qc_kstar2lo, qc_kstar3lo, np.log10(Zlo)])
        upper_bound = np.array([m1hi, qhi, np.log10(tbhi), ehi, alphahi_1, alphahi_2, vkhi, thetahi, phihi, acc_limhi_1, acc_limhi_2, qc_kstar2hi, qc_kstar3hi, np.log10(Zhi)])
        
        evolution = evolv2_lowmass_secondary
        
    if (config_name == "backpop_lowmass_secondary_minimal"):
        lower_bound = np.array([m1lo, qlo, np.log10(tblo), elo, alphalo_1, vklo, thetalo, philo, acc_limlo_1, qc_kstar2lo, qc_kstar3lo, np.log10(Zlo)])
        upper_bound = np.array([m1hi, qhi, np.log10(tbhi), ehi, alphahi_1, vkhi, thetahi, phihi, acc_limhi_1, qc_kstar2hi, qc_kstar3hi, np.log10(Zhi)])

        evolution = evolv2_lowmass_secondary_minimal
        
    return evolution, lower_bound, upper_bound


# Cosmo routines
Om0Planck = Planck15.Om0
H0Planck = Planck15.H0.value
speed_of_light = constants.c.to('km/s').value

zMax = 100
zgrid = np.expm1(np.linspace(np.log(1), np.log(zMax+1), 10000))
dLgrid = Planck15.luminosity_distance(zgrid).to('Mpc').value
tgrid = Planck15.lookback_time(zgrid).to('Myr').value

tofdL = interp1d(dLgrid,13700-tgrid,bounds_error=False,fill_value=1e100)
dLoft = interp1d(13700-tgrid,dLgrid,bounds_error=False,fill_value=1e100)
ddLdt = interp1d(tgrid,np.gradient(dLgrid,tgrid),bounds_error=False,fill_value=1e100)

zofdL = interp1d(dLgrid,zgrid)
dLofz = interp1d(zgrid,dLgrid,bounds_error=False,fill_value=1e100)
ddLdz = interp1d(zgrid,np.gradient(dLgrid,zgrid),bounds_error=False,fill_value=1e100)

zoft = interp1d(13700-tgrid,zgrid,bounds_error=False,fill_value=1000)
tofz = interp1d(zgrid,13700-tgrid,bounds_error=False,fill_value=1e100)
dtdz = interp1d(zgrid,np.gradient(13700-tgrid,zgrid),bounds_error=False,fill_value=1e100)

def load_data(samples_path, weights=True):
    data = read(samples_path, package="gw")

    m1det = data.samples_dict['C01:Mixed']['mass_1']
    m2det = data.samples_dict['C01:Mixed']['mass_2']
    dL = data.samples_dict['C01:Mixed']['luminosity_distance']

    redshift = zofdL(dL)
    m1s = m1det/(1+redshift)
    m2s = m2det/(1+redshift)
    mcs = (m1s*m2s)**(3/5)/(m1s + m2s)**(1/5)
    Ms = m1s + m2s
    qs = m2s/m1s
    
    # 0.08134597870775964 0.14305660510532858 6.021922663985887 6.203484262510087
    qmin, qmax = az.hdi(qs,0.999)
    mcmin, mcmax = az.hdi(mcs,0.999)
    print(qmin,qmax,mcmin,mcmax)
    
    if weights is True:
        wts = 1/(dL**2*(1+redshift)**2*ddLdz(redshift)*m1s**2/mcs)
    else:
        wts = np.ones(len(m1s))

    wts = wts/np.sum(wts)

    gwsamples = np.column_stack([mcs,qs])
    KDE = gaussian_kde(gwsamples.T, weights=wts)
    
    gwsamples_kde = KDE.resample(len(gwsamples)).T

    return KDE, gwsamples, gwsamples_kde, qmin, qmax, mcmin, mcmax
    