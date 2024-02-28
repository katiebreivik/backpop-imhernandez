import numpy as np
import emcee

from scipy.stats import gaussian_kde
import multiprocessing
from multiprocessing import Pool

import os
from argparse import ArgumentParser
import glob

from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d

import h5py
from astropy.cosmology import Planck15, FlatLambdaCDM, z_at_value
from astropy import units as u
import astropy.constants as constants
from pesummary.io import read
from backpop import *
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams['font.family']= 'Times New Roman'
matplotlib.rcParams['font.sans-serif']= ['Bitstream Vera Sans']
matplotlib.rcParams['text.usetex']= False
matplotlib.rcParams['mathtext.fontset']= 'cm'
matplotlib.rcParams['figure.figsize'] = (16.0, 10.0)

import seaborn as sns
sns.set_context('talk')
sns.set_style('ticks')
sns.set_palette('colorblind')
cs = sns.color_palette('colorblind',as_cmap=True)

import corner

optp = ArgumentParser()
optp.add_argument("--samples_path", help="path to event run dir")
optp.add_argument("--event_name", help="name of event")

optp.add_argument("--redshift_likelihood", type=str_to_bool, nargs='?', const=True, default=True)

optp.add_argument('--fixed_kicks', type=str_to_bool, nargs='?', const=True, default=False)
optp.add_argument('--same_alphas', type=str_to_bool, nargs='?', const=True, default=False)
optp.add_argument('--lowmass_secondary', type=str_to_bool, nargs='?', const=True, default=False)
optp.add_argument('--fixed_alphas', type=str_to_bool, nargs='?', const=True, default=False)

optp.add_argument("--nwalkers", type=int)
optp.add_argument("--nsteps", type=int)

optp.add_argument('--resume', type=str_to_bool, nargs='?', const=True, default=False)

opts = optp.parse_args()

samples_path = opts.samples_path
event_name = opts.event_name

output_path = "./results/" + event_name

try:
    os.makedirs(output_path, exist_ok=False)
except:
    print("Output directoryy already exists. Continuing...")
    pass

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

data = read(samples_path, package="gw")
samples = data.samples_dict['C01:Mixed']

m1det = data.samples_dict['C01:Mixed']['mass_1']
m2det = data.samples_dict['C01:Mixed']['mass_2']
dL = data.samples_dict['C01:Mixed']['luminosity_distance']

redshift = zofdL(dL)
m1s = m1det/(1+redshift)
m2s = m2det/(1+redshift)
mcs = (m1s*m2s)**(3/5)/(m1s + m2s)**(1/5)
Ms = m1s + m2s
qs = m2s/m1s

m1lo = 5.0
m2lo = 5.0
tblo = 5.0
elo = 0.0
alphalo_1 = 0.1
alphalo_2 = 0.1
vklo = 0.0
thetalo = 0.0
philo = -90.0
omegalo = 0.0
acc_limlo_1 = 0
acc_limlo_2 = 0
qc_kstar2lo = 0.5
qc_kstar3lo = 0.5
Zlo = 0.0001

m1hi = 150.0
m2hi = 150.0
tbhi = 5000.0
ehi = 0.9
alphahi_1 = 20.0
alphahi_2 = 20.0
vkhi = 300.0
thetahi = 360.0
phihi = 90.0
omegahi = 360
acc_limhi_1 = 1.0
acc_limhi_2 = 1.0
qc_kstar2hi = 10.0
qc_kstar3hi = 10.0
Zhi = 0.03

fixed_kicks = opts.fixed_kicks
same_alphas = opts.same_alphas
lowmass_secondary = opts.lowmass_secondary
fixed_alphas = opts.fixed_alphas

if (lowmass_secondary is False):
    if (fixed_kicks is True) and (same_alphas is True):
        lower_bound = np.array([m1lo, m2lo, np.log10(tblo), elo, alphalo_1, acc_limlo_1, acc_limlo_2, qc_kstar2lo, qc_kstar3lo, np.log10(Zlo)])
        upper_bound = np.array([m1hi, m2hi, np.log10(tbhi), ehi, alphahi_1, acc_limhi_1, acc_limhi_2, qc_kstar2hi, qc_kstar3hi, np.log10(Zhi)])

        evolution = evolv2_fixed_kicks_same_alphas

        config_name = "backpop_fixed_kicks_same_alphas"

    if (fixed_kicks is True) and (same_alphas is False):
        lower_bound = np.array([m1lo, m2lo, np.log10(tblo), elo, alphalo_1, alphalo_2, acc_limlo_1, acc_limlo_2, qc_kstar2lo, qc_kstar3lo, np.log10(Zlo)])
        upper_bound = np.array([m1hi, m2hi, np.log10(tbhi), ehi, alphahi_1, alphahi_2, acc_limhi_1, acc_limhi_2, qc_kstar2hi, qc_kstar3hi, np.log10(Zhi)])

        evolution = evolv2_fixed_kicks

        config_name = "backpop_fixed_kicks"

    if (fixed_kicks is False) and (same_alphas is True):
        lower_bound = np.array([m1lo, m2lo, np.log10(tblo), elo, alphalo_1, vklo, thetalo, philo, omegalo, vklo, thetalo, philo, omegalo, acc_limlo_1, acc_limlo_2, qc_kstar2lo, qc_kstar3lo, np.log10(Zlo)])
        upper_bound = np.array([m1hi, m2hi, np.log10(tbhi), ehi, alphahi_1, vkhi, thetahi, phihi, omegahi, vkhi, thetahi, phihi, omegahi, acc_limhi_1, acc_limhi_2, qc_kstar2hi, qc_kstar3hi, np.log10(Zhi)])

        evolution = evolv2_same_alphas

        config_name = "backpop_same_alphas"

    if (fixed_kicks is False) and (same_alphas is False):
        lower_bound = np.array([m1lo, m2lo, np.log10(tblo), elo, alphalo_1, alphalo_2, vklo, thetalo, philo, omegalo, vklo, thetalo, philo, omegalo, acc_limlo_1, acc_limlo_2, qc_kstar2lo, qc_kstar3lo, np.log10(Zlo)])
        upper_bound = np.array([m1hi, m2hi, np.log10(tbhi), ehi, alphahi_1, alphahi_2, vkhi, thetahi, phihi, omegahi, vkhi, thetahi, phihi, omegahi, acc_limhi_1, acc_limhi_2, qc_kstar2hi, qc_kstar3hi, np.log10(Zhi)])

        evolution = evolv2

        config_name = "backpop"

else:
    lower_bound = np.array([m1lo, m2lo, np.log10(tblo), elo, alphalo_1, alphalo_2, vklo, vklo, thetalo, philo, acc_limlo_1, acc_limlo_2, qc_kstar2lo, qc_kstar3lo, np.log10(Zlo)])
    upper_bound = np.array([m1hi, m2hi, np.log10(tbhi), ehi, alphahi_1, alphahi_2, vkhi, vkhi, thetahi, phihi, acc_limhi_1, acc_limhi_2, qc_kstar2hi, qc_kstar3hi, np.log10(Zhi)])

    evolution = evolv2_lowmass_secondary_new

    config_name = "backpop_lowmass_secondary_new" 

print(config_name)


redshift_likelihood = opts.redshift_likelihood

if redshift_likelihood is True:
    qmin = qs.min()
    qmax = qs.max()
    mcmin = mcs.min()
    mcmax = mcs.max()
    print(qmin,qmax,mcmin,mcmax)

    samples = np.column_stack([mcs,qs,redshift])
    KDE = gaussian_kde(samples.T)

    def likelihood(coord):
        for i in range(len(coord)):
            if (coord[i]<lower_bound[i] or coord[i]>upper_bound[i]):
                return -np.inf
        if coord[1] > coord[0]:
            return -np.inf
        result = evolution(*coord)
        m1, m2, z, dL = result[1][0], result[1][1], zoft(result[0]), dLoft(result[0])
        if (m1 == 0.0) or (m2 == 0.0):
            return -np.inf
        q = m2/m1
        mc = (m1*m2)**(3/5)/(m1 + m2)**(1/5)
        if ((q > qmin) and (q < qmax) and (mc > mcmin) and (mc < mcmax)):
            gw_coord = np.array([mc, q, z])
            ll = KDE.logpdf(gw_coord) - 2*np.log(dL) - 2*np.log1p(z) - np.log(ddLdz(z)) - np.log(m1**2/mc)
            return ll[0]
        else:
            return -np.inf
else:
    print("sampling using masses only")
    ddLdz = ddL_of_z(redshift,dL)
    jacobian = dL**2*(1+redshift)**2*ddLdz

    samples = np.column_stack([m1s,m2s])#,redshift])
    KDE = gaussian_kde(samples.T)

    def likelihood(coord):
        for i in range(len(coord)):
            if (coord[i]<lower_bound[i] or coord[i]>upper_bound[i]):
                return -np.inf
        if coord[1] > coord[0]:
            return -np.inf
        result = evolution(*coord)
        #m1, m2, z = result[1][0], result[1][1], zoft(result[0])
        m1, m2 = result[1][0], result[1][1]
        if (m1 == 0.0) or (m2 == 0.0):
            return -np.inf
    #     gw_coord = np.array([m1, m2, z])
    #     dL = dLofz(z)
        gw_coord = np.array([m1, m2])
        return KDE.logpdf(gw_coord)# - 2*np.log(dL) - 2*np.log1p(z) - np.log(ddL_of_z(z,dL)) - 2*np.log(dL)

n_dim = len(lower_bound)
n_walkers = opts.nwalkers
n_steps = opts.nsteps

p0 = np.random.uniform(lower_bound, upper_bound, size=(n_walkers, len(lower_bound)))
m2s, m1s = np.sort([p0[:,0],p0[:,1]],axis=0)
p0[:,0] = m1s
p0[:,1] = m2s

print(p0.shape)

resume = opts.resume
if resume is True:
    data = np.load("./results/" + event_name + "/" + config_name + ".npz")

    chain_resume = data["chain"]
    n_walkers = data["nwalkers"]
    p0 = chain_resume[-1]
    print(p0.shape)

num_cores = int(len(os.sched_getaffinity(0)))
print("using multiprocessing with " + str(num_cores) + " cores")
with Pool(int(2*num_cores-2)) as pool:
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, likelihood, pool=pool,
                                    moves=[emcee.moves.KDEMove()])
    
    sampler.run_mcmc(p0, n_steps, progress=True)


if resume is True:
    chain = np.concatenate((chain_resume, sampler.get_chain()))
    n_steps = n_steps + chain_resume.shape[0]
    
else:
    chain = sampler.get_chain()
    

cut = int(0.5*n_steps)
flat_chain = chain[cut:,:,:].reshape(-1,n_dim)

np.savez("./results/" + event_name + "/" + config_name,
         nwalkers=n_walkers,
         n_steps=n_steps,
         chain=chain,
         flat_chain=flat_chain)

print("Flat chain has " + str(chain.shape[0]) + " samples")