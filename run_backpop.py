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


optp = ArgumentParser()
optp.add_argument("--samples_path", help="path to event run dir")
optp.add_argument("--event_name", help="name of event")
optp.add_argument("--redshift_likelihood", type=str_to_bool, nargs='?', const=False, default=True)

optp.add_argument('--config_name', help="configuration to use")

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

config_name = opts.config_name
redshift_likelihood = opts.redshift_likelihood
print(config_name)

evolution, lower_bound, upper_bound = get_backpop_config(config_name)

KDE, gwsamples, qmin, qmax, mcmin, mcmax = load_data(samples_path, redshift_likelihood=redshift_likelihood)

if redshift_likelihood is True:
    config_name = config_name + '_redshift'
    print("using redshift")

    def likelihood(coord):
        for i in range(len(coord)):
            if (coord[i]<lower_bound[i] or coord[i]>upper_bound[i]):
                return -np.inf, None
        
        result = evolution(*coord)
        m1, m2, z = result[1][0], result[1][1], zoft(result[0])
        if (m1 == 0.0) or (m2 == 0.0):
            return -np.inf, None
        q = m2/m1
        mc = (m1*m2)**(3/5)/(m1 + m2)**(1/5)
        if ((q < qmax) and (mc < mcmax) and (q > qmin) and (mc > mcmin)):
            print(q)
            gw_coord = np.array([mc, q, z])
            ll = KDE.logpdf(gw_coord)
            return ll[0], result[0]
        else:
            return -np.inf, None
else:
    print("NOT using redshift")

    def likelihood(coord):
        for i in range(len(coord)):
            if (coord[i]<lower_bound[i] or coord[i]>upper_bound[i]):
                return -np.inf, None

        result = evolution(*coord)
        m1, m2, z = result[1][0], result[1][1], zoft(result[0])
        if (m1 == 0.0) or (m2 == 0.0):
            return -np.inf, None
        q = m2/m1
        mc = (m1*m2)**(3/5)/(m1 + m2)**(1/5)
        if ((q < qmax) and (mc < mcmax) and (q > qmin) and (mc > mcmin)):
            print(q)
            gw_coord = np.array([mc, q])
            ll = KDE.logpdf(gw_coord)
            return ll[0], result[0]
        else:
            return -np.inf, None

n_dim = len(lower_bound)
n_walkers = opts.nwalkers
n_steps = opts.nsteps

dtype = [("delay_time", float)]

p0 = np.random.uniform(lower_bound, upper_bound, size=(n_walkers, len(lower_bound)))

print(p0.shape)

resume = opts.resume
if resume is True:
    data = np.load("./results/" + event_name + "/" + config_name + ".npz")

    p0 = data["chain_resume"]
    n_walkers = p0.shape[0]
    print(p0.shape)

num_cores = int(len(os.sched_getaffinity(0)))
print("using multiprocessing with " + str(num_cores) + " cores")
with Pool(int(2*num_cores-2)) as pool:
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, likelihood, pool=pool, blobs_dtype=dtype,
                                    moves=[emcee.moves.KDEMove()])
    
    sampler.run_mcmc(p0, n_steps, thin_by=2, progress=True)

chain = sampler.get_chain()
blobs = sampler.get_blobs()
log_probs = sampler.get_log_prob()

delay_times = blobs["delay_time"]

chain_resume = chain[np.where((log_probs==np.max(log_probs)))[0]][-1]

cut = int(0.5*n_steps)

flat_chain = chain[cut:,:,:].reshape(-1,n_dim)
flat_delay_times = delay_times[cut:,:].reshape(-1,1)
flat_log_probs = log_probs[cut:,:].reshape(-1,1)

flat_chain_length = flat_chain.shape[0]

if flat_chain_length > 256000:
    select = np.random.choice(flat_chain_length,256000,replace=False)
else:
    select = np.random.choice(flat_chain_length,flat_chain_length,replace=False)

flat_chain = flat_chain[select]
flat_delay_times = flat_delay_times[select]
flat_log_probs = flat_log_probs[select]

np.savez("./results/" + event_name + "/" + config_name,
         chain_resume=chain_resume,
         flat_chain=flat_chain,
         flat_delay_times=flat_delay_times,
         flat_log_probs=flat_log_probs,
         gwsamples=gwsamples)
