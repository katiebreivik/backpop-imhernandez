import numpy as np
import time

from scipy.stats import gaussian_kde

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
from nautilus import Prior, Sampler

start = time.time()
print("Starting timer")

optp = ArgumentParser()
optp.add_argument("--samples_path", help="path to event run dir")
optp.add_argument("--event_name", help="name of event")
optp.add_argument('--config_name', help="configuration to use")
optp.add_argument("--weights", type=str_to_bool, nargs='?', const=False, default=True)
optp.add_argument("--nlive", type=int, default=3000)
optp.add_argument("--neff", type=int, default=10000)
optp.add_argument("--resume", type=str_to_bool, nargs='?', const=False, default=False)

opts = optp.parse_args()

samples_path = opts.samples_path
event_name = opts.event_name
weights = opts.weights
resume = opts.resume
print(weights)
print(resume)

output_path = "./results/" + event_name

try:
    os.makedirs(output_path, exist_ok=False)
except:
    print("Output directory already exists. Continuing...")
    pass

config_name = opts.config_name
params = labels_dict[config_name]
print(config_name)

evolution, lower_bound, upper_bound, params_in = get_backpop_config(config_name)

#mean = np.array([6.1, 0.112])
#cov = np.array([[0.1**2, 0], [0, 0.005**2]])
#rv = multivariate_normal(mean, cov)
KDE, gwsamples, gwsamples_kde, qmin, qmax, mcmin, mcmax = load_data(samples_path, weights)

# split out rv with KDE if you have GW samples
def likelihood(KDE, lower_bound, upper_bound, params_out, qmax, params_in):
    # enforce limits on physical values
    for i, name in enumerate(params_in):
        val = params_in[name]
        if name in ["theta1", "phi1", "omega1", "theta2", "phi2", "omega2"]:
            if val < lower_bound[i] or val > upper_bound[i]:
                # return invalid flattened arrays
                return -np.inf, np.full(np.prod(BPP_SHAPE), np.nan, dtype=float), np.full(np.prod(KICK_SHAPE), np.nan, dtype=float)

    # evolve the binary
    result = evolv2(params_in, params_out)
    # check result
    if result[0] is None:
        return -np.inf, np.full(np.prod(BPP_SHAPE), np.nan, dtype=float), np.full(np.prod(KICK_SHAPE), np.nan, dtype=float)

    # flatten arrays and force dtype
    bpp_flat = np.array(result[1], dtype=float).ravel()
    kick_flat = np.array(result[2], dtype=float).ravel()
    
    # check shapes
    if bpp_flat.size != np.prod(BPP_SHAPE) or kick_flat.size != np.prod(KICK_SHAPE):
        return -np.inf, np.full(np.prod(BPP_SHAPE), np.nan, dtype=float), np.full(np.prod(KICK_SHAPE), np.nan, dtype=float)
    
    m1 = result[0]['mass_1']
    m2 = result[0]['mass_2']
    q = np.where(m2 <= m1, m2/m1, m1/m2)  # Ensure q is defined only when m2 <= m1
    mc = (m1*m2)**(3/5)/(m1 + m2)**(1/5)
    if (q < qmax):
        gw_coord = np.array([mc, q])
        ll = KDE.logpdf(gw_coord)
        return (ll[0], bpp_flat, kick_flat)
    else:
        return -np.inf, np.full(np.prod(BPP_SHAPE), np.nan, dtype=float), np.full(np.prod(KICK_SHAPE), np.nan, dtype=float)

# Set up Nautilus prior
prior = Prior()
for i in range(len(params_in)):
    prior.add_parameter(params_in[i], dist=(lower_bound[i], upper_bound[i]))



params_out=['mass_1', 'mass_2']
qmax = 1.0
#num_cores = int(len(os.sched_getaffinity(0)))
#num_threads = int(2*num_cores-2)
num_threads = 1
print("using multiprocessing with " + str(num_threads) + " threads")
    
dtype = [('bpp', float, 35*len(BPP_COLUMNS)), ('kick_info', float, 2*len(KICK_COLUMNS))]
n_live = opts.nlive
n_eff = opts.neff
sampler = Sampler(
    prior=prior, 
    likelihood=likelihood, 
    n_live=n_live, 
    pool=num_threads,
    blobs_dtype=dtype,
    filepath="./results/" + event_name + "/" + config_name + "_checkpoint.hdf5",
    resume=resume, 
    likelihood_args=(KDE, lower_bound, upper_bound, params_out, qmax)
)
sampler.run(n_eff=n_eff,verbose=True,discard_exploration=True)
    
points, log_w, log_l, blobs = sampler.posterior(return_blobs=True)
    
np.save("./results/" + event_name + "/" + config_name + "/points.npy", points)
np.save("./results/" + event_name + "/" + config_name + "/log_w.npy", log_w)
np.save("./results/" + event_name + "/" + config_name + "/log_l.npy", log_l)
np.save("./results/" + event_name + "/" + config_name + "/blobs.npy", blobs)

#def likelihood(coord):
#    vals = list(coord.values())
#    result = evolution(*vals)
#    m1, m2, dt = result[1][0], result[1][1], result[0]
#    if (m1 == 0.0) or (m2 == 0.0):
#        return (-np.inf, dt)
#    q = m2/m1
#    mc = (m1*m2)**(3/5)/(m1 + m2)**(1/5)
#    if ((q < qmax)):
#        gw_coord = np.array([mc, q])
#        ll = KDE.logpdf(gw_coord)
#        return (ll[0], dt)
#    else:
#        return (-np.inf, dt)

#prior = Prior()
#for i in range(len(params)):
#    prior.add_parameter(params[i], dist=(lower_bound[i], upper_bound[i]))
#
#num_cores = int(len(os.sched_getaffinity(0)))
#num_threads = int(2*num_cores-2)
#print("using multiprocessing with " + str(num_threads) + " threads")
#
#dtype = [("delay_time", float)]
#
#n_live = opts.nlive
#n_eff = opts.neff
#sampler = Sampler(prior, likelihood, n_live=n_live, pool=num_threads, blobs_dtype=dtype,
#                  filepath="./results/" + event_name + "/" + config_name + "_checkpoint.hdf5", resume=resume)
#sampler.run(n_eff=n_eff,verbose=True,discard_exploration=True)
#
#points, log_w, log_l, delay_times = sampler.posterior(return_blobs=True)
#
#logz = sampler.log_z
#dweights = np.exp(log_w - logz)
#postsamples = resample_equal(points, dweights)
#
#np.savez("./results/" + event_name + "/" + config_name,
#         flat_chain=postsamples,
#         flat_delay_times=delay_times,
#         gwsamples=gwsamples,
#         gwsamples_kde=gwsamples_kde)
#
#end = time.time()
#print("Execution time: " + str(end - start) + " seconds")