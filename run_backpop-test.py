import numpy as np

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
from dynesty.utils import resample_equal
import arviz as az

optp = ArgumentParser()
optp.add_argument("--event_name", help="name of event")
optp.add_argument('--config_name', help="configuration to use")
optp.add_argument("--nlive", type=int, default=1000)
optp.add_argument("--resume", type=str_to_bool, nargs='?', const=False, default=False)

opts = optp.parse_args()

event_name = opts.event_name
resume = opts.resume
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

evolution, lower_bound, upper_bound = get_backpop_config(config_name)

from scipy.stats import multivariate_normal
mean = np.array([30.0, 8.0])
cov = np.array([[1.5**2, 0,], [0, 0.3**2]])
rv = multivariate_normal(mean, cov)

samples = rv.rvs(10000)
m1s = samples[:,0]
m2s = samples[:,1]

m2s, m1s = np.sort([m1s,m2s],axis=0)
mcs = (m1s*m2s)**(3/5)/(m1s + m2s)**(1/5)
qs = m2s/m1s

gwsamples = np.column_stack([mcs,qs])

m1min, m1max = az.hdi(m1s,0.99)
m2min, m2max = az.hdi(m2s,0.99)

def likelihood(coord):
    vals = list(coord.values())
    result = evolution(*vals)
    m1, m2, dt = result[1][0], result[1][1], result[0]
    if (m1 == 0.0) or (m2 == 0.0):
        return (-np.inf, dt)
    q = m2/m1
    if ((q < 0.5)):
        gw_coord = np.array([m1, m2])
        ll = rv.logpdf(gw_coord)
        return (ll, dt)
    else:
        return (-np.inf, dt)

prior = Prior()
for i in range(len(params)):
    prior.add_parameter(params[i], dist=(lower_bound[i], upper_bound[i]))

num_cores = int(len(os.sched_getaffinity(0)))
num_threads = int(2*num_cores-2)
print("using multiprocessing with " + str(num_cores) + " cores")

dtype = [("delay_time", float)]

n_live = opts.nlive
sampler = Sampler(prior, likelihood, n_live=n_live,pool=num_threads, blobs_dtype=dtype,
                  filepath="./results/" + event_name + "/" + config_name + "_checkpoint.hdf5", resume=resume)
sampler.run(verbose=True)

points, log_w, log_l, delay_times = sampler.posterior(return_blobs=True)

logz = sampler.log_z
dweights = np.exp(log_w - logz)
postsamples = resample_equal(points, dweights)

np.savez("./results/" + event_name + "/" + config_name,
         flat_chain=postsamples,
         flat_delay_times=delay_times,
         gwsamples=gwsamples)
