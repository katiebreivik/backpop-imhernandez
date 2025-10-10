import numpy as np
import pandas as pd
import emcee
import bilby

import os
from argparse import ArgumentParser
import glob

from backpop import *
import h5py
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

cols_keep = ['tphys', 'mass_1', 'mass_2', 'kstar_1', 'kstar_2', 'porb', 'ecc', 'evol_type', 'rad_1', 'rad_2']

def load_process_data_fast(dat_path_in, param_columns, n_posterior_samples=10000):
   
    # Load data
    points = np.load(dat_path_in + "/points.npy")
    log_z = np.load(dat_path_in + "/log_z.npy")
    log_w = np.load(dat_path_in + "/log_w.npy")
    blobs = np.load(dat_path_in + "/blobs.npy", allow_pickle=True)

    dweights = np.exp(log_w - log_z)
    dweights = dweights/np.sum(dweights)
    
    indices = np.random.choice(len(points), size=n_posterior_samples, replace=True, p=dweights)
    points_sample = pd.DataFrame(points[indices], columns=param_columns)
    points = []
    log_w = []
    log_z = []
    dweights = []

    # Process blobs
    blobs_sample = blobs[indices]
    blobs = []
    
    # --- Vectorized BPP and Kick assembly ---
    # Extract bpp and kick arrays from all blobs in one pass
    bpp_list = [b['bpp'].reshape(BPP_SHAPE) for b in blobs_sample]
    kick_list = [b['kick_info'].reshape(KICK_SHAPE) for b in blobs_sample]
    
    # Concatenate into big arrays
    bpp_all = np.concatenate(bpp_list, axis=0)   # shape: (sum of all rows, BPP_COLS)
    kick_all = np.concatenate(kick_list, axis=0) # shape: (sum of all rows, KICK_COLS)

    # Build bin_num arrays
    bin_sizes_bpp = np.array([b.shape[0] for b in bpp_list])
    bin_sizes_kick = np.array([k.shape[0] for k in kick_list])
    bin_nums_bpp = np.repeat(np.arange(len(bpp_list)), bin_sizes_bpp)
    bin_nums_kick = np.repeat(np.arange(len(kick_list)), bin_sizes_kick)

    # Build DataFrames
    bpp_full = pd.DataFrame(bpp_all, columns=cols_keep)
    kick_full = pd.DataFrame(kick_all, columns=KICK_COLUMNS)
    bpp_full['bin_num'] = bin_nums_bpp
    kick_full['bin_num'] = bin_nums_kick

    # Filter valid bpp rows
    bpp_full = bpp_full.loc[bpp_full.mass_1 > 0]

    return points_sample, bpp_full, kick_full

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure ti

import corner

optp = ArgumentParser()
optp.add_argument("--samples", help="path to backpop data")
optp.add_argument("--event_name", help="name of event")
optp.add_argument('--config_name', help="configuration to use")

opts = optp.parse_args()

samples = opts.samples
event_name = opts.event_name
config_name = opts.config_name
print(config_name)

output_path = "./results/" + event_name + "/" + config_name + "/"

try:
    os.makedirs(output_path, exist_ok=False)
except:
    print("Output directoryy already exists. Continuing...")
    pass

labels = labels_dict[config_name]

points_sample, bpp_full, kick_full = load_process_data_fast(samples, labels, n_posterior_samples=10000)

data = points_sample.to_numpy()

m2 = data[:,0]*data[:,1]
data[:,1] = m2

fig = corner.corner(data,labels=labels,
                    color=cs[2],levels=[0.68,0.95],  quantiles=[0.05,0.68,0.95],  show_titles=True,
                    label_kwargs={"fontsize": 20},
                    title_kwargs={"fontsize": 18},
                    hist_kwargs={"linewidth": 2})

for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=14)

plt.savefig(output_path + "corner.pdf")
plt.close()

print(bpp_full)
# #### MAKE m1,m2 PLOT ####
# m1s_b = []
# m2s_b = []
# ts_b = []
# bpps = []

# for k in tqdm(range(len(flat_chain[select]))):
#     result = evolution(*flat_chain[select][k])
#     m1b, m2b, tb = result[1][0], result[1][1], result[0]
#     bpp = result[3]
#     if ((m1b == 0.0) or (m2b == 0.0)):
#         continue
#     else:
#         m1s_b.append(m1b)
#         m2s_b.append(m2b)
#         ts_b.append(tb)
#         bpps.append(bpp)
# bpps = pd.concat(bpps)
# bpps.to_hdf("./results/" + event_name + "/" + os.path.basename(samples)[:-4] + "_bpp.h5", key="bpp")

# print(len(m1s_b))
# m1s_b = np.array(m1s_b)
# m2s_b = np.array(m2s_b)
# zs_b = zoft(np.array(ts_b))

# m1s,m2s = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(gwsamples[:,0], gwsamples[:,1])
# m1s_kde,m2s_kde = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(gwsamples_kde[:,0], gwsamples_kde[:,1])

# mcs_b = bilby.gw.conversion.component_masses_to_chirp_mass(m1s_b,m2s_b)
# qs_b = bilby.gw.conversion.component_masses_to_mass_ratio(m1s_b,m2s_b)

# mcs = bilby.gw.conversion.component_masses_to_chirp_mass(m1s,m2s)
# qs = bilby.gw.conversion.component_masses_to_mass_ratio(m1s,m2s)

# mcs_kde = bilby.gw.conversion.component_masses_to_chirp_mass(m1s_kde,m2s_kde)
# qs_kde = bilby.gw.conversion.component_masses_to_mass_ratio(m1s_kde,m2s_kde)


# backpop_samples_plot = np.column_stack([mcs_b,qs_b])
# gwsamples_plot = np.column_stack([mcs,qs])
# gwsamples_kde_plot = np.column_stack([mcs_kde,qs_kde])


# weights_1=np.ones(len(gwsamples_kde_plot))*len(backpop_samples_plot)/len(gwsamples_kde_plot)
# weights_2=np.ones(len(backpop_samples_plot))*len(gwsamples_plot)/len(backpop_samples_plot)

# fig = corner.corner(backpop_samples_plot,labels=[r'$M_c$', r'$q$'], color='green',
#                     levels=[0.68,0.95],  quantiles=[0.05,0.68,0.95],# show_titles=True,
#                     label_kwargs={"fontsize": 28},
#                     title_kwargs={"fontsize": 18},
#                     hist_kwargs={"linewidth": 2})
# corner.corner(gwsamples_kde_plot,color='gray',fig=fig,weights=weights_1)
# #corner.corner(gwsamples_plot,color='green',fig=fig,weights=weights_2)

# for ax in fig.get_axes():
#     ax.tick_params(axis='both', labelsize=14)
    
# plt.savefig("./results/" + event_name + "/" + os.path.basename(samples)[:-4] + "_forward.pdf")
# plt.close()
