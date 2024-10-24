import numpy as np
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
optp.add_argument("--nsamples", type=int)

opts = optp.parse_args()

samples = opts.samples
event_name = opts.event_name
nsamples = opts.nsamples

output_path = "./results/" + event_name

try:
    os.makedirs(output_path, exist_ok=False)
except:
    print("Output directoryy already exists. Continuing...")
    pass

data = np.load(samples)
config_name = os.path.basename(samples)[:-4]

print(config_name)

labels = labels_dict[config_name]

flat_chain = data["flat_chain"]
gwsamples = data["gwsamples"]
gwsamples_kde = data["gwsamples_kde"]

evolution, lower_bound, upper_bound = get_backpop_config(config_name)

print("Flat chain has " + str(flat_chain.shape[0]) + " samples")

select = np.random.choice(flat_chain.shape[0],nsamples,replace=False)
flat_chain_plot = flat_chain[select]

m2 = flat_chain_plot[:,0]*flat_chain_plot[:,1]
flat_chain_plot_corner = flat_chain_plot
flat_chain_plot_corner[:,1] = m2

fig = corner.corner(flat_chain_plot_corner,labels=labels,
                    color=cs[2],levels=[0.68,0.95],  quantiles=[0.05,0.68,0.95],  show_titles=True,
                    label_kwargs={"fontsize": 20},
                    title_kwargs={"fontsize": 18},
                    hist_kwargs={"linewidth": 2})

for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=14)

plt.savefig("./results/" + event_name + "/" + os.path.basename(samples)[:-4] + ".pdf")
plt.close()

#### MAKE m1,m2 PLOT ####
m1s_b = []
m2s_b = []
ts_b = []

for k in tqdm(range(len(flat_chain[select]))):
    result = evolution(*flat_chain[select][k])
    m1b, m2b, tb = result[1][0], result[1][1], result[0]
    if ((m1b == 0.0) or (m2b == 0.0)):
        continue
    else:
        m1s_b.append(m1b)
        m2s_b.append(m2b)
        ts_b.append(tb)

print(len(m1s_b))
m1s_b = np.array(m1s_b)
m2s_b = np.array(m2s_b)
zs_b = zoft(np.array(ts_b))

m1s,m2s = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(gwsamples[:,0], gwsamples[:,1])
m1s_kde,m2s_kde = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_component_masses(gwsamples_kde[:,0], gwsamples_kde[:,1])

backpop_samples_plot = np.column_stack([m1s_b,m2s_b])
gwsamples_plot = np.column_stack([m1s,m2s])
gwsamples_kde_plot = np.column_stack([m1s_kde,m2s_kde])


weights_1=np.ones(len(gwsamples_kde_plot))*len(gwsamples_plot)/len(gwsamples_kde_plot)
weights_2=np.ones(len(backpop_samples_plot))*len(gwsamples_plot)/len(backpop_samples_plot)

fig = corner.corner(gwsamples_plot,labels=[r'$m_1$', r'$m_2$', r'$z$'],
                    color='gray',levels=[0.68,0.95],  quantiles=[0.05,0.68,0.95],  show_titles=True,
                    label_kwargs={"fontsize": 20},
                    title_kwargs={"fontsize": 18},
                    hist_kwargs={"linewidth": 2})
#corner.corner(gwsamples_kde_plot,color='orange',fig=fig,weights=weights_1)
corner.corner(backpop_samples_plot,color='green',fig=fig,weights=weights_2)

for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=14)
    
plt.savefig("./results/" + event_name + "/" + os.path.basename(samples)[:-4] + "_forward.pdf")
plt.close()
