import numpy as np
import emcee

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

import corner

optp = ArgumentParser()
optp.add_argument("--samples", help="path to backpop data")
optp.add_argument("--event_name", help="name of event")
optp.add_argument("--nsamples", type=int)
optp.add_argument("--fcut", type=float, default=0.5)

opts = optp.parse_args()

samples = opts.samples
event_name = opts.event_name
nsamples = opts.nsamples
fcut = opts.fcut

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

n_dim = len(labels)
chain = data["chain"]
nwalkers = data["nwalkers"]
nsteps = data["n_steps"]
gwsamples = data["gwsamples"]

evolution, lower_bound, upper_bound = get_backpop_config(config_name)
# chain = chain.reshape(-1,n_dim)


# for i in range(len(labels)):
#     print(labels[i])
#     plt.figure()
#     plt.plot(chain[:,i].T)
#     plt.xlabel(labels[i])
#     plt.savefig("./results/" + event_name + "/" + config_name + labels[i] + "_chain.pdf")
#     plt.close()


cut = int(fcut*nsteps)
flat_chain = chain[cut:,:,:].reshape(-1,n_dim)

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

print("Flat chain has " + str(flat_chain.shape[0]) + " samples")

select = np.random.choice(flat_chain.shape[0],nsamples,replace=False)
flat_chain_plot = flat_chain[select]

fig = corner.corner(flat_chain_plot,labels=labels,
                    color=cs[2],levels=[0.68,0.95],  quantiles=[0.05,0.68,0.95],  show_titles=True,
                    label_kwargs={"fontsize": 20},
                    title_kwargs={"fontsize": 18},
                    hist_kwargs={"linewidth": 2})

for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=14)

plt.savefig("./results/" + event_name + "/" + config_name + ".pdf")
plt.close()

#### MAKE mcq PLOT ####
m1s_b = []
m2s_b = []
ts_b = []

for k in tqdm(range(len(flat_chain_plot))):
    result = evolution(*flat_chain_plot[k])
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
ts_b = np.array(ts_b)

qs_b = m2s_b/m1s_b
mcs_b = (m1s_b*m2s_b)**(3/5)/(m1s_b + m2s_b)**(1/5)

fig = corner.corner(np.column_stack([gwsamples[:,0],gwsamples[:,1]]),labels=[r'$M_c$', r'$q$'],
                    color='gray',levels=[0.68,0.95],  quantiles=[0.05,0.68,0.95],  show_titles=True,
                    label_kwargs={"fontsize": 20},
                    title_kwargs={"fontsize": 18},
                    hist_kwargs={"linewidth": 2})
corner.corner(np.column_stack([mcs_b,qs_b]),color='green',fig=fig)

for ax in fig.get_axes():
    ax.tick_params(axis='both', labelsize=14)
    
plt.savefig("./results/" + event_name + "/" + config_name + "_forward.pdf"')
plt.close()
