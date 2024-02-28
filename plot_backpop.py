import numpy as np
import emcee

import os
from argparse import ArgumentParser
import glob


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

labels_dict = {"backpop_fixed_kicks_same_alphas" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$e$',r'$\alpha$', r'$f_{\rm lim,1}$', r'$f_{\rm lim,2}$', r'$q_{\rm HG}$', r'$q_{\rm GB}$', r'$\log_{10}Z$'],
               "backpop_fixed_kicks" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$e$',r'$\alpha$',r'$f_{\rm lim,1}$', r'$f_{\rm lim,2}$', r'$q_{\rm HG}$', r'$q_{\rm GB}$', r'$\log_{10}Z$'],
               "backpop_same_alphas" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$e$',r'$\alpha$',r'$v_1$',r'$\theta_1$',r'$\phi_1$',r'$\omega_1$',r'$v_2$',r'$\theta_2$',r'$\phi_2$',r'$\omega_2$',r'$f_{\rm lim,1}$', r'$f_{\rm lim,2}$', r'$q_{\rm HG}$', r'$q_{\rm GB}$', r'$\log_{10}Z$'],
               "backpop" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$e$',r'$\alpha_1$',r'$\alpha_2$',r'$v_1$',r'$\theta_1$',r'$\phi_1$',r'$\omega_1$',r'$v_2$',r'$\theta_2$',r'$\phi_2$',r'$\omega_2$',r'$f_{\rm lim,1}$', r'$f_{\rm lim,2}$', r'$q_{\rm HG}$', r'$q_{\rm GB}$', r'$\log_{10}Z$'],
               "backpop_lowmass_secondary" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$e$',r'$\alpha_1$',r'$\alpha_2$',r'$v_2$',r'$\theta_2$',r'$\phi_2$',r'$\omega_2$',r'$f_{\rm lim,1}$', r'$f_{\rm lim,2}$', r'$q_{\rm HG}$', r'$q_{\rm GB}$', r'$\log_{10}Z$'],
               "backpop_lowmass_secondary_new" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$e$',r'$\alpha_1$',r'$\alpha_2$',r'$v_1$',r'$v_2$',r'$\theta_2$',r'$\phi_2$',r'$f_{\rm lim,1}$', r'$f_{\rm lim,2}$', r'$q_{\rm HG}$', r'$q_{\rm GB}$', r'$\log_{10}Z$'],
               "backpop_lowmass_secondary_new_fixed_alphas" : [r'$m_1$',r'$m_2$',r'$\log_{10}t_b$',r'$e$',r'$v_1$',r'$v_2$',r'$\theta_2$',r'$\phi_2$',r'$f_{\rm lim,1}$', r'$f_{\rm lim,2}$', r'$q_{\rm HG}$', r'$q_{\rm GB}$', r'$\log_{10}Z$']
              }

labels = labels_dict[config_name]

n_dim = len(labels)
chain = data["chain"]
nwalkers = data["nwalkers"]
nsteps = data["n_steps"]

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

np.savez("./results/" + event_name + "/" + config_name + "_flatchain", flat_chain=flat_chain)
