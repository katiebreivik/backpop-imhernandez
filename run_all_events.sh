#!/bin/python
python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_nocosmo.h5 --event_name GW150914 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False
python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW150914_095045_PEDataRelease_mixed_nocosmo.h5 --event_name GW150914 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False --resume True
python plot_backpop.py --samples ./results/GW150914/backpop_fixed_kicks_minimal.npz --event_name GW150914 --nsamples 10000

python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW151012_095443_PEDataRelease_mixed_nocosmo.h5 --event_name GW151012 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 1000000 --redshift_likelihood False
python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW151012_095443_PEDataRelease_mixed_nocosmo.h5 --event_name GW151012 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False --resume True
python plot_backpop.py --samples ./results/GW151012/backpop_fixed_kicks_minimal.npz --event_name GW151012 --nsamples 10000

python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW151226_033853_PEDataRelease_mixed_nocosmo.h5 --event_name GW151226 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False
python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW151226_033853_PEDataRelease_mixed_nocosmo.h5 --event_name GW151226 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False --resume True
python plot_backpop.py --samples ./results/GW151226/backpop_fixed_kicks_minimal.npz --event_name GW151226 --nsamples 10000

python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW170104_101158_PEDataRelease_mixed_nocosmo.h5 --event_name GW170104 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False
python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW170104_101158_PEDataRelease_mixed_nocosmo.h5 --event_name GW170104 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False --resume True
python plot_backpop.py --samples ./results/GW170104/backpop_fixed_kicks_minimal.npz --event_name GW170104 --nsamples 10000

python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW170608_020116_PEDataRelease_mixed_nocosmo.h5 --event_name GW170608 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False
python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW170608_020116_PEDataRelease_mixed_nocosmo.h5 --event_name GW170608 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False --resume True
python plot_backpop.py --samples ./results/GW170608/backpop_fixed_kicks_minimal.npz --event_name GW170608 --nsamples 10000

python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW170729_185629_PEDataRelease_mixed_nocosmo.h5 --event_name GW170729 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False
python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW170729_185629_PEDataRelease_mixed_nocosmo.h5 --event_name GW170729 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False --resume True
python plot_backpop.py --samples ./results/GW170729/backpop_fixed_kicks_minimal.npz --event_name GW170729 --nsamples 10000

python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW170809_082821_PEDataRelease_mixed_nocosmo.h5 --event_name GW170809 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False
python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW170809_082821_PEDataRelease_mixed_nocosmo.h5 --event_name GW170809 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False --resume True
python plot_backpop.py --samples ./results/GW170809/backpop_fixed_kicks_minimal.npz --event_name GW170809 --nsamples 10000

python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW170814_103043_PEDataRelease_mixed_nocosmo.h5 --event_name GW170814 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False
python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW170814_103043_PEDataRelease_mixed_nocosmo.h5 --event_name GW170814 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False --resume True
python plot_backpop.py --samples ./results/GW170814/backpop_fixed_kicks_minimal.npz --event_name GW170814 --nsamples 10000

python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW170818_022509_PEDataRelease_mixed_nocosmo.h5 --event_name GW170818 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False
python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW170818_022509_PEDataRelease_mixed_nocosmo.h5 --event_name GW170818 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False --resume True
python plot_backpop.py --samples ./results/GW170818/backpop_fixed_kicks_minimal.npz --event_name GW170818 --nsamples 10000

python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW170823_131358_PEDataRelease_mixed_nocosmo.h5 --event_name GW170823 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False
python run_backpop.py --samples_path ../../GWTC-PESamples/gwtc3_bbh_1peryr/IGWN-GWTC2p1-v2-GW170823_131358_PEDataRelease_mixed_nocosmo.h5 --event_name GW170823 --config_name backpop_fixed_kicks_minimal --nwalkers 256 --nsteps 10000 --redshift_likelihood False --resume True
python plot_backpop.py --samples ./results/GW170823/backpop_fixed_kicks_minimal.npz --event_name GW170823 --nsamples 10000