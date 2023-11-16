from math import pi
import numpy as np
from pathlib import Path
from rate_pi_t import dndtdE
from scipy.integrate import simpson

detector_list = ['DUNE','HK','HK','HK','HK','HK']
channels_list = ['','ES','ES','ES','ES','IBD']
nu_list = ['nue','nue','nuebar','numu','numubar','nuebar']
mix_list = ['NH','IH']

Deltam21 = 7.42e-5 #eV² +0.21 -0.20
theta12 = 33.45 #+0.77 -0.75 NuFIT 2021 https://arxiv.org/pdf/2111.03086.pdf
theta12 = theta12*pi/180

n = 401

t_list = np.linspace(0, 0.030, n)
t12_list = np.linspace(theta12-0.75*pi/180, theta12+0.77*pi/180, n)

t_listg, t12_listg  = np.meshgrid(t_list, t12_list, indexing='ij', sparse=True)

dndtdE_vec = np.vectorize(dndtdE)

for i in range(len(detector_list)):
    for mix in mix_list:
        if detector_list[i] == 'HK':
            file_name = 'interpolation_pi_t/dndt_%s_%s_%s.npy'%(detector_list[i], channels_list[i], nu_list[i])
        elif detector_list[i] == 'DUNE':
            file_name = 'interpolation_pi_t/dndt_%s.npy'%detector_list[i]
        path = Path(file_name)

        if detector_list[i] == 'DUNE':
            Eth = 4.5
        elif detector_list[i] == 'HK':
            Eth = 3

        Enu = np.linspace(Eth,51,201)

        if path.is_file():
            print('Jumping ' + file_name + '\n')
            continue
        else:
            print('Working on ' + file_name + '...')
            data = simpson(dndtdE_vec(Enu, t_listg, mix, detector_list[i], channels_list[i], nu_list[i], t12_listg), Enu)
            np.save(file_name, data)
            print('Saved ' + file_name + '\n')