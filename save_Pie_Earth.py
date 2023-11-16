from Pie_earth import Pje_Earth
from math import pi
import numpy as np
from time import time
from pathlib import Path

ti = time()
Deltam21 = 7.42e-5 #eV² +0.21 -0.20
theta12 = 33.45 #+0.77 -0.75 NuFIT 2021 https://arxiv.org/pdf/2111.03086.pdf
theta12 = theta12*pi/180

dune_angles = [180,160,140,120]
hk_angles = [97.84421607943858, 95.47306275942896, 129.64136416059688, 129.37887605122702]
juno_angles = [105.7170051252261, 120.75643738758598, 146.56494382389732, 146.62548196139585]

######################### OLD WRONG ###############################
# hk_angles = [138.79549761627743, 118.79549761627743, 98.79549761627743, 78.79549761627743]
# juno_angles = [127.01649647296367, 107.01649647296367]
###################################################################

thetaz_detectors = [dune_angles, hk_angles, juno_angles]
detectors_list = ['hk','juno']

nt, nw = 401,401

d21_i = Deltam21 - 0.2e-5
d21_f = Deltam21 + 0.21e-5

mix_list = ['NH','IH']
j_list = ['1','2']
nu_nubar_list = ['nue','nuebar']

idd = int(input(' 1.DUNE\n 2.HK\n 3.JUNO\n Type:'))
idm = int(input(' 1.NH\n 2.IH\n Type:'))

thetaz_list = thetaz_detectors[idd-1]
mix = mix_list[idm-1]

w21_list = np.linspace(d21_i/60e6, d21_f/1e6, nw)
t12_list = np.linspace(theta12-0.75*pi/180, theta12+0.77*pi/180, nt)
w21_listg, t12_listg  = np.meshgrid(w21_list, t12_list, indexing='ij', sparse=True)

for thetaz in thetaz_list:
    for j in j_list:
        for nu_nubar in nu_nubar_list:
            file_name = 'regeneration/P%se_%s_%s_%.1fz_%sth_%sw21.npy'%(j, mix, nu_nubar, thetaz, nt, nw)
            path = Path(file_name)

            if path.is_file():
                print('Jumping ' + file_name + '\n')
                continue
            else:
                print('Working on ' + file_name + '...')
                data = Pje_Earth(w21_listg, thetaz, mix, j, nu_nubar, t12_listg)
                np.save(file_name, data)
                print('Saved ' + file_name + '\n')