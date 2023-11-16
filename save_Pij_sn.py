from sn_matter_qd.Pij_sn_conserved_E import Pij_SN_surface_E
from math import pi
import numpy as np
from time import time
from pathlib import Path
import sqlite3
import socket
import platform
import os
from random import randint

ti = time()
Deltam21 = 7.42e-5 #eV² +0.21 -0.20
theta12 = 33.45 #+0.77 -0.75 NuFIT 2021 https://arxiv.org/pdf/2111.03086.pdf
theta12 = theta12*pi/180

nE, ng = 201,201

mix_list = ['NH','IH']
nu_nubar_list = ['nue','nuebar']
n_list = [0,2,5/2]
ij_list = ['31','32','33']

mix_list = ['NH']
nu_nubar_list = ['nue']
n_list = [0]
ij_list = ['31']


E_list = np.linspace(0.1, 61, nE)

for ij in ij_list:
    for n in n_list:
        for nu_nubar in nu_nubar_list:
            for mix in mix_list:
                if n == 0:
                    file_name = 'saved_Pij_sn_matter/P%s_%s_%s_n%i_%iE_%ig.npy'%(ij, mix, nu_nubar, n, nE, ng)
                    g_list = np.linspace(-22, -10, ng)
                if n == 2:
                    file_name = 'saved_Pij_sn_matter/P%s_%s_%s_n%i_%iE_%ig.npy'%(ij, mix, nu_nubar, n, nE, ng)
                    g_list = np.linspace(-35, -19, ng)
                elif n == 2.5:
                    file_name = 'saved_Pij_sn_matter/P%s_%s_%s_n%.1f_%iE_%ig.npy'%(ij, mix, nu_nubar, n, nE, ng)
                    g_list = np.linspace(-40, -20, ng)
                path = Path(file_name)

                if path.is_file():
                    print('Jumping ' + file_name + '\n')
                    continue
                else:
                    ti = time()
                    print('Working on ' + file_name + '...')
                    Pij = []
                    for E in E_list:
                        Pij_g = []
                        for g in g_list:
                            Pij_g.append(Pij_SN_surface_E(n, E, g, mix, ij, nu_nubar))
                        print('Run %i of %i: %i, %.1f'%((len(Pij)+1), nE, g, E))
                        Pij.append(Pij_g)
                    
                    print('Time: %.2f minutes or %.2f hours spent.'%((time()-ti)/60, (time()-ti)/3600))
                    np.save(file_name, Pij)
                    print('Saved ' + file_name + '\n')
