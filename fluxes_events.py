import numpy as np
# import matplotlib.pyplot as plt
import math
from scipy.special import gamma
import os
import pickle

#Pedro Dedin code to collect the data from files
nu_types = ['nubar_e','nu_e','nu_x']
folder_std ='data/neutrinos-LS220-s27.0c/' #standard simulation: all calculations until 01/09 were based on this
t_data, E_data, E2_data, Lum_data, alpha_data = [],[],[],[],[]

model = 1
models = ['LS220-s11.2c','LS220-s27.0c','LS220-s27.0co','Shen-s11.2c']

#loop over flavors
for i in range(3):
    path = 'data/neutrinos-' + models[model] + '/neutrino_signal_'+ nu_types[i] + '-' + models[model] + '.data'
    # path = 'data/neutrinos-' + model + '/neutrino_signal_'+ nu_types[i] + '-' + model + '.data'
    data = np.loadtxt(path, skiprows=5)
    t = data[:,0]
    Lum = data[:,1]
    E = data[:,2]
    E2 = data[:,3]
    alpha = (2*E - E2)/(E2 - E**2)
    t_data.append(t)
    E_data.append(E)
    E2_data.append(E2)
    Lum_data.append(Lum)
    alpha_data.append(alpha)


#https://arxiv.org/pdf/astro-ph/0308228.pdf
def flux(E, L, D, Emean, Emean_square):
    alpha = (2 * Emean**2 - Emean_square)/(Emean_square - Emean**2)
    c = (alpha+1)**(-(alpha+1)) * gamma(alpha+1) * Emean
    #spectrum
    f = 1/c * (E/Emean)**alpha * np.exp(-(alpha+1)*E/Emean) #1/MeV
    erg_to_MeV = 624151
    kpc_to_cm = 3.086e21
    D = D * kpc_to_cm
    L = L * erg_to_MeV * 10**51
    #flux
    flux = f * L/Emean/(4*math.pi*D**2) #1/MeV 1/MeV MeV/s 1/cm^2 = 1/(MeV cm^2 s)
    return flux
