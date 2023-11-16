import numpy as np
from scipy.integrate import simpson
from interaction import dsigma_dEr_nu_e_ES

Enu_list = np.linspace(2.9,61,101) 
Ee_list = np.linspace(0,61,101)
nu_list = ['nue','nuebar','numu','numubar']

for nu_alpha in nu_list:
    s = []
    for Enu in Enu_list:
        s.append(simpson(dsigma_dEr_nu_e_ES(Enu, Ee_list, nu_alpha), Ee_list))
    file_name = 'data/interp_sigma_ES_pi_t/sigma_ES_integrated_%s.npy'%nu_alpha
    np.save(file_name, s)