import numpy as np
from integ_fluxes import interp_fluxes
from scipy.interpolate import interp1d
from interaction import sigma_nue_Ar, sigma_IBD
from flavor_conversion import P_NH_nue, P_IH_nue, P_IH_antie, P_NH_antie, Pie_earth_interp_func
from menu import config_list
from read_database import read_work_function
from flavor_conversion import U3
import math
import sqlite3
import platform
import socket

pc_name = socket.gethostname()
#DOCS: https://docs.python.org/3/library/sqlite3.html
if platform.system() == 'Windows':
    connection = sqlite3.connect('data/database_windows.db')
else:
    if pc_name == 'neutrino7':
        data_path = 'data/database_neutrino7.db'
    elif pc_name == 'drcpc65':
        data_path = 'data/database_drcpc65.db'
    elif pc_name == 'Marconis':
        data_path = 'data/database_marconis.db'
    connection = sqlite3.connect(data_path)
cursor = connection.cursor()

me = 0.510998950 #MeV
# D = config_list[1] #kpc
thetaz = config_list[6]
Ne_HK = 1.25e35 #from https://arxiv.org/abs/2011.10933
NAr_DUNE = 6.03e32
Ne_HK = 1.25e35 #from https://arxiv.org/abs/2011.10933
Np_HK = 2.5e34 #from https://arxiv.org/abs/2011.10933
NAr_DUNE = 6.03e32
Np_JUNO = 1.21e33 #from https://arxiv.org/abs/2011.10933
efficiency_HK_ES = 0.6
efficiency_HK_IBD = 0.6
efficiency_JUNO_IBD = 0.5

def dndE(Enu, mix, detector, channel, par, bin_i):

    if 'NH' in config_list[5]:
        hierarchy = 'NH'
    elif 'IH' in config_list[5]:
        hierarchy = 'IH'
    
        
    model = config_list[0]
    D = config_list[1] #kpc
    thetaz = config_list[6]

    #integrated fluxes
    Fe0_bar_i, Fe0_i, Fx0_i = interp_fluxes(model, Enu)
    kpc_to_cm = 3.086e21
    D_cm = D * kpc_to_cm
    Fe0_bar = Fe0_bar_i/(4*math.pi*D_cm**2)
    Fe0 = Fe0_i/(4*math.pi*D_cm**2)
    Fx0 = Fx0_i/(4*math.pi*D_cm**2)

    Dm21 = 7.42e-5 #eV² +0.21 -0.20
    t12 = 33.45 #+0.77 -0.75 NuFIT 2021 https://arxiv.org/pdf/2111.03086.pdf
    t12 = t12*math.pi/180
    Pee = 1
    Pee_bar = 1
    #mixed fluxes
    if mix == 'no':
        Pee = 1
        Fe = Fe0
        Fx = Fx0
    
    elif mix == 'NH':
        Pee = P_NH_nue()
        Pee_bar = P_NH_antie()
    
    elif mix == 'IH':
        Pee = P_IH_nue()
        Pee_bar = P_IH_antie()
    
    elif mix == 'NH(m)':
        Pee = abs(U3[0,2])**2
        w21 = Dm21/(Enu*1e6)
        Pee_bar = Pie_earth_interp_func(w21, t12, 'nuebar', 'NH', thetaz, detector, 1)

    elif mix == 'IH(m)':
        Pee_bar = abs(U3[0,2])**2
        w21 = Dm21/(Enu*1e6)
        Pee = Pie_earth_interp_func(w21, t12, 'nue', 'IH', thetaz, detector, 2)
    
    elif mix == 'Pee':
        Pee, Pee_bar = par

    Fe = Pee * Fe0 + (1 - Pee) * Fx0
    Fe_bar = Pee_bar * Fe0_bar + (1 - Pee_bar) * Fx0
    Fx = 1/4 * ((1 - Pee) * Fe0 + (2 + Pee + Pee_bar) * Fx0 + (1 - Pee_bar) * Fe0_bar)

    if detector == 'HK':
        if channel == 'ES':
            #work function has effective cross section
            Enu_interp_hk_es, wnue, wnumu, wantinue, wantinumu = read_work_function(bin_i, 'hk', 'elastic', hierarchy, 0)
            wnue_func = interp1d(Enu_interp_hk_es, wnue)
            wnumu_func = interp1d(Enu_interp_hk_es, wnumu)
            wantinue_func = interp1d(Enu_interp_hk_es, wantinue)
            wantinumu_func = interp1d(Enu_interp_hk_es, wantinumu)
            df = Fe * wnue_func(Enu) + Fe_bar * wantinue_func(Enu) + 2*Fx * wnumu_func(Enu) + 2*Fx * wantinumu_func(Enu)
            d = Ne_HK * efficiency_HK_ES * df
            return d
            
        elif channel == 'IBD':
            Enu_interp_hk_ibd, w_ibd = read_work_function(bin_i, 'hk', 'ibd', hierarchy, 0)
            w_ibd_func = interp1d(Enu_interp_hk_ibd, w_ibd)
            df = Fe_bar * sigma_IBD(Enu)
            d = Np_HK * efficiency_HK_IBD * df * w_ibd_func(Enu)
            return d
    
    elif detector == 'JN':
        Enu_interp_jn_ibd, w_ibd = read_work_function(bin_i, 'jn', 'ibd', hierarchy, 0)
        w_jn_ibd_func = np.interp(Enu, Enu_interp_jn_ibd, w_ibd)
        df = Fe_bar * sigma_IBD(Enu)
        d = Np_JUNO * efficiency_JUNO_IBD * df * w_jn_ibd_func
        return d

    elif detector == 'DUNE':
        Enu_interp_dune, w_nueAr = read_work_function(bin_i, 'dune', 'nue-Ar', hierarchy, 0)
        w_nueAr_func = interp1d(Enu_interp_dune, w_nueAr)
        df = Fe * sigma_nue_Ar(Enu)
        d = NAr_DUNE * dune_efficiency(Enu) * df * w_nueAr_func(Enu)
        return d


#taken from figure 7 of https://arxiv.org/pdf/2008.06647.pdf (chosen 5MeV for deposited energy threshold)
def dune_efficiency(Enu):
    data_file = np.loadtxt('data/dune_eff_5MeV.dat', delimiter=' ')
    Energy = data_file[:,0]
    eff = data_file[:,1]
    eff_func = interp1d(Energy,eff,kind='cubic',fill_value='extrapolate')
    return eff_func(Enu)
