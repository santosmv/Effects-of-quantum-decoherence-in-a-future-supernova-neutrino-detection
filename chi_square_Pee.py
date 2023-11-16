import numpy as np
from scipy.integrate import simps
from importlib import reload
from rate_Pee import dndE
from read_database import read_bins

def reload_config(module_name):
    reload(module_name)

def chi_square(Pee, Pee_bar, a):
    
    import import_config
    reload_config(import_config)
    from import_config import config_list

    option_std = config_list[5]
    option_th = 'Pee'

    if 'NH' in option_std:
        hierarchy = 'NH'
    elif 'IH' in option_std:
        hierarchy = 'IH'

    par = Pee, Pee_bar

    detector = config_list[2]

    if detector == 'dune_hk_jn':
        n_dune = []
        n_dune_Pee = []
        n_hk_es = []
        n_hk_ibd = []
        n_hk_es_Pee = []
        n_hk_ibd_Pee = []
        n_jn = []
        n_jn_Pee = []

        #DUNE
        Enu_bins_dune = read_bins('dune', hierarchy)
        Enu_list = np.linspace(4.5, 50, 101)
        for i in range(len(Enu_bins_dune)-1):
            bin_i = i+1
            # Enubin = np.linspace(Enu_bins_dune[i], Enu_bins_dune[i+1], 101)
            n_dune.append(simps(dndE(Enu_list, option_std, 'DUNE', '', par, bin_i), Enu_list))
            n_dune_Pee.append(simps(dndE(Enu_list, option_th, 'DUNE', '', par, bin_i), Enu_list))

        #HK
        #ES
        Enu_bins_hk = read_bins('hk', hierarchy)
        Enu_list = np.linspace(3, 50, 101)
        for i in range(len(Enu_bins_hk)-1):
            bin_i = i+1
            #ES
            #Enubin = np.linspace(Enu_bins_hk[i], Enu_bins_hk[i+1], 101)
            n_hk_es.append(simps(dndE(Enu_list, option_std, 'HK', 'ES', par, bin_i), Enu_list))
            n_hk_es_Pee.append(simps(dndE(Enu_list, option_th, 'HK', 'ES', par, bin_i), Enu_list))    
            #IBD
            n_hk_ibd.append(simps(dndE(Enu_list, option_std, 'HK', 'IBD', par, bin_i), Enu_list))
            n_hk_ibd_Pee.append(simps(dndE(Enu_list, option_th, 'HK', 'IBD', par, bin_i), Enu_list))

        #JUNO
        Enu_bins_jn = read_bins('jn', hierarchy)
        Enu_list = np.linspace(3, 50, 101)
        for i in range(len(Enu_bins_jn)-1):
            bin_i = i+1
            #Enubin = np.linspace(Enu_bins_jn[i], Enu_bins_jn[i+1], 101)
            n_jn.append(simps(dndE(Enu_list, option_std, 'JN', '', par, bin_i), Enu_list))
            n_jn_Pee.append(simps(dndE(Enu_list, option_th, 'JN', '', par, bin_i), Enu_list))

        #chi DUNE
        chi_list_dune = []
        for i in range(len(n_dune)):
            nstd = n_dune[i]
            nth = n_dune_Pee[i]
            if nth == 0:
                chi_dune = 1e9
            else:
                chi_dune = (nstd - (1+a)*nth)**2/nth
            chi_list_dune.append(chi_dune)

        #chi HK
        #ES
        chi_list_hk_es = []
        for i in range(len(n_hk_es)):
            nstd = n_hk_es[i]
            nth = n_hk_es_Pee[i]
            if nth == 0:
                chi_hk = 1e9
            else:
                chi_hk = (nstd - (1+a)*nth)**2/nth
            chi_list_hk_es.append(chi_hk)
        #IBD
        chi_list_hk_ibd = []
        for i in range(len(n_hk_ibd)-1):
            nstd = n_hk_ibd[i]
            nth = n_hk_ibd_Pee[i]
            if nth == 0:
                chi_hk = 1e9
            else:
                chi_hk = (nstd - (1+a)*nth)**2/nth
            chi_list_hk_ibd.append(chi_hk)

        #chi JUNO
        chi_list_jn = []
        for i in range(len(n_jn)):
            nstd = n_jn[i]
            nth = n_jn_Pee[i]
            if nth == 0:
                chi_jn = 1e9
            else:
                chi_jn = (nstd - (1+a)*nth)**2/nth
            chi_list_jn.append(chi_jn)

        sigma_a = 0.40 #uncertainty in the flux

        chi_tot = sum(chi_list_dune) + a**2/sigma_a**2 + sum(chi_list_hk_es) + sum(chi_list_hk_ibd) + sum(chi_list_jn)

        # print('chi =', chi_tot)
        return chi_tot

    else:
        if detector == 'hk':
            Enu_list = np.linspace(3, 50, 101)
            channels = ['IBD','ES']
        elif detector == 'jn':
            Enu_list = np.linspace(3, 50, 101)
            channels = ['']
        elif detector == 'dune':
            Enu_list = np.linspace(4.5, 50, 101)
            channels = ['']
                
        n_list = []
        n_Pee_list = []
        chi_list = []
        for channel in channels:
            Enu_bins = read_bins(detector, hierarchy)
            for i in range(len(Enu_bins)-1):
                bin_i = i+1
                n = simps(dndE(Enu_list, option_std, detector.upper(), channel, par, bin_i), Enu_list)
                noqs = simps(dndE(Enu_list, option_th, detector.upper(), channel, par, bin_i), Enu_list)
                n_list.append(n)
                n_Pee_list.append(noqs)

            #chi
            for i in range(len(n_list)):
                nstd = n_list[i]
                nth = n_Pee_list[i]
                if nth == 0:
                    chi = 1e9
                else:
                    chi = (nstd - (1+a)*nth)**2/nth
                chi_list.append(chi)
        
        sigma_a = 0.40 #uncertainty in the flux
        chi_tot = sum(chi_list) + a**2/sigma_a**2

        # print('Chi Tot=',chi_tot)
        return chi_tot

