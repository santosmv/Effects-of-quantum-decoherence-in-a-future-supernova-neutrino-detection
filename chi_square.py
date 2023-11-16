import numpy as np
from scipy.integrate import simps, simpson
from importlib import reload
from rate import dndE
from read_database import read_bins
from numba import jit, njit
from time import time
from bins import bins_loss

def reload_config(module_name):
    reload(module_name)

#Here we list individual and combined chi squares

#################### MSC diagonal model #####################

def chi_square(g3_exp, g8_exp, theta12, a):
    import import_config
    reload_config(import_config)
    from import_config import config_list

    option_std = config_list[5][:2]
    option_th = config_list[5][6:]

    if 'NH' in option_std:
        hierarchy = 'NH'
    elif 'IH' in option_std:
        hierarchy = 'IH'
    
    if 'IH' in config_list[5] and 'NH' in config_list[5]:
        hierarchy = 'NH'

    par = g3_exp, g8_exp, theta12

    detector = config_list[2]

    if detector == 'dune_hk_jn':
        n_dune = []
        n_dune_oqs = []
        n_hk = []
        n_hk_oqs = []
        n_hk_es = []
        n_hk_ibd = []
        n_hk_es_oqs = []
        n_hk_ibd_oqs = []
        n_jn = []
        n_jn_oqs = []

        #DUNE
        Enu_bins_dune = read_bins('dune', hierarchy)
        Enu_list = np.linspace(4.5, 60, 101)
        for i in range(len(Enu_bins_dune)-1):
            bin_i = i+1
            n_dune.append(simps(dndE(Enu_list, option_std, 'DUNE', '', par, bin_i), Enu_list))
            n_dune_oqs.append(simps(dndE(Enu_list, option_th, 'DUNE', '', par, bin_i), Enu_list))

        #HK
        #ES
        Enu_bins_hk = read_bins('hk', hierarchy)
        Enu_list = np.linspace(3, 60, 101)
        for i in range(len(Enu_bins_hk)-1):
            bin_i = i+1
            #ES
            # n_hk_es.append(simps(dndE(Enu_list, option_std, 'HK', 'ES', par, bin_i), Enu_list))
            # n_hk_es_oqs.append(simps(dndE(Enu_list, option_th, 'HK', 'ES', par, bin_i), Enu_list))    
            # #IBD
            # n_hk_ibd.append(simps(dndE(Enu_list, option_std, 'HK', 'IBD', par, bin_i), Enu_list))
            # n_hk_ibd_oqs.append(simps(dndE(Enu_list, option_th, 'HK', 'IBD', par, bin_i), Enu_list))
            #ES
            n_hk_es = simps(dndE(Enu_list, option_std, 'HK', 'ES', par, bin_i), Enu_list)
            n_hk_es_oqs = simps(dndE(Enu_list, option_th, 'HK', 'ES', par, bin_i), Enu_list)
            #IBD
            n_hk_ibd = simps(dndE(Enu_list, option_std, 'HK', 'IBD', par, bin_i), Enu_list)
            n_hk_ibd_oqs = simps(dndE(Enu_list, option_th, 'HK', 'IBD', par, bin_i), Enu_list)
            n_hk.append(n_hk_es + n_hk_ibd)
            n_hk_oqs.append(n_hk_es_oqs + n_hk_ibd_oqs)

        #JUNO
        Enu_bins_jn = read_bins('jn', hierarchy)
        Enu_list = np.linspace(3, 60, 101)
        for i in range(len(Enu_bins_jn)-1):
            bin_i = i+1
            n_jn.append(simps(dndE(Enu_list, option_std, 'JN', '', par, bin_i), Enu_list))
            n_jn_oqs.append(simps(dndE(Enu_list, option_th, 'JN', '', par, bin_i), Enu_list))

        # print(n_hk)
        # print(n_hk_oqs)
        # print(n_dune)
        # print(n_dune_oqs)
        # print(n_jn)
        # print(n_jn_oqs)

        #chi DUNE
        chi_list_dune = []
        for i in range(len(n_dune)):
            nstd = n_dune[i]
            nth = n_dune_oqs[i]
            if nth == 0:
                chi_dune = 1e9
            else:
                chi_dune = (nstd - (1+a)*nth)**2/nth
            chi_list_dune.append(chi_dune)

        #chi HK
        chi_list_hk = []
        for i in range(len(n_hk)):
            nstd = n_hk[i]
            nth = n_hk_oqs[i]
            if nth == 0:
                chi_hk = 1e9
            else:
                chi_hk = (nstd - (1+a)*nth)**2/nth
            chi_list_hk.append(chi_hk)
        
        #ES
        # chi_list_hk_es = []
        # for i in range(len(n_hk_es)):
        #     nstd = n_hk_es[i]
        #     nth = n_hk_es_oqs[i]
        #     if nth == 0:
        #         chi_hk = 1e9
        #     else:
        #         chi_hk = (nstd - (1+a)*nth)**2/nth
        #     chi_list_hk_es.append(chi_hk)
        #IBD
        # chi_list_hk_ibd = []
        # for i in range(len(n_hk_ibd)-1):
        #     nstd = n_hk_ibd[i]
        #     nth = n_hk_ibd_oqs[i]
        #     if nth == 0:
        #         chi_hk = 1e9
        #     else:
        #         chi_hk = (nstd - (1+a)*nth)**2/nth
        #     chi_list_hk_ibd.append(chi_hk)

        #chi JUNO
        chi_list_jn = []
        for i in range(len(n_jn)):
            nstd = n_jn[i]
            nth = n_jn_oqs[i]
            if nth == 0:
                chi_jn = 1e9
            else:
                chi_jn = (nstd - (1+a)*nth)**2/nth
            chi_list_jn.append(chi_jn)

        sigma_a = 0.40 #uncertainty in the flux

        # chi_tot = sum(chi_list_dune) + a**2/sigma_a**2 + sum(chi_list_hk_es) + sum(chi_list_hk_ibd) + sum(chi_list_jn)
        chi_tot = sum(chi_list_dune) + a**2/sigma_a**2 + sum(chi_list_hk) + sum(chi_list_jn)

        # print('chi =', chi_tot)
        return chi_tot

    else:
        if detector == 'hk':
            Enu_list = np.linspace(3, 60, 101)
            channels = ['IBD','ES']
        elif detector == 'jn':
            Enu_list = np.linspace(3, 60, 101)
            channels = ['']
        elif detector == 'dune':
            Enu_list = np.linspace(4.5, 60, 101)
            channels = ['']

        n_list = []
        n_oqs_list = []
        chi_list = []
        Enu_bins = read_bins(detector, hierarchy)
        for i in range(len(Enu_bins)-1):
            bin_i = i+1
            n = 0
            noqs = 0
            for channel in channels:
                n += simps(dndE(Enu_list, option_std, detector.upper(), channel, par, bin_i), Enu_list)
                noqs += simps(dndE(Enu_list, option_th, detector.upper(), channel, par, bin_i), Enu_list)
            n_list.append(n)
            n_oqs_list.append(noqs)

        #chi
        for i in range(len(n_list)):
            nstd = n_list[i]
            nth = n_oqs_list[i]
            if nth == 0:
                chi = 1e9
            else:
                chi = (nstd - (1+a)*nth)**2/nth
            chi_list.append(chi)
        
        sigma_a = 0.40 #uncertainty in the flux
        chi_tot = sum(chi_list) + a**2/sigma_a**2

        # print('Chi Tot=',chi_tot)
        return chi_tot

# import math
# theta12 = 33.45 #+0.77 -0.75 NuFIT 2021 https://arxiv.org/pdf/2111.03086.pdf
# theta12 = theta12*math.pi/180
# print(chi_square(g3_exp=-28, g8_exp=-28, theta12=theta12, a=0.4))

############ matter-effects ##############

def chi_square_m(g3_exp, g8_exp, theta12, Deltam21, a):
    import import_config
    reload_config(import_config)
    from import_config import config_list

    option_std = config_list[5][:5]
    option_th = config_list[5][9:]

    if 'NH' in option_std:
        hierarchy = 'NH'
    elif 'IH' in option_std:
        hierarchy = 'IH'
    
    if 'IH' in config_list[5] and 'NH' in config_list[5]:
        hierarchy = 'NH'

    par = g3_exp, g8_exp, theta12, Deltam21

    detector = config_list[2]

    if detector == 'dune_hk_jn':
        n_dune = []
        n_dune_oqs = []
        n_hk_es = []
        n_hk_ibd = []
        n_hk = []
        n_hk_oqs = []
        n_hk_es_oqs = []
        n_hk_ibd_oqs = []
        n_jn = []
        n_jn_oqs = []

        #DUNE
        Enu_bins_dune = read_bins('dune', hierarchy)
        Enu_list = np.linspace(4.5, 60, 101)
        for i in range(len(Enu_bins_dune)-1):
            bin_i = i+1
            n_dune.append(simps(dndE(Enu_list, option_std, 'DUNE', '', par, bin_i), Enu_list))
            n_dune_oqs.append(simps(dndE(Enu_list, option_th, 'DUNE', '', par, bin_i), Enu_list))

        #HK
        #ES
        Enu_bins_hk = read_bins('hk', hierarchy)
        Enu_list = np.linspace(3, 60, 101)
        for i in range(len(Enu_bins_hk)-1):
            bin_i = i+1
            ############### splitting chi in channels ###############
            #ES
            #Enubin = np.linspace(Enu_bins_hk[i], Enu_bins_hk[i+1], 101)
            # n_hk_es.append(simps(dndE(Enu_list, option_std, 'HK', 'ES', par, bin_i), Enu_list))
            # n_hk_es_oqs.append(simps(dndE(Enu_list, option_th, 'HK', 'ES', par, bin_i), Enu_list))    
            # #IBD
            # n_hk_ibd.append(simps(dndE(Enu_list, option_std, 'HK', 'IBD', par, bin_i), Enu_list))
            # n_hk_ibd_oqs.append(simps(dndE(Enu_list, option_th, 'HK', 'IBD', par, bin_i), Enu_list))
            n_hk_es = simps(dndE(Enu_list, option_std, 'HK', 'ES', par, bin_i), Enu_list)
            n_hk_es_oqs = simps(dndE(Enu_list, option_th, 'HK', 'ES', par, bin_i), Enu_list)
            #IBD
            n_hk_ibd = simps(dndE(Enu_list, option_std, 'HK', 'IBD', par, bin_i), Enu_list)
            n_hk_ibd_oqs = simps(dndE(Enu_list, option_th, 'HK', 'IBD', par, bin_i), Enu_list)
            n_hk.append(n_hk_es + n_hk_ibd)
            n_hk_oqs.append(n_hk_es_oqs + n_hk_ibd_oqs)

        #JUNO
        Enu_bins_jn = read_bins('jn', hierarchy)
        Enu_list = np.linspace(3, 60, 101)
        for i in range(len(Enu_bins_jn)-1):
            bin_i = i+1
            n_jn.append(simps(dndE(Enu_list, option_std, 'JN', '', par, bin_i), Enu_list))
            n_jn_oqs.append(simps(dndE(Enu_list, option_th, 'JN', '', par, bin_i), Enu_list))

        #chi DUNE
        chi_list_dune = []
        for i in range(len(n_dune)):
            nstd = n_dune[i]
            nth = n_dune_oqs[i]
            if nth == 0:
                chi_dune = 1e9
            else:
                chi_dune = (nstd - (1+a)*nth)**2/nth
            chi_list_dune.append(chi_dune)

        #chi HK
        chi_list_hk = []
        for i in range(len(n_hk)):
            nstd = n_hk[i]
            nth = n_hk_oqs[i]
            if nth == 0:
                chi_hk = 1e9
            else:
                chi_hk = (nstd - (1+a)*nth)**2/nth
            chi_list_hk.append(chi_hk)

        ############### splitting chi in channels ###############
        #ES
        # chi_list_hk_es = []
        # for i in range(len(n_hk_es)):
        #     nstd = n_hk_es[i]
        #     nth = n_hk_es_oqs[i]
        #     if nth == 0:
        #         chi_hk = 1e9
        #     else:
        #         chi_hk = (nstd - (1+a)*nth)**2/nth
        #     chi_list_hk_es.append(chi_hk)
        # #IBD
        # chi_list_hk_ibd = []
        # for i in range(len(n_hk_ibd)-1):
        #     nstd = n_hk_ibd[i]
        #     nth = n_hk_ibd_oqs[i]
        #     if nth == 0:
        #         chi_hk = 1e9
        #     else:
        #         chi_hk = (nstd - (1+a)*nth)**2/nth
        #     chi_list_hk_ibd.append(chi_hk)

        #chi JUNO
        chi_list_jn = []
        for i in range(len(n_jn)):
            nstd = n_jn[i]
            nth = n_jn_oqs[i]
            if nth == 0:
                chi_jn = 1e9
            else:
                chi_jn = (nstd - (1+a)*nth)**2/nth
            chi_list_jn.append(chi_jn)

        sigma_a = 0.40 #uncertainty in the flux

        # chi_tot = sum(chi_list_dune) + a**2/sigma_a**2 + sum(chi_list_hk_es) + sum(chi_list_hk_ibd) + sum(chi_list_jn)
        chi_tot = sum(chi_list_dune) + a**2/sigma_a**2 + sum(chi_list_hk) + sum(chi_list_jn)

        # print('chi =', chi_tot)
        return chi_tot

    else:
        if detector == 'hk':
            Enu_list = np.linspace(3, 60, 101)
            channels = ['IBD','ES']
        elif detector == 'jn':
            Enu_list = np.linspace(3, 60, 101)
            channels = ['']
        elif detector == 'dune':
            Enu_list = np.linspace(4.5, 60, 101)
            channels = ['']
        
        n_list = []
        n_oqs_list = []
        chi_list = []
        Enu_bins = read_bins(detector, hierarchy)
        for i in range(len(Enu_bins)-1):
            bin_i = i+1
            n = 0
            noqs = 0
            for channel in channels:
                n += simps(dndE(Enu_list, option_std, detector.upper(), channel, par, bin_i), Enu_list)
                noqs += simps(dndE(Enu_list, option_th, detector.upper(), channel, par, bin_i), Enu_list)
            n_list.append(n)
            n_oqs_list.append(noqs)

            #chi
            for i in range(len(n_list)):
                nstd = n_list[i]
                nth = n_oqs_list[i]
                if nth == 0:
                    chi = 1e9
                else:
                    chi = (nstd - (1+a)*nth)**2/nth
                chi_list.append(chi)
        
        sigma_a = 0.40 #uncertainty in the flux
        chi_tot = sum(chi_list) + a**2/sigma_a**2

        # print('Chi Tot=',chi_tot)
        return chi_tot



#################### MSC diagoonal model with D marginalized #####################

def chi_square_D(D, g_exp, theta12, a):
    import import_config
    reload_config(import_config)
    from import_config import config_list

    option_std = config_list[5][:2]
    option_th = config_list[5][6:]

    if 'NH' in option_std:
        hierarchy = 'NH'
    elif 'IH' in option_std:
        hierarchy = 'IH'
    
    if 'IH' in config_list[5] and 'NH' in config_list[5]:
        hierarchy = 'NH'

    par = D, g_exp, theta12

    detector = config_list[2]

    if detector == 'dune_hk_jn':
        n_dune = []
        n_dune_oqs = []
        n_hk = []
        n_hk_oqs = []
        n_hk_es = []
        n_hk_ibd = []
        n_hk_es_oqs = []
        n_hk_ibd_oqs = []
        n_jn = []
        n_jn_oqs = []

        #DUNE
        Enu_bins_dune = read_bins('dune', hierarchy)
        Enu_list = np.linspace(4.5, 60, 101)
        for i in range(len(Enu_bins_dune)-1):
            bin_i = i+1
            n_dune.append(simps(dndE(Enu_list, option_std, 'DUNE', '', par, bin_i), Enu_list))
            n_dune_oqs.append(simps(dndE(Enu_list, option_th, 'DUNE', '', par, bin_i), Enu_list))

        #HK
        #ES
        Enu_bins_hk = read_bins('hk', hierarchy)
        Enu_list = np.linspace(3, 60, 101)
        for i in range(len(Enu_bins_hk)-1):
            bin_i = i+1
            #ES
            n_hk_es = simps(dndE(Enu_list, option_std, 'HK', 'ES', par, bin_i), Enu_list)
            n_hk_es_oqs = simps(dndE(Enu_list, option_th, 'HK', 'ES', par, bin_i), Enu_list)
            #IBD
            n_hk_ibd = simps(dndE(Enu_list, option_std, 'HK', 'IBD', par, bin_i), Enu_list)
            n_hk_ibd_oqs = simps(dndE(Enu_list, option_th, 'HK', 'IBD', par, bin_i), Enu_list)
            n_hk.append(n_hk_es + n_hk_ibd)
            n_hk_oqs.append(n_hk_es_oqs + n_hk_ibd_oqs)

        #JUNO
        Enu_bins_jn = read_bins('jn', hierarchy)
        Enu_list = np.linspace(3, 60, 101)
        for i in range(len(Enu_bins_jn)-1):
            bin_i = i+1
            n_jn.append(simps(dndE(Enu_list, option_std, 'JN', '', par, bin_i), Enu_list))
            n_jn_oqs.append(simps(dndE(Enu_list, option_th, 'JN', '', par, bin_i), Enu_list))

        #chi DUNE
        chi_list_dune = []
        for i in range(len(n_dune)):
            nstd = n_dune[i]
            nth = n_dune_oqs[i]
            if nth == 0:
                chi_dune = 1e9
            else:
                chi_dune = (nstd - (1+a)*nth)**2/nth
            chi_list_dune.append(chi_dune)

        #chi HK
        chi_list_hk = []
        for i in range(len(n_hk)):
            nstd = n_hk[i]
            nth = n_hk_oqs[i]
            if nth == 0:
                chi_hk = 1e9
            else:
                chi_hk = (nstd - (1+a)*nth)**2/nth
            chi_list_hk.append(chi_hk)
        
        #chi JUNO
        chi_list_jn = []
        for i in range(len(n_jn)):
            nstd = n_jn[i]
            nth = n_jn_oqs[i]
            if nth == 0:
                chi_jn = 1e9
            else:
                chi_jn = (nstd - (1+a)*nth)**2/nth
            chi_list_jn.append(chi_jn)

        sigma_a = 0.40 #uncertainty in the flux

        # chi_tot = sum(chi_list_dune) + a**2/sigma_a**2 + sum(chi_list_hk_es) + sum(chi_list_hk_ibd) + sum(chi_list_jn)
        chi_tot = sum(chi_list_dune) + a**2/sigma_a**2 + sum(chi_list_hk) + sum(chi_list_jn)

        # print('chi =', chi_tot)
        return chi_tot

    else:
        if detector == 'hk':
            Enu_list = np.linspace(3, 60, 101)
            channels = ['IBD','ES']
        elif detector == 'jn':
            Enu_list = np.linspace(3, 60, 101)
            channels = ['']
        elif detector == 'dune':
            Enu_list = np.linspace(4.5, 60, 101)
            channels = ['']

        n_list = []
        n_oqs_list = []
        chi_list = []
        Enu_bins = read_bins(detector, hierarchy)
        for i in range(len(Enu_bins)-1):
            bin_i = i+1
            n = 0
            noqs = 0
            for channel in channels:
                n += simps(dndE(Enu_list, option_std, detector.upper(), channel, par, bin_i), Enu_list)
                noqs += simps(dndE(Enu_list, option_th, detector.upper(), channel, par, bin_i), Enu_list)
            n_list.append(n)
            n_oqs_list.append(noqs)

        #chi
        for i in range(len(n_list)):
            nstd = n_list[i]
            nth = n_oqs_list[i]
            if nth == 0:
                chi = 1e9
            else:
                chi = (nstd - (1+a)*nth)**2/nth
            chi_list.append(chi)
        
        sigma_a = 0.40 #uncertainty in the flux
        chi_tot = sum(chi_list) + a**2/sigma_a**2

        # print('Chi Tot=', chi_tot, D, g_exp)
        return chi_tot


#################### MSC non-diagonal model #####################

def chi_square_non_diag(g_exp, theta12, a):
    import import_config
    reload_config(import_config)
    from import_config import config_list

    option_std = config_list[5][:2]
    option_th = config_list[5][6:]

    if 'NH' in option_std:
        hierarchy = 'NH'
    elif 'IH' in option_std:
        hierarchy = 'IH'
    
    if 'IH' in config_list[5] and 'NH' in config_list[5]:
        hierarchy = 'NH'

    par = g_exp, theta12

    detector = config_list[2]

    if detector == 'dune_hk_jn':
        n_dune = []
        n_dune_oqs = []
        n_hk = []
        n_hk_oqs = []
        n_hk_es = []
        n_hk_ibd = []
        n_hk_es_oqs = []
        n_hk_ibd_oqs = []
        n_jn = []
        n_jn_oqs = []

        #DUNE
        Enu_bins_dune = read_bins('dune', hierarchy)
        Enu_list = np.linspace(4.5, 60, 101)
        for i in range(len(Enu_bins_dune)-1):
            bin_i = i+1
            n_dune.append(simps(dndE(Enu_list, option_std, 'DUNE', '', par, bin_i), Enu_list))
            n_dune_oqs.append(simps(dndE(Enu_list, option_th, 'DUNE', '', par, bin_i), Enu_list))

        #HK
        #ES
        Enu_bins_hk = read_bins('hk', hierarchy)
        Enu_list = np.linspace(3, 60, 101)
        for i in range(len(Enu_bins_hk)-1):
            bin_i = i+1
            #ES
            n_hk_es = simps(dndE(Enu_list, option_std, 'HK', 'ES', par, bin_i), Enu_list)
            n_hk_es_oqs = simps(dndE(Enu_list, option_th, 'HK', 'ES', par, bin_i), Enu_list)
            #IBD
            n_hk_ibd = simps(dndE(Enu_list, option_std, 'HK', 'IBD', par, bin_i), Enu_list)
            n_hk_ibd_oqs = simps(dndE(Enu_list, option_th, 'HK', 'IBD', par, bin_i), Enu_list)
            n_hk.append(n_hk_es + n_hk_ibd)
            n_hk_oqs.append(n_hk_es_oqs + n_hk_ibd_oqs)

        #JUNO
        Enu_bins_jn = read_bins('jn', hierarchy)
        Enu_list = np.linspace(3, 60, 101)
        for i in range(len(Enu_bins_jn)-1):
            bin_i = i+1
            n_jn.append(simps(dndE(Enu_list, option_std, 'JN', '', par, bin_i), Enu_list))
            n_jn_oqs.append(simps(dndE(Enu_list, option_th, 'JN', '', par, bin_i), Enu_list))

        #chi DUNE
        chi_list_dune = []
        for i in range(len(n_dune)):
            nstd = n_dune[i]
            nth = n_dune_oqs[i]
            if nth == 0:
                chi_dune = 1e9
            else:
                chi_dune = (nstd - (1+a)*nth)**2/nth
            chi_list_dune.append(chi_dune)

        #chi HK
        chi_list_hk = []
        for i in range(len(n_hk)):
            nstd = n_hk[i]
            nth = n_hk_oqs[i]
            if nth == 0:
                chi_hk = 1e9
            else:
                chi_hk = (nstd - (1+a)*nth)**2/nth
            chi_list_hk.append(chi_hk)
        
        #chi JUNO
        chi_list_jn = []
        for i in range(len(n_jn)):
            nstd = n_jn[i]
            nth = n_jn_oqs[i]
            if nth == 0:
                chi_jn = 1e9
            else:
                chi_jn = (nstd - (1+a)*nth)**2/nth
            chi_list_jn.append(chi_jn)

        sigma_a = 0.40 #uncertainty in the flux

        # chi_tot = sum(chi_list_dune) + a**2/sigma_a**2 + sum(chi_list_hk_es) + sum(chi_list_hk_ibd) + sum(chi_list_jn)
        chi_tot = sum(chi_list_dune) + a**2/sigma_a**2 + sum(chi_list_hk) + sum(chi_list_jn)

        # print('chi =', chi_tot)
        return chi_tot

    else:
        if detector == 'hk':
            Enu_list = np.linspace(3, 60, 101)
            channels = ['IBD','ES']
        elif detector == 'jn':
            Enu_list = np.linspace(3, 60, 101)
            channels = ['']
        elif detector == 'dune':
            Enu_list = np.linspace(4.5, 60, 101)
            channels = ['']

        n_list = []
        n_oqs_list = []
        chi_list = []
        Enu_bins = read_bins(detector, hierarchy)
        for i in range(len(Enu_bins)-1):
            bin_i = i+1
            n = 0
            noqs = 0
            for channel in channels:
                n += simps(dndE(Enu_list, option_std, detector.upper(), channel, par, bin_i), Enu_list)
                noqs += simps(dndE(Enu_list, option_th, detector.upper(), channel, par, bin_i), Enu_list)
            n_list.append(n)
            n_oqs_list.append(noqs)

        #chi
        for i in range(len(n_list)):
            nstd = n_list[i]
            nth = n_oqs_list[i]
            if nth == 0:
                chi = 1e9
            else:
                chi = (nstd - (1+a)*nth)**2/nth
            chi_list.append(chi)
        
        sigma_a = 0.40 #uncertainty in the flux
        chi_tot = sum(chi_list) + a**2/sigma_a**2

        #print('Chi Tot=',chi_tot)
        return chi_tot



################### nu-loss model #####################

def chi_square_loss(g_exp, theta12, a):
    import import_config
    reload_config(import_config)
    from import_config import config_list

    option_std = config_list[5][:2]
    option_th = config_list[5][6:]

    if 'NH' in option_std:
        hierarchy = 'NH-l'
        option_std = 'NH-l'
    elif 'IH' in option_std:
        hierarchy = 'IH-l'
        option_std = 'IH-l'
    
    if 'IH' in config_list[5] and 'NH' in config_list[5]:
        hierarchy = 'NH-l'

    par = g_exp, theta12

    detector = config_list[2]

    if detector == 'dune_hk_jn':
        n_dune = []
        n_dune_oqs = []
        n_hk = []
        n_hk_oqs = []
        n_hk_es = []
        n_hk_ibd = []
        n_hk_es_oqs = []
        n_hk_ibd_oqs = []
        n_jn = []
        n_jn_oqs = []

        #DUNE
        Enu_bins_dune = bins_loss('dune')
        Enu_list = np.linspace(4.5, 60, 101)
        for i in range(len(Enu_bins_dune)-1):
            bin_i = i+1
            n_dune.append(simps(dndE(Enu_list, option_std, 'DUNE', '', par, bin_i), Enu_list))
            n_dune_oqs.append(simps(dndE(Enu_list, option_th, 'DUNE', '', par, bin_i), Enu_list))

        #HK
        #ES
        Enu_bins_hk = bins_loss('hk')
        Enu_list = np.linspace(3, 60, 101)
        for i in range(len(Enu_bins_hk)-1):
            bin_i = i+1
            # #ES
            # #Enubin = np.linspace(Enu_bins_hk[i], Enu_bins_hk[i+1], 101)
            # n_hk_es.append(simps(dndE(Enu_list, option_std, 'HK', 'ES', par, bin_i), Enu_list))
            # n_hk_es_oqs.append(simps(dndE(Enu_list, option_th, 'HK', 'ES', par, bin_i), Enu_list))    
            # #IBD
            # n_hk_ibd.append(simps(dndE(Enu_list, option_std, 'HK', 'IBD', par, bin_i), Enu_list))
            # n_hk_ibd_oqs.append(simps(dndE(Enu_list, option_th, 'HK', 'IBD', par, bin_i), Enu_list))
            #ES
            n_hk_es = simps(dndE(Enu_list, option_std, 'HK', 'ES', par, bin_i), Enu_list)
            n_hk_es_oqs = simps(dndE(Enu_list, option_th, 'HK', 'ES', par, bin_i), Enu_list)
            #IBD
            n_hk_ibd = simps(dndE(Enu_list, option_std, 'HK', 'IBD', par, bin_i), Enu_list)
            n_hk_ibd_oqs = simps(dndE(Enu_list, option_th, 'HK', 'IBD', par, bin_i), Enu_list)
            n_hk.append(n_hk_es + n_hk_ibd)
            n_hk_oqs.append(n_hk_es_oqs + n_hk_ibd_oqs)
        
        #JUNO
        Enu_list = np.linspace(3, 60, 101)
        Enu_bins_jn = bins_loss('jn')
        for i in range(len(Enu_bins_jn)-1):
            bin_i = i+1
            n_jn.append(simps(dndE(Enu_list, option_std, 'JN', '', par, bin_i), Enu_list))
            n_jn_oqs.append(simps(dndE(Enu_list, option_th, 'JN', '', par, bin_i), Enu_list))

        #chi DUNE
        chi_list_dune = []
        for i in range(len(n_dune)):
            nstd = n_dune[i]
            nth = n_dune_oqs[i]
            if nth == 0:
                chi_dune = 1e9
            else:
                chi_dune = (nstd - (1+a)*nth)**2/nth
            chi_list_dune.append(chi_dune)

        #chi HK
        chi_list_hk = []
        for i in range(len(n_hk)):
            nstd = n_hk[i]
            nth = n_hk_oqs[i]
            if nth == 0:
                chi_hk = 1e9
            else:
                chi_hk = (nstd - (1+a)*nth)**2/nth
            chi_list_hk.append(chi_hk)
   
        #chi JUNO
        chi_list_jn = []
        for i in range(len(n_jn)):
            nstd = n_jn[i]
            nth = n_jn_oqs[i]
            if nth == 0:
                chi_jn = 1e9
            else:
                chi_jn = (nstd - (1+a)*nth)**2/nth
            chi_list_jn.append(chi_jn)

        sigma_a = 0.40 #uncertainty in the flux

        # chi_tot = sum(chi_list_dune) + a**2/sigma_a**2 + sum(chi_list_hk_es) + sum(chi_list_hk_ibd) + sum(chi_list_jn)
        chi_tot = sum(chi_list_dune) + a**2/sigma_a**2 + sum(chi_list_hk) + sum(chi_list_jn)

        # print('chi =', chi_tot)
        return chi_tot

    else:
        if detector == 'hk':
            Enu_list = np.linspace(3, 60, 101)
            channels = ['IBD','ES']
        elif detector == 'jn':
            Enu_list = np.linspace(3, 60, 101)
            channels = ['']
        elif detector == 'dune':
            Enu_list = np.linspace(4.5, 60, 101)
            channels = ['']
        # import rate
        # reload(rate)
        n_list = []
        n_oqs_list = []
        chi_list = []
        Enu_bins = bins_loss(detector)
        # print(option_std, '/', option_th)
        for i in range(len(Enu_bins)-1):
            bin_i = i+1
            n = 0
            noqs = 0
            for channel in channels:
                n += simps(dndE(Enu_list, option_std, detector.upper(), channel, par, bin_i), Enu_list)
                noqs += simps(dndE(Enu_list, option_th, detector.upper(), channel, par, bin_i), Enu_list)
                # print(n, noqs)
                if n<2 or noqs<2:
                    print('events =', n, noqs, g_exp)
            n_list.append(n)
            n_oqs_list.append(noqs)

            #chi
            for i in range(len(n_list)):
                nstd = n_list[i]
                nth = n_oqs_list[i]
                if nth == 0:
                    chi = 1e9
                else:
                    chi = (nstd - (1+a)*nth)**2/nth
                chi_list.append(chi)

        sigma_a = 0.40 #uncertainty in the flux
        chi_tot = sum(chi_list) + a**2/sigma_a**2
        # print('Chi Tot=',chi_tot)
        return chi_tot


############ nu-loss with matter-effects ##############

def chi_square_loss_m(g_exp, theta12, Deltam21, a):
    import import_config
    reload_config(import_config)
    from import_config import config_list

    option_std = config_list[5][:5]
    option_th = config_list[5][9:]

    if 'NH' in option_std:
        hierarchy = 'NH-l'
        option_std = 'NH-l'
    elif 'IH' in option_std:
        hierarchy = 'IH-l'
        option_std = 'IH-l'
    
    if 'IH' in config_list[5] and 'NH' in config_list[5]:
        hierarchy = 'NH-l'

    par = g_exp, theta12, Deltam21

    detector = config_list[2]

    if detector == 'dune_hk_jn':
        n_dune = []
        n_dune_oqs = []
        n_hk = []
        n_hk_oqs = []
        n_hk_es = []
        n_hk_ibd = []
        n_hk_es_oqs = []
        n_hk_ibd_oqs = []
        n_jn = []
        n_jn_oqs = []

        #DUNE
        Enu_bins_dune = bins_loss('dune')
        Enu_list = np.linspace(4.5, 60, 101)
        for i in range(len(Enu_bins_dune)-1):
            bin_i = i+1
            n_dune.append(simps(dndE(Enu_list, option_std, 'DUNE', '', par, bin_i), Enu_list))
            n_dune_oqs.append(simps(dndE(Enu_list, option_th, 'DUNE', '', par, bin_i), Enu_list))

        #HK
        #ES
        Enu_bins_hk = bins_loss('hk')
        Enu_list = np.linspace(3, 60, 101)
        for i in range(len(Enu_bins_hk)-1):
            bin_i = i+1
            #ES
            #Enubin = np.linspace(Enu_bins_hk[i], Enu_bins_hk[i+1], 101)
            # n_hk_es.append(simps(dndE(Enu_list, option_std, 'HK', 'ES', par, bin_i), Enu_list))
            # n_hk_es_oqs.append(simps(dndE(Enu_list, option_th, 'HK', 'ES', par, bin_i), Enu_list))    
            # #IBD
            # n_hk_ibd.append(simps(dndE(Enu_list, option_std, 'HK', 'IBD', par, bin_i), Enu_list))
            # n_hk_ibd_oqs.append(simps(dndE(Enu_list, option_th, 'HK', 'IBD', par, bin_i), Enu_list))
            n_hk_es = simps(dndE(Enu_list, option_std, 'HK', 'ES', par, bin_i), Enu_list)
            n_hk_es_oqs = simps(dndE(Enu_list, option_th, 'HK', 'ES', par, bin_i), Enu_list)
            #IBD
            n_hk_ibd = simps(dndE(Enu_list, option_std, 'HK', 'IBD', par, bin_i), Enu_list)
            n_hk_ibd_oqs = simps(dndE(Enu_list, option_th, 'HK', 'IBD', par, bin_i), Enu_list)
            n_hk.append(n_hk_es + n_hk_ibd)
            n_hk_oqs.append(n_hk_es_oqs + n_hk_ibd_oqs)

        #JUNO
        Enu_list = np.linspace(3, 60, 101)
        Enu_bins_jn = bins_loss('jn')
        for i in range(len(Enu_bins_jn)-1):
            bin_i = i+1
            n_jn.append(simps(dndE(Enu_list, option_std, 'JN', '', par, bin_i), Enu_list))
            n_jn_oqs.append(simps(dndE(Enu_list, option_th, 'JN', '', par, bin_i), Enu_list))

        #chi DUNE
        chi_list_dune = []
        for i in range(len(n_dune)):
            nstd = n_dune[i]
            nth = n_dune_oqs[i]
            if nth == 0:
                chi_dune = 1e9
            else:
                chi_dune = (nstd - (1+a)*nth)**2/nth
            chi_list_dune.append(chi_dune)

        #chi HK
        chi_list_hk = []
        for i in range(len(n_hk)):
            nstd = n_hk[i]
            nth = n_hk_oqs[i]
            if nth == 0:
                chi_hk = 1e9
            else:
                chi_hk = (nstd - (1+a)*nth)**2/nth
            chi_list_hk.append(chi_hk)
        #chi HK
        #ES
        # chi_list_hk_es = []
        # for i in range(len(n_hk_es)):
        #     nstd = n_hk_es[i]
        #     nth = n_hk_es_oqs[i]
        #     if nth == 0:
        #         chi_hk = 1e9
        #     else:
        #         chi_hk = (nstd - (1+a)*nth)**2/nth
        #     chi_list_hk_es.append(chi_hk)
        # #IBD
        # chi_list_hk_ibd = []
        # for i in range(len(n_hk_ibd)-1):
        #     nstd = n_hk_ibd[i]
        #     nth = n_hk_ibd_oqs[i]
        #     if nth == 0:
        #         chi_hk = 1e9
        #     else:
        #         chi_hk = (nstd - (1+a)*nth)**2/nth
        #     chi_list_hk_ibd.append(chi_hk)

        #chi JUNO
        chi_list_jn = []
        for i in range(len(n_jn)):
            nstd = n_jn[i]
            nth = n_jn_oqs[i]
            if nth == 0:
                chi_jn = 1e9
            else:
                chi_jn = (nstd - (1+a)*nth)**2/nth
            chi_list_jn.append(chi_jn)

        sigma_a = 0.40 #uncertainty in the flux

        # chi_tot = sum(chi_list_dune) + a**2/sigma_a**2 + sum(chi_list_hk_es) + sum(chi_list_hk_ibd) + sum(chi_list_jn)
        chi_tot = sum(chi_list_dune) + a**2/sigma_a**2 + sum(chi_list_hk) + sum(chi_list_jn)

        # print('chi =', chi_tot)
        return chi_tot

    else:
        if detector == 'hk':
            Enu_list = np.linspace(3, 60, 101)
            channels = ['IBD','ES']
        elif detector == 'jn':
            Enu_list = np.linspace(3, 60, 101)
            channels = ['']
        elif detector == 'dune':
            Enu_list = np.linspace(4.5, 60, 101)
            channels = ['']
        
        n_list = []
        n_oqs_list = []
        chi_list = []
        Enu_bins = bins_loss(detector)
        for i in range(len(Enu_bins)-1):
            bin_i = i+1
            n = 0
            noqs = 0
            for channel in channels:
                n += simps(dndE(Enu_list, option_std, detector.upper(), channel, par, bin_i), Enu_list)
                noqs += simps(dndE(Enu_list, option_th, detector.upper(), channel, par, bin_i), Enu_list)
                if n<2 or noqs<2:
                    print('events =',n, noqs, g_exp)
            n_list.append(n)
            n_oqs_list.append(noqs)

            #chi
            for i in range(len(n_list)):
                nstd = n_list[i]
                nth = n_oqs_list[i]
                if nth == 0:
                    chi = 1e9
                else:
                    chi = (nstd - (1+a)*nth)**2/nth
                chi_list.append(chi)
        
        sigma_a = 0.40 #uncertainty in the flux
        chi_tot = sum(chi_list) + a**2/sigma_a**2

        # print('Chi Tot=',chi_tot)
        return chi_tot
