import numpy as np
import math
import sqlite3
from scipy.interpolate import interp1d
from scipy.integrate import simpson
from interaction import sigma_nue_Ar, sigma_IBD
from flavor_conversion import P_NH_nue, P_IH_nue, P_IH_antie, P_NH_antie, Pee_msc_vacuum_E, Pee_deco_vac_basis_loss_E, Pee_bar_msc_vacuum_E, Pee_bar_deco_vac_basis_loss_E
from flavor_conversion import P_combined_msc_E, P_combined_loss_E
from integ_fluxes import interp_fluxes
import platform
import socket

pc_name = socket.gethostname()

# DOCS: https://docs.python.org/3/library/sqlite3.html
if platform.system() == 'Windows':
    connection = sqlite3.connect('data/database_windows.db')
else:
    data_path = 'data/database.db'
    connection = sqlite3.connect(data_path)
cursor = connection.cursor()


me = 0.510998950 #MeV
NAr_DUNE = 6.03e32
Ne_HK = 1.25e35 #from https://arxiv.org/abs/2011.10933
Np_HK = 2.5e34 #from https://arxiv.org/abs/2011.10933
Np_JUNO = 1.5e33 #from https://arxiv.org/abs/2011.10933
efficiency_HK_ES = 0.6
efficiency_HK_IBD = 0.6
efficiency_JUNO_IBD = 0.5

def dndE(Enu, mix, detector):
    model = 'LS220-s11.2c'
    D = 10 #kpc
    thetaz = 0
    par = 0

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
    Pee = 1.
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

    elif mix == 'NH + OQS' or mix == 'IH + OQS':
        g3_exp, g8_exp, theta12 = par
        D_kpc = D
        gamma3_nat = 10**g3_exp
        gamma8_nat = 10**g8_exp
        Pee = Pee_msc_vacuum_E(0, 0, D_kpc, gamma3_nat, gamma8_nat, theta12, mix[:2])
        Pee_bar = Pee_bar_msc_vacuum_E(0, 0, D_kpc, gamma3_nat, gamma8_nat, theta12, mix[:2])
    
    elif mix == 'NH(m) + OQS' or mix == 'IH(m) + OQS':
        g3_exp, g8_exp, theta12, Deltam21 = par
        oscil_par = theta12, Deltam21
        D_kpc = D
        gamma3_nat = 10**g3_exp
        gamma8_nat = 10**g8_exp
        Pee = P_combined_msc_E(0, D_kpc, Enu, gamma3_nat, gamma8_nat, mix[:2], thetaz, 'nue', oscil_par, detector)
        Pee_bar = P_combined_msc_E(0, D_kpc, Enu, gamma3_nat, gamma8_nat, mix[:2], thetaz, 'nuebar', oscil_par, detector)
    
    elif mix == 'NH + OQS-loss' or mix == 'IH + OQS-loss':
        g_exp, theta12 = par
        D_kpc = D
        gamma_nat = 10**g_exp
        Pee = Pee_deco_vac_basis_loss_E(0, 0, D_kpc, gamma_nat, theta12, mix[:2])
        Pee_bar = Pee_bar_deco_vac_basis_loss_E(0, 0, D_kpc, gamma_nat, theta12, mix[:2])

    elif mix == 'NH(m) + OQS-loss' or mix == 'IH(m) + OQS-loss':
        g_exp, theta12, Deltam21 = par
        oscil_par = theta12, Deltam21
        D_kpc = D
        gamma_nat = 10**g_exp
        Pee = P_combined_loss_E(0, D_kpc, Enu, gamma_nat, mix[:2], thetaz, 'nue', oscil_par, detector)
        Pee_bar = P_combined_loss_E(0, D_kpc, Enu, gamma_nat, mix[:2], thetaz, 'nuebar', oscil_par, detector)

    elif mix == 'NH + OQS-E' or mix == 'IH + OQS-E':
        g3_exp, g8_exp, theta12 = par
        D_kpc = D
        gamma3_nat = 10**g3_exp
        gamma8_nat = 10**g8_exp
        Pee = Pee_msc_vacuum_E(2, Enu, D_kpc, gamma3_nat, gamma8_nat, theta12, mix[:2])
        Pee_bar = Pee_bar_msc_vacuum_E(2, Enu, D_kpc, gamma3_nat, gamma8_nat, theta12, mix[:2])

    elif mix == 'NH + OQS-E-2.5' or mix == 'IH + OQS-E-2.5':
        g3_exp, g8_exp, theta12 = par
        D_kpc = D
        gamma3_nat = 10**g3_exp
        gamma8_nat = 10**g8_exp
        Pee = Pee_msc_vacuum_E(2.5, Enu, D_kpc, gamma3_nat, gamma8_nat, theta12, mix[:2])
        Pee_bar = Pee_bar_msc_vacuum_E(2.5, Enu, D_kpc, gamma3_nat, gamma8_nat, theta12, mix[:2])
    
    elif mix == 'NH(m) + OQS-E' or mix == 'IH(m) + OQS-E':
        g3_exp, g8_exp, theta12, Deltam21 = par
        oscil_par = theta12, Deltam21
        D_kpc = D
        gamma3_nat = 10**g3_exp
        gamma8_nat = 10**g8_exp
        Pee = P_combined_msc_E(2, D_kpc, Enu, gamma3_nat, gamma8_nat, mix[:2], thetaz, 'nue', oscil_par, detector)
        Pee_bar = P_combined_msc_E(2, D_kpc, Enu, gamma3_nat, gamma8_nat, mix[:2], thetaz, 'nuebar', oscil_par, detector)
    
    elif mix == 'NH(m) + OQS-E-2.5' or mix == 'IH(m) + OQS-E-2.5':
        g3_exp, g8_exp, theta12, Deltam21 = par
        oscil_par = theta12, Deltam21
        D_kpc = D
        gamma3_nat = 10**g3_exp
        gamma8_nat = 10**g8_exp
        Pee = P_combined_msc_E(2.5, D_kpc, Enu, gamma3_nat, gamma8_nat, mix[:2], thetaz, 'nue', oscil_par, detector)
        Pee_bar = P_combined_msc_E(2.5, D_kpc, Enu, gamma3_nat, gamma8_nat, mix[:2], thetaz, 'nuebar', oscil_par, detector)


    elif mix == 'NH + OQS-loss-E' or mix == 'IH + OQS-loss-E':
        g_exp, theta12 = par
        D_kpc = D
        gamma_nat = 10**g_exp
        Pee = Pee_deco_vac_basis_loss_E(2, Enu, D_kpc, gamma_nat, theta12, mix[:2])
        Pee_bar = Pee_bar_deco_vac_basis_loss_E(2, Enu, D_kpc, gamma_nat, theta12, mix[:2])

    elif mix == 'NH + OQS-loss-E-2.5' or mix == 'IH + OQS-loss-E-2.5':
        g_exp, theta12 = par
        D_kpc = D
        gamma_nat = 10**g_exp
        Pee = Pee_deco_vac_basis_loss_E(2.5, Enu, D_kpc, gamma_nat, theta12, mix[:2])
        Pee_bar = Pee_bar_deco_vac_basis_loss_E(2.5, Enu, D_kpc, gamma_nat, theta12, mix[:2])

    elif mix == 'NH(m) + OQS-loss-E' or mix == 'IH(m) + OQS-loss-E':
        g_exp, theta12, Deltam21 = par
        oscil_par = theta12, Deltam21
        D_kpc = D
        gamma_nat = 10**g_exp
        Pee = P_combined_loss_E(2, D_kpc, Enu, gamma_nat, mix[:2], thetaz, 'nue', oscil_par, detector)
        Pee_bar = P_combined_loss_E(2, D_kpc, Enu, gamma_nat, mix[:2], thetaz, 'nuebar', oscil_par, detector)

    elif mix == 'NH(m) + OQS-loss-E-2.5' or mix == 'IH(m) + OQS-loss-E-2.5':
        g_exp, theta12, Deltam21 = par
        oscil_par = theta12, Deltam21
        D_kpc = D
        gamma_nat = 10**g_exp
        Pee = P_combined_loss_E(2.5, D_kpc, Enu, gamma_nat, mix[:2], thetaz, 'nue', oscil_par, detector)
        Pee_bar = P_combined_loss_E(2.5, D_kpc, Enu, gamma_nat, mix[:2], thetaz, 'nuebar', oscil_par, detector)

    Fe = Pee * Fe0 + (1 - Pee) * Fx0
    Fe_bar = Pee_bar * Fe0_bar + (1 - Pee_bar) * Fx0
    Fx = 1/4 * ((1 - Pee) * Fe0 + (2 + Pee + Pee_bar) * Fx0 + (1 - Pee_bar) * Fe0_bar)

    if detector == 'HK':
        from interaction import cross_ES

        df1 = Fe + Fe_bar + 2*Fx + 2*Fx
        d1 = Ne_HK * efficiency_HK_ES * df1 * cross_ES(Enu)
        
        df2 = Fe_bar * sigma_IBD(Enu)
        d2 = Np_HK * efficiency_HK_IBD * df2
        return d1
    
    elif detector == 'JN':
        df = Fe_bar * sigma_IBD(Enu)
        d = Np_JUNO * efficiency_JUNO_IBD * df
        return d

    elif detector == 'DUNE':
        df = Fe * sigma_nue_Ar(Enu)
        d = NAr_DUNE * dune_efficiency(Enu) * df
        return d

#taken from figure 7 of https://arxiv.org/pdf/2008.06647.pdf (chosen 5MeV for deposited energy threshold)
def dune_efficiency(Enu):
    data_file = np.loadtxt('data/dune_eff_5MeV.dat', delimiter=' ')
    Energy = data_file[:,0]
    eff = data_file[:,1]
    eff_func = interp1d(Energy,eff,kind='cubic',fill_value='extrapolate')
    return eff_func(Enu)


def sigma_E_func():
    Te_list = np.linspace(0.1,60,101)
    s_list_hk = []
    s_list_dune = []
    s_list_jn = []
    for Te in Te_list:
        # sig = -0.0839 + 0.349 * np.sqrt(Te + 0.511) + 0.0397*(Te + 0.511)
        #sigma of resolution in HK
        sigma_hk = -0.0839+0.349*np.sqrt(Te)+0.0397*Te
        s_list_hk.append(2*sigma_hk)
        #sigma of resolution in DUNE
        sigma_dune = 0.11*np.sqrt(Te) + 0.02*Te
        s_list_dune.append(2*sigma_dune)
        #sigma of resolution in JUNO
        sigma_jn = 0.03*np.sqrt(Te)
        s_list_jn.append(2*sigma_jn)

    sigma_func_hk = interp1d(Te_list, s_list_hk, fill_value='extrapolate')
    sigma_func_dune = interp1d(Te_list, s_list_dune, fill_value='extrapolate')
    sigma_func_jn = interp1d(Te_list, s_list_jn, fill_value='extrapolate')

    return sigma_func_hk, sigma_func_dune, sigma_func_jn


#we automatically create a number of bins with twice the resolution at each energy
def create_bins(trigger_l, mix):    
    if trigger_l == 'loss':
        bins_list_hk = bins_loss('hk')
        bins_list_dune = bins_loss('dune')
        bins_list_jn = bins_loss('jn')
    else:
        bins_list_hk = bins(3, 'hk', mix)
        bins_list_dune = bins(4.5, 'dune', mix)
        bins_list_jn = bins(3, 'jn', mix)
    return bins_list_hk, bins_list_dune, bins_list_jn

def save_bins():
    cursor.execute("delete from bins")
    connection.commit()

    def save_func(trigger_l, mix):
        if trigger_l == 'loss':
            bins_list_hk = bins_loss('hk')
            bins_list_dune = bins_loss('dune')
            bins_list_jn = bins_loss('jn')
        else:
            bins_list_hk = bins(3, 'hk', mix)
            bins_list_dune = bins(4.5, 'dune', mix)
            bins_list_jn = bins(3, 'jn', mix)

        #write the bin values in the database
        for i in range(len(bins_list_hk)):
            cursor.execute("insert into bins values (?,?,?,?)", ('hk', mix, 3, bins_list_hk[i]))
        for i in range(len(bins_list_dune)):
            cursor.execute("insert into bins values (?,?,?,?)", ('dune', mix, 4.5, bins_list_dune[i]))
        for i in range(len(bins_list_jn)):
            cursor.execute("insert into bins values (?,?,?,?)", ('jn', mix, 3, bins_list_jn[i]))

    save_func(0,'NH')
    save_func(0,'IH')

    save_func('loss','NH-l')
    save_func('loss','IH-l')

    connection.commit()
    connection.close()


def bins(Eth, detector, mix):
    sigma_func_hk, sigma_func_dune, sigma_func_jn = sigma_E_func()

    if detector == 'hk':
        sigma_func = sigma_func_hk
    elif detector == 'dune':
        sigma_func = sigma_func_dune
    elif detector == 'jn':
        sigma_func = sigma_func_jn

    bins_list = [Eth]

    left_edge = Eth
    right_edge = Eth + 2*sigma_func(Eth)

    while True:
        Enu_list = np.linspace(left_edge, right_edge, 201)
        n = simpson(dndE(Enu_list, mix, detector.upper()), Enu_list)
        mean = (right_edge-left_edge)/2
        delta = right_edge-left_edge >= 2*sigma_func(mean)
        
        if n > 5 and delta and detector != 'jn':
            bins_list.append(right_edge)
            left_edge = right_edge
            right_edge += 2*sigma_func(left_edge)
        
        elif detector == 'jn':
            bins_list = np.array([3., 16., 61.])
            return bins_list

        else:
            right_edge += 1
        
        if right_edge >= 61:
            if detector == 'jn':
                bins_list.append(61)
                break
            elif detector == 'dune':
                bins_list = bins_list[:-3]
                bins_list.append(30.)
                bins_list.append(42.)
                bins_list.append(61.)
                break
            
            elif detector == 'hk':
                bins_list.append(61)
                break
    # print(bins_list)
    return np.array(bins_list)


def bins_loss(detector):
    if detector == 'dune':
        # bins_list = np.linspace(4.5,61,4)
        bins_list = [4.5,20,31,61]
    elif detector == 'hk':
        # bins_list = np.linspace(3,61,4)
        # bins_list = [3,11,17,22,61]
        bins_list = [3,11,17,22,28,61]
    elif detector == 'jn':
        bins_list = np.linspace(3,61,2)
    return bins_list

# save_bins()
# print(bins(3, 'hk', 'IH'))
