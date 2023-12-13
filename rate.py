import numpy as np
from importlib import reload
from flavor_conversion import U3
import math
import sqlite3
import platform
import socket
from scipy.interpolate import interp1d
from interaction import sigma_nue_Ar, sigma_IBD
from flavor_conversion import P_NH_nue, P_IH_nue, P_IH_antie, P_NH_antie, Pee_msc_vacuum_E, Pee_bar_msc_vacuum_E, Pab_vacuum_nu_loss_E, Pee_msc_non_diag_vacuum_E, Pee_bar_msc_non_diag_vacuum_E, P_combined_msc_non_diag_E, Pee_msc_vacuum_conserved_E, Pee_bar_msc_vacuum_conserved_E
from flavor_conversion import P_combined_msc_E, P_combined_loss_E, Pie_earth_interp_func, P_combined_msc_conserved_E
from read_database import read_work_function
from integ_fluxes import interp_fluxes

def reload_config(module_name):
    reload(module_name)

pc_name = socket.gethostname()

#DOCS: https://docs.python.org/3/library/sqlite3.html
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
Np_JUNO = 1.5e33 #pg.74 of https://arxiv.org/abs/1507.05613v2
efficiency_HK_ES = 0.6
efficiency_HK_IBD = 0.6
efficiency_JUNO_IBD = 0.5

def dndE(Enu, mix, detector, channel, par, bin_i):
    import import_config
    reload_config(import_config)
    from import_config import config_list

    if 'NH' in mix:
        hierarchy = 'NH'
    elif 'IH' in mix:
        hierarchy = 'IH'
    
    if 'IH' in config_list[5] and 'NH' in config_list[5]:
        hierarchy = 'NH'

    trigger_l = 0

    if mix == 'NH-l':
        mix = 'NH'
        trigger_l = 'loss'
    elif mix == 'IH-l':
        mix = 'IH'
        trigger_l = 'loss'
    elif 'loss' in mix:
        trigger_l = 'loss'

    model = config_list[0]
    D_kpc = config_list[1] #kpc
    thetaz = config_list[6]

    #integrated fluxes
    Fe0_bar_i, Fe0_i, Fx0_i = interp_fluxes(model, Enu)
    kpc_to_cm = 3.086e21
    D_cm = D_kpc * kpc_to_cm
    Fe0_bar = Fe0_bar_i/(4*math.pi*D_cm**2)
    Fe0 = Fe0_i/(4*math.pi*D_cm**2)
    Fx0 = Fx0_i/(4*math.pi*D_cm**2)

    Dm21 = 7.42e-5 #eV² +0.21 -0.20
    t12 = 33.45 #+0.77 -0.75 NuFIT 2021 https://arxiv.org/pdf/2111.03086.pdf
    t12 = t12*math.pi/180

    #mixed fluxes
    if mix == 'no':
        Pee = 1
        Pee_bar = 1
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

    elif mix == 'NH + OQS' or mix == 'IH + OQS':
        g3_exp, g8_exp, theta12 = par
        gamma3_nat = 10**g3_exp
        gamma8_nat = 10**g8_exp
        Pee = Pee_msc_vacuum_E(0, 0, D_kpc, gamma3_nat, gamma8_nat, theta12, mix[:2])
        Pee_bar = Pee_bar_msc_vacuum_E(0, 0, D_kpc, gamma3_nat, gamma8_nat, theta12, mix[:2])
    
    elif mix == 'NH(m) + OQS' or mix == 'IH(m) + OQS':
        g3_exp, g8_exp, theta12, Deltam21 = par
        oscil_par = theta12, Deltam21
        gamma3_nat = 10**g3_exp
        gamma8_nat = 10**g8_exp
        Pee, Pee_bar = P_combined_msc_E(0, D_kpc, Enu, gamma3_nat, gamma8_nat, mix[:2], thetaz, oscil_par, detector)
    
    elif mix == 'NH + OQS conserved' or mix == 'IH + OQS conserved':
        g_exp, theta12 = par
        gamma_nat = 10**g_exp
        Pee = Pee_msc_vacuum_conserved_E(0, 0, gamma_nat, theta12, mix[:2])
        Pee_bar = Pee_bar_msc_vacuum_conserved_E(0, 0, gamma_nat, theta12, mix[:2])
    
    elif mix == 'NH(m) + OQS conserved' or mix == 'IH(m) + OQS conserved':
        g_exp, theta12 = par
        gamma_nat = 10**g_exp
        Pee, Pee_bar = P_combined_msc_conserved_E(0, Enu, gamma_nat, mix[:2], thetaz, oscil_par, detector)

    elif mix == 'NH + OQS conserved D' or mix == 'IH + OQS conserved D':
        D, g_exp, theta12 = par
        D_cm = D * kpc_to_cm
        Fe0_bar = Fe0_bar_i/(4*math.pi*D_cm**2)
        Fe0 = Fe0_i/(4*math.pi*D_cm**2)
        Fx0 = Fx0_i/(4*math.pi*D_cm**2)
        gamma_nat = 10**g_exp
        Pee = Pee_msc_vacuum_conserved_E(0, 0, gamma_nat, theta12, mix[:2])
        Pee_bar = Pee_bar_msc_vacuum_conserved_E(0, 0, gamma_nat, theta12, mix[:2])

    elif mix == 'NH + OQS ND' or mix == 'IH + OQS ND':
        g_exp, theta12 = par
        gamma_nat = 10**g_exp
        Pee = Pee_msc_non_diag_vacuum_E(0, 0, D_kpc, gamma_nat, theta12, mix[:2])
        Pee_bar = Pee_bar_msc_non_diag_vacuum_E(0, 0, D_kpc, gamma_nat, theta12, mix[:2])

    elif mix == 'NH(m) + OQS ND' or mix == 'IH(m) + OQS ND':
        g_exp, theta12, Deltam21 = par
        oscil_par = theta12, Deltam21
        gamma_nat = 10**g_exp
        Pee, Pee_bar = P_combined_msc_non_diag_E(0, D_kpc, Enu, gamma_nat, mix[:2], thetaz, oscil_par, detector)

    elif mix == 'NH + OQS-loss' or mix == 'IH + OQS-loss':
        g_exp, theta12 = par
        gamma_nat = 10**g_exp
        Pee, Pee_bar, Pme, Pte, Pme_bar, Pte_bar, Pmm, Pmt, Ptt, Ptm, Pem_bar, Pet_bar, Pmm_bar, Pmt_bar, Ptt_bar, Ptm_bar = Pab_vacuum_nu_loss_E(0, 0, D_kpc, gamma_nat, theta12, mix[:2])

    elif mix == 'NH(m) + OQS-loss' or mix == 'IH(m) + OQS-loss':
        g_exp, theta12, Deltam21 = par
        oscil_par = theta12, Deltam21
        gamma_nat = 10**g_exp
        Pee, Pee_bar, Pme, Pte, Pme_bar, Pte_bar, Pmm, Pmt, Ptt, Ptm, Pem_bar, Pet_bar, Pmm_bar, Pmt_bar, Ptt_bar, Ptm_bar = P_combined_loss_E(0, D_kpc, Enu, gamma_nat, mix[:2], thetaz, oscil_par, detector)

    elif mix == 'NH + OQS-E' or mix == 'IH + OQS-E':
        g3_exp, g8_exp, theta12 = par
        gamma3_nat = 10**g3_exp
        gamma8_nat = 10**g8_exp
        Pee = Pee_msc_vacuum_E(2, Enu, D_kpc, gamma3_nat, gamma8_nat, theta12, mix[:2])
        Pee_bar = Pee_bar_msc_vacuum_E(2, Enu, D_kpc, gamma3_nat, gamma8_nat, theta12, mix[:2])

    elif mix == 'NH + OQS-E-2.5' or mix == 'IH + OQS-E-2.5':
        g3_exp, g8_exp, theta12 = par
        gamma3_nat = 10**g3_exp
        gamma8_nat = 10**g8_exp
        Pee = Pee_msc_vacuum_E(2.5, Enu, D_kpc, gamma3_nat, gamma8_nat, theta12, mix[:2])
        Pee_bar = Pee_bar_msc_vacuum_E(2.5, Enu, D_kpc, gamma3_nat, gamma8_nat, theta12, mix[:2])

    elif mix == 'NH + OQS-E conserved' or mix == 'IH + OQS-E conserved':
        g_exp, theta12 = par
        gamma_nat = 10**g_exp
        Pee = Pee_msc_vacuum_conserved_E(2, Enu, gamma_nat, theta12, mix[:2])
        Pee_bar = Pee_bar_msc_vacuum_conserved_E(2, Enu, gamma_nat, theta12, mix[:2])

    elif mix == 'NH + OQS-E-2.5 conserved' or mix == 'IH + OQS-E-2.5 conserved':
        g_exp, theta12 = par
        gamma_nat = 10**g_exp
        Pee = Pee_msc_vacuum_conserved_E(2.5, Enu, gamma_nat, theta12, mix[:2])
        Pee_bar = Pee_bar_msc_vacuum_conserved_E(2.5, Enu, gamma_nat, theta12, mix[:2])

    elif mix == 'NH(m) + OQS-E-2.5 conserved' or mix == 'IH(m) + OQS-E-2.5 conserved':
        g_exp, theta12 = par
        gamma_nat = 10**g_exp
        Pee, Pee_bar = P_combined_msc_conserved_E(2.5, Enu, gamma_nat, mix[:2], thetaz, oscil_par, detector)

    elif mix == 'NH + OQS-E ND' or mix == 'IH + OQS-E ND':
        g_exp, theta12 = par
        gamma_nat = 10**g_exp
        Pee = Pee_msc_non_diag_vacuum_E(2, Enu, D_kpc, gamma_nat, theta12, mix[:2])
        Pee_bar = Pee_bar_msc_non_diag_vacuum_E(2, Enu, D_kpc, gamma_nat, theta12, mix[:2])

    elif mix == 'NH + OQS-E-2.5 ND' or mix == 'IH + OQS-E-2.5 ND':
        g_exp, theta12 = par
        gamma_nat = 10**g_exp
        Pee = Pee_msc_non_diag_vacuum_E(2.5, Enu, D_kpc, gamma_nat, theta12, mix[:2])
        Pee_bar = Pee_bar_msc_non_diag_vacuum_E(2.5, Enu, D_kpc, gamma_nat, theta12, mix[:2])
    
    elif mix == 'NH(m) + OQS-E' or mix == 'IH(m) + OQS-E':
        g3_exp, g8_exp, theta12, Deltam21 = par
        oscil_par = theta12, Deltam21
        gamma3_nat = 10**g3_exp
        gamma8_nat = 10**g8_exp
        Pee, Pee_bar = P_combined_msc_E(2, D_kpc, Enu, gamma3_nat, gamma8_nat, mix[:2], thetaz, oscil_par, detector)
    
    elif mix == 'NH(m) + OQS-E-2.5' or mix == 'IH(m) + OQS-E-2.5':
        g3_exp, g8_exp, theta12, Deltam21 = par
        oscil_par = theta12, Deltam21
        gamma3_nat = 10**g3_exp
        gamma8_nat = 10**g8_exp
        Pee, Pee_bar = P_combined_msc_E(2.5, D_kpc, Enu, gamma3_nat, gamma8_nat, mix[:2], thetaz, oscil_par, detector)

    elif mix == 'NH + OQS-loss-E' or mix == 'IH + OQS-loss-E':
        g_exp, theta12 = par
        gamma_nat = 10**g_exp
        Pee, Pee_bar, Pme, Pte, Pme_bar, Pte_bar, Pmm, Pmt, Ptt, Ptm, Pem_bar, Pet_bar, Pmm_bar, Pmt_bar, Ptt_bar, Ptm_bar = Pab_vacuum_nu_loss_E(2, Enu, D_kpc, gamma_nat, theta12, mix[:2])

    elif mix == 'NH + OQS-loss-E-2.5' or mix == 'IH + OQS-loss-E-2.5':
        g_exp, theta12 = par
        gamma_nat = 10**g_exp
        Pee, Pee_bar, Pme, Pte, Pme_bar, Pte_bar, Pmm, Pmt, Ptt, Ptm, Pem_bar, Pet_bar, Pmm_bar, Pmt_bar, Ptt_bar, Ptm_bar = Pab_vacuum_nu_loss_E(2.5, Enu, D_kpc, gamma_nat, theta12, mix[:2])

    elif mix == 'NH(m) + OQS-loss-E' or mix == 'IH(m) + OQS-loss-E':
        g_exp, theta12, Deltam21 = par
        oscil_par = theta12, Deltam21
        gamma_nat = 10**g_exp
        Pee, Pee_bar, Pme, Pte, Pme_bar, Pte_bar, Pmm, Pmt, Ptt, Ptm, Pem_bar, Pet_bar, Pmm_bar, Pmt_bar, Ptt_bar, Ptm_bar = P_combined_loss_E(2, D_kpc, Enu, gamma_nat, mix[:2], thetaz, oscil_par, detector)

    elif mix == 'NH(m) + OQS-loss-E-2.5' or mix == 'IH(m) + OQS-loss-E-2.5':
        g_exp, theta12, Deltam21 = par
        oscil_par = theta12, Deltam21
        gamma_nat = 10**g_exp
        Pee, Pee_bar, Pme, Pte, Pme_bar, Pte_bar, Pmm, Pmt, Ptt, Ptm, Pem_bar, Pet_bar, Pmm_bar, Pmt_bar, Ptt_bar, Ptm_bar = P_combined_loss_E(2.5, D_kpc, Enu, gamma_nat, mix[:2], thetaz, oscil_par, detector)

    if 'loss' in mix:
        Fx0_bar = Fx0
        Fe = Fe0 * Pee + Fx0 * (Pme + Pte)
        Fe_bar = Fe0_bar * Pee_bar + Fx0 * (Pme_bar + Pte_bar)
        Fx_p = Fe0 * (Pme + Pte) + Fx0 * (Pmm + Pmt + Ptt + Ptm)
        Fx_bar_p = Fe0_bar * (Pem_bar + Pet_bar) + Fx0_bar * (Pmm_bar + Pmt_bar + Ptt_bar + Ptm_bar)
        Fx = Fx_p + Fx_bar_p
        Fx = Fx/4
    else:
        Fe = Pee * Fe0 + (1 - Pee) * Fx0
        Fe_bar = Pee_bar * Fe0_bar + (1 - Pee_bar) * Fx0
        Fx = 1/4 * ((1 - Pee) * Fe0 + (2 + Pee + Pee_bar) * Fx0 + (1 - Pee_bar) * Fe0_bar)

    if detector == 'HK':
        if channel == 'ES':
            #work function has effective cross section
            Enu_interp_hk_es, wnue, wnumu, wantinue, wantinumu = read_work_function(bin_i, 'hk', 'elastic', hierarchy, trigger_l)
            wnue_func = interp1d(Enu_interp_hk_es, wnue)
            wnumu_func = interp1d(Enu_interp_hk_es, wnumu)
            wantinue_func = interp1d(Enu_interp_hk_es, wantinue)
            wantinumu_func = interp1d(Enu_interp_hk_es, wantinumu)
            df = Fe * wnue_func(Enu) + Fe_bar * wantinue_func(Enu) + 2*Fx * wnumu_func(Enu) + 2*Fx * wantinumu_func(Enu)
            d = Ne_HK * efficiency_HK_ES * df
            return d
            
        elif channel == 'IBD':
            Enu_interp_hk_ibd, w_ibd = read_work_function(bin_i, 'hk', 'ibd', hierarchy, trigger_l)
            w_ibd_func = interp1d(Enu_interp_hk_ibd, w_ibd)
            df = Fe_bar * sigma_IBD(Enu)
            d = Np_HK * efficiency_HK_IBD * df * w_ibd_func(Enu)
            return d
    
    elif detector == 'JN':
        Enu_interp_jn_ibd, w_ibd = read_work_function(bin_i, 'jn', 'ibd', hierarchy, trigger_l)
        w_jn_ibd_func = np.interp(Enu, Enu_interp_jn_ibd, w_ibd)
        df = Fe_bar * sigma_IBD(Enu) * w_jn_ibd_func
        d = Np_JUNO * efficiency_JUNO_IBD * df
        return d

    elif detector == 'DUNE':
        Enu_interp_dune, w_nueAr = read_work_function(bin_i, 'dune', 'nue-Ar', hierarchy, trigger_l)
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
