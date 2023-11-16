import numpy as np

def interp_sigma_ES(Enu):
    Emin,Emax = 2,61
    Enu_list = np.linspace(Emin,Emax,201)

    sigma_nue = np.load('data/interp_sigma_ES_pi/sigma_ES_integrated_nue.npy')
    sigma_nuebar = np.load('data/interp_sigma_ES_pi/sigma_ES_integrated_nuebar.npy')
    sigma_numu = np.load('data/interp_sigma_ES_pi/sigma_ES_integrated_numu.npy')
    sigma_numubar = np.load('data/interp_sigma_ES_pi/sigma_ES_integrated_numubar.npy')

    sigma_nue_func = np.interp(Enu, Enu_list, sigma_nue)
    sigma_nuebar_func = np.interp(Enu, Enu_list, sigma_nuebar)
    sigma_numu_func = np.interp(Enu, Enu_list, sigma_numu)
    sigma_numubar_func = np.interp(Enu, Enu_list, sigma_numubar)

    return sigma_nue_func, sigma_nuebar_func, sigma_numu_func, sigma_numubar_func