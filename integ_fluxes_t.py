import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.special import gamma
from scipy.integrate import simpson

#https://arxiv.org/pdf/astro-ph/0308228.pdf
def flux(E, L, Emean, Emean_square):
    alpha = (2 * Emean**2 - Emean_square)/(Emean_square - Emean**2)
    c = (alpha+1)**(-(alpha+1)) * gamma(alpha+1) * Emean
    #spectrum
    f = 1/c * (E/Emean)**alpha * np.exp(-(alpha+1)*E/Emean) #1/MeV
    erg_to_MeV = 624151
    # kpc_to_cm = 3.086e21
    # D = D * kpc_to_cm
    L = L * erg_to_MeV * 10**51
    #flux
    flux = f * L/Emean#/(4*math.pi*D**2) #1/MeV 1/MeV MeV/s 1/cm^2 = 1/(MeV cm^2 s)
    return flux

def create_fluxes_t():
    models = ['LS220-s11.2c','LS220-s27.0c','LS220-s27.0co','Shen-s11.2c']
    nu_types = ['nubar_e','nu_e','nu_x']

    for model in models:
        t_data, E_data, E2_data, Lum_data, alpha_data = [],[],[],[],[]
        #loop over flavors
        for i in range(3):
            path = 'data/neutrinos-' + model + '/neutrino_signal_' + nu_types[i] + '-' + model + '.data'
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
        
        Fe0_bar_list, Fe0_list, Fx0_list = [],[],[]

        t_list = np.linspace(0,0.030,301)
        Enu_list = np.linspace(0.1,61,301)

        for t in t_list:
            nu = 0 #antinue
            Emean_func = interp1d(t_data[nu], E_data[nu])
            E2mean_func = interp1d(t_data[nu], E2_data[nu])
            Lum_func = interp1d(t_data[nu], Lum_data[nu])
            Fe0_bar = simpson(flux(Enu_list, Lum_func(t), Emean_func(t), E2mean_func(t)), Enu_list)
            Fe0_bar_list.append(Fe0_bar)

            nu = 1 #nue
            Emean_func = interp1d(t_data[nu], E_data[nu])
            E2mean_func = interp1d(t_data[nu], E2_data[nu])
            Lum_func = interp1d(t_data[nu], Lum_data[nu])
            Fe0 = simpson(flux(Enu_list, Lum_func(t), Emean_func(t), E2mean_func(t)), Enu_list)
            Fe0_list.append(Fe0)

            nu = 2 #nux
            Emean_func = interp1d(t_data[nu], E_data[nu])
            E2mean_func = interp1d(t_data[nu], E2_data[nu])
            Lum_func = interp1d(t_data[nu], Lum_data[nu])
            Fx0 = simpson(flux(Enu_list, Lum_func(t), Emean_func(t), E2mean_func(t)), Enu_list)
            Fx0_list.append(Fx0)

        np.save('data/integ_fluxes/Fe0_bar_%s_t'%model, Fe0_bar_list)
        np.save('data/integ_fluxes/Fe0_%s_t'%model, Fe0_list)
        np.save('data/integ_fluxes/Fx0_%s_t'%model, Fx0_list)

# create_fluxes_t()


def interp_fluxes_t(model, t):
    Fe0_bar_list = np.load('data/integ_fluxes/Fe0_bar_%s_t.npy'%model)
    Fe0_list = np.load('data/integ_fluxes/Fe0_%s_t.npy'%model)
    Fx0_list = np.load('data/integ_fluxes/Fx0_%s_t.npy'%model)
    t_list = np.linspace(0,0.030,301)

    Fe0_bar = np.interp(t, t_list, Fe0_bar_list)
    Fe0 = np.interp(t, t_list, Fe0_list)
    Fx0 = np.interp(t, t_list, Fx0_list)

    return Fe0_bar, Fe0, Fx0
