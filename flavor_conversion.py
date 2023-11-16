import numpy as np
from numpy import sqrt,exp
import math

#Mixing parameters-----------

Deltam21 = 7.42e-5 #eV² +0.21 -0.20
Deltam31 = 2.510e-3 #NH #eV² +-0.027
Deltam32 = Deltam31 - Deltam21

theta12 = 33.45 #+0.77 -0.75 NuFIT 2021 https://arxiv.org/pdf/2111.03086.pdf
theta13 = 8.62  #+-0.12 NuFIT 2021 https://arxiv.org/pdf/2111.03086.pdf
theta23 = 49.2 #+1.0 -1.3 NuFIT 2021 https://arxiv.org/pdf/2111.03086.pdf
theta12 = theta12*math.pi/180
theta13 = theta13*math.pi/180
theta23 = theta23*math.pi/180
c12=np.cos(theta12)
s12=np.sin(theta12)
c13=np.cos(theta13)
s13=np.sin(theta13)
c23=np.cos(theta23)
s23=np.sin(theta23)

U23 = np.array([ [1,0,0],[0,c23,s23],[0,-s23,c23] ])
U13 = np.array([ [c13,0,s13],[0,1,0],[-s13,0,c13] ])
U12 = np.array([ [c12,s12,0],[-s12,c12,0],[0,0,1] ])
U3 = U23 @ U13 @ U12
U3_dag = np.transpose(U3)

#COMPLETELY ADIABATIC PROPAGATION
#CALCULATING SURVIVOR PROBABILITY NH-----------------------------------------------------

#survivor prob. MSW NH electron antineutrino
def P_NH_antie():
    P = U3[0][0]**2
    return P

#survivor prob. MSW NH electron neutrino
def P_NH_nue():
    P = U3[0][2]**2
    return P

#CALCULATING SURVIVOR PROBABILITY IH-----------------------------------------------------

#survivor prob. MSW IH electron antineutrino
def P_IH_antie():
    P = U3[0][2]**2
    return P

#survivor prob. MSW IH electron neutrino
def P_IH_nue():
    P = U3[0][1]**2
    return P

########################### OQS MSC for diag. D and Gamma = Gamma(E**n) (Gamma = Gamma0 E**n) ################################

#decoherence for SU(3) VACUUM diagonal D matrix
#gamma_nat in eV
def Pee_msc_vacuum_E(n, E, D_kpc, gamma3_nat, gamma8_nat, theta12, mix):
    E = E*1e6 #eV
    gamma3_nat = gamma3_nat * E**n
    gamma8_nat = gamma8_nat * E**n
    gamma3_kpc = gamma3_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    gamma8_kpc = gamma8_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    P11 = 1/3 + 1/2 * np.exp(-(gamma3_kpc + gamma8_kpc/3) * D_kpc) + 1/6 * np.exp(-gamma8_kpc * D_kpc)
    P12 = 1/3 - 1/2 * np.exp(-(gamma3_kpc + gamma8_kpc/3) * D_kpc) + 1/6 * np.exp(-gamma8_kpc * D_kpc)
    P21= P12
    P22 = P11
    P31 = 1/3 - 1/3 * np.exp(-gamma8_kpc * D_kpc)
    P23 = P31
    P13 = P31
    P32 = P31
    P33 = 1/3 + 2/3 * np.exp(-gamma8_kpc * D_kpc)

    #Theta is already in rad
    c12=np.cos(theta12)
    s12=np.sin(theta12)
    c13=np.cos(theta13)
    s13=np.sin(theta13)
    P1e = c12**2 * c13**2
    P2e = s12**2 * c13**2
    P3e = s13**2

    if mix == 'NH':
        # NH:
        Pe3m = 1
        Pee = Pe3m*P33*P3e + Pe3m*P32*P2e + Pe3m*P31*P1e
    elif mix == 'IH':
        # IH:
        Pe2m = 1
        Pee = Pe2m*P22*P2e + Pe2m*P21*P1e + Pe2m*P23*P3e
    else:
        return
    return Pee

#gamma_nat in eV
def Pee_bar_msc_vacuum_E(n, E, D_kpc, gamma3_nat, gamma8_nat, theta12, mix):
    E = E*1e6 #eV
    gamma3_nat = gamma3_nat * E**n
    gamma8_nat = gamma8_nat * E**n
    gamma3_kpc = gamma3_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    gamma8_kpc = gamma8_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    P11 = 1/3 + 1/2 * np.exp(-(gamma3_kpc + gamma8_kpc/3) * D_kpc) + 1/6 * np.exp(-gamma8_kpc * D_kpc)
    P12 = 1/3 - 1/2 * np.exp(-(gamma3_kpc + gamma8_kpc/3) * D_kpc) + 1/6 * np.exp(-gamma8_kpc * D_kpc)
    P21= P12
    P22 = P11
    P31 = 1/3 - 1/3 * np.exp(-gamma8_kpc * D_kpc)
    P23 = P31
    P13 = P31
    P32 = P31
    P33 = 1/3 + 2/3 * np.exp(-gamma8_kpc * D_kpc)

    #Theta is already in rad
    c12=np.cos(theta12)
    s12=np.sin(theta12)
    c13=np.cos(theta13)
    s13=np.sin(theta13)
    P1e = c12**2 * c13**2
    P2e = s12**2 * c13**2
    P3e = s13**2

    if mix == 'NH':
        # NH:
        Pe1m = 1
        # P1e, P2e, P3e = Pei_vec
        Pee = Pe1m*P11*P1e + Pe1m*P12*P2e + Pe1m*P13*P3e
    elif mix == 'IH':
        # IH:
        Pe3m = 1
        Pee = Pe3m*P33*P3e + Pe3m*P31*P1e + Pe3m*P32*P2e
    else:
        return
    return Pee

def P11_E_func(n, E, D_kpc, gamma3_nat, gamma8_nat):
    E = E*1e6 #eV
    gamma3_nat = gamma3_nat * E**n
    gamma8_nat = gamma8_nat * E**n
    gamma3_kpc = gamma3_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    gamma8_kpc = gamma8_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    P11 = 1/3 + 1/2 * np.exp(-(gamma3_kpc + gamma8_kpc/3) * D_kpc) + 1/6 * np.exp(-gamma8_kpc * D_kpc)
    return P11

def P12_E_func(n, E, D_kpc, gamma3_nat, gamma8_nat):
    E = E*1e6 #eV
    gamma3_nat = gamma3_nat * E**n
    gamma8_nat = gamma8_nat * E**n
    gamma3_kpc = gamma3_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    gamma8_kpc = gamma8_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    P12 = 1/3 - 1/2 * np.exp(-(gamma3_kpc + gamma8_kpc/3) * D_kpc) + 1/6 * np.exp(-gamma8_kpc * D_kpc)
    return P12

def P31_E_func(n, E, D_kpc, gamma8_nat):
    E = E*1e6 #eV
    # gamma3_nat = gamma3_nat * E**n
    gamma8_nat = gamma8_nat * E**n
    gamma8_kpc = gamma8_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    P31 = 1/3 - 1/3 * np.exp(-gamma8_kpc * D_kpc)
    return P31

def P33_E_func(n, E, D_kpc, gamma8_nat):
    E = E*1e6 #eV
    # gamma3_nat = gamma3_nat * E**n
    gamma8_nat = gamma8_nat * E**n
    gamma8_kpc = gamma8_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    P33 = 1/3 + 2/3 * np.exp(-gamma8_kpc * D_kpc)
    return P33

# P21 = P12 
# P22 = P11
# P23 = P31
# P13 = P31
# P32 = P31
#####################################################################################################


########################### OQS MSC for non diagonal D and Gamma = Gamma(E**n) (Gamma = Gamma0 E**n) ################################

#gamma_nat in eV
def Pee_msc_non_diag_vacuum_E(n, E, D_kpc, gamma_nat, beta38_nat, theta12, mix):
    P21 = P21_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat)
    P22 = P22_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat)
    P23 = P23_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat)
    P31 = P31_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat)
    P32 = P32_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat)
    P33 = P33_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat)

    #Theta is already in rad
    c12=np.cos(theta12)
    s12=np.sin(theta12)
    c13=np.cos(theta13)
    s13=np.sin(theta13)
    P1e = c12**2 * c13**2
    P2e = s12**2 * c13**2
    P3e = s13**2

    if mix == 'NH':
        # NH:
        Pe3m = 1
        Pee = Pe3m*P33*P3e + Pe3m*P32*P2e + Pe3m*P31*P1e
    elif mix == 'IH':
        # IH:
        Pe2m = 1
        Pee = Pe2m*P22*P2e + Pe2m*P21*P1e + Pe2m*P23*P3e
    else:
        return
    return Pee

#gamma_nat in eV
def Pee_bar_msc_non_diag_vacuum_E(n, E, D_kpc, gamma_nat, beta38_nat, theta12, mix):
    P11 = P11_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat)
    P12 = P12_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat)
    P13 = P13_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat)
    P31 = P31_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat)
    P32 = P32_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat)
    P33 = P33_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat)

    #Theta is already in rad
    c12=np.cos(theta12)
    s12=np.sin(theta12)
    c13=np.cos(theta13)
    s13=np.sin(theta13)
    P1e = c12**2 * c13**2
    P2e = s12**2 * c13**2
    P3e = s13**2

    if mix == 'NH':
        # NH:
        Pe1m = 1
        # P1e, P2e, P3e = Pei_vec
        Pee = Pe1m*P11*P1e + Pe1m*P12*P2e + Pe1m*P13*P3e
    elif mix == 'IH':
        # IH:
        Pe3m = 1
        Pee = Pe3m*P33*P3e + Pe3m*P31*P1e + Pe3m*P32*P2e
    else:
        return
    return Pee


def P11_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat):
    E = E*1e6 #eV
    gamma_nat = gamma_nat * E**n
    beta38_nat = beta38_nat * E**n
    Gamma = gamma_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    beta38 = beta38_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    return (2*exp(Gamma*D_kpc) + sqrt(3)*exp(beta38*D_kpc) + 2*exp(beta38*D_kpc) - sqrt(3)*exp(-beta38*D_kpc) + 2*exp(-beta38*D_kpc))*exp(-Gamma*D_kpc)/6

def P12_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat):
    E = E*1e6 #eV
    gamma_nat = gamma_nat * E**n
    beta38_nat = beta38_nat * E**n
    Gamma = gamma_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    beta38 = beta38_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    return (2*exp(Gamma*D_kpc) - exp(beta38*D_kpc) - exp(-beta38*D_kpc))*exp(-Gamma*D_kpc)/6

def P13_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat):
    E = E*1e6 #eV
    gamma_nat = gamma_nat * E**n
    beta38_nat = beta38_nat * E**n
    Gamma = gamma_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    beta38 = beta38_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    return (2*exp(Gamma*D_kpc) - sqrt(3)*exp(beta38*D_kpc) - exp(beta38*D_kpc) - exp(-beta38*D_kpc) + sqrt(3)*exp(-beta38*D_kpc))*exp(-Gamma*D_kpc)/6

def P21_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat):
    E = E*1e6 #eV
    gamma_nat = gamma_nat * E**n
    beta38_nat = beta38_nat * E**n
    Gamma = gamma_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    beta38 = beta38_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    return (2*exp(Gamma*D_kpc) - exp(beta38*D_kpc) - exp(-beta38*D_kpc))*exp(-Gamma*D_kpc)/6

def P22_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat):
    E = E*1e6 #eV
    gamma_nat = gamma_nat * E**n
    beta38_nat = beta38_nat * E**n
    Gamma = gamma_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    beta38 = beta38_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    return sqrt(3)*((sqrt(3) + 3)*exp(-D_kpc*(Gamma + beta38))/12 - (3 - sqrt(3))*exp(-D_kpc*(Gamma - beta38))/12)/3 + 1/3 + (sqrt(3) + 3)*exp(-D_kpc*(Gamma + beta38))/12 + (3 - sqrt(3))*exp(-D_kpc*(Gamma - beta38))/12

def P23_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat):
    E = E*1e6 #eV
    gamma_nat = gamma_nat * E**n
    beta38_nat = beta38_nat * E**n
    Gamma = gamma_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    beta38 = beta38_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    return (2*exp(Gamma*D_kpc) - exp(beta38*D_kpc) + sqrt(3)*exp(beta38*D_kpc) - sqrt(3)*exp(-beta38*D_kpc) - exp(-beta38*D_kpc))*exp(-Gamma*D_kpc)/6

def P31_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat):
    E = E*1e6 #eV
    gamma_nat = gamma_nat * E**n
    beta38_nat = beta38_nat * E**n
    Gamma = gamma_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    beta38 = beta38_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    return (2*exp(Gamma*D_kpc) - sqrt(3)*exp(beta38*D_kpc) - exp(beta38*D_kpc) - exp(-beta38*D_kpc) + sqrt(3)*exp(-beta38*D_kpc))*exp(-Gamma*D_kpc)/6

def P32_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat):
    E = E*1e6 #eV
    gamma_nat = gamma_nat * E**n
    beta38_nat = beta38_nat * E**n
    Gamma = gamma_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    beta38 = beta38_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    return (2*exp(Gamma*D_kpc) - sqrt(3)*exp(beta38*D_kpc) - exp(beta38*D_kpc) - exp(-beta38*D_kpc) + sqrt(3)*exp(-beta38*D_kpc))*exp(-Gamma*D_kpc)/6

def P33_non_diag_E_func(n, E, D_kpc, gamma_nat, beta38_nat):
    E = E*1e6 #eV
    gamma_nat = gamma_nat * E**n
    beta38_nat = beta38_nat * E**n
    Gamma = gamma_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    beta38 = beta38_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    return (exp(Gamma*D_kpc) + exp(beta38*D_kpc) + exp(-beta38*D_kpc))*exp(-Gamma*D_kpc)/3

#####################################################################################################



########################### ENERGY CONSERVED OQS MSC for diagonal D and Gamma = Gamma(E**n) (Gamma = Gamma0 E**n) ################################
# In this case, the Pij regards the propagation in the SN 

#gamma_nat in eV
def Pee_msc_vacuum_conserved_E(n, E, gamma_nat, theta12, mix):
    nu_nubar = 'nue'
    P21 = Pij_conserved_E_interps_func(n, E, gamma_nat, '21', mix, nu_nubar)
    P22 = Pij_conserved_E_interps_func(n, E, gamma_nat, '22', mix, nu_nubar)
    P23 = Pij_conserved_E_interps_func(n, E, gamma_nat, '23', mix, nu_nubar)
    P31 = Pij_conserved_E_interps_func(n, E, gamma_nat, '31', mix, nu_nubar)
    P32 = Pij_conserved_E_interps_func(n, E, gamma_nat, '32', mix, nu_nubar)
    P33 = Pij_conserved_E_interps_func(n, E, gamma_nat, '33', mix, nu_nubar)

    #Theta is already in rad
    c12=np.cos(theta12)
    s12=np.sin(theta12)
    c13=np.cos(theta13)
    s13=np.sin(theta13)
    P1e = c12**2 * c13**2
    P2e = s12**2 * c13**2
    P3e = s13**2

    if mix == 'NH':
        # NH:
        Pe3m = 1
        Pee = Pe3m*P33*P3e + Pe3m*P32*P2e + Pe3m*P31*P1e
    elif mix == 'IH':
        # IH:
        Pe2m = 1
        Pee = Pe2m*P22*P2e + Pe2m*P21*P1e + Pe2m*P23*P3e
    else:
        return
    return Pee

#gamma_nat in eV
def Pee_bar_msc_vacuum_conserved_E(n, E, gamma_nat, theta12, mix):
    nu_nubar = 'nuebar'
    P11 = Pij_conserved_E_interps_func(n, E, gamma_nat, '11', mix, nu_nubar)
    P12 = Pij_conserved_E_interps_func(n, E, gamma_nat, '12', mix, nu_nubar)
    P13 = Pij_conserved_E_interps_func(n, E, gamma_nat, '13', mix, nu_nubar)
    P31 = Pij_conserved_E_interps_func(n, E, gamma_nat, '31', mix, nu_nubar)
    P32 = Pij_conserved_E_interps_func(n, E, gamma_nat, '32', mix, nu_nubar)
    P33 = Pij_conserved_E_interps_func(n, E, gamma_nat, '33', mix, nu_nubar)

    #Theta is already in rad
    c12=np.cos(theta12)
    s12=np.sin(theta12)
    c13=np.cos(theta13)
    s13=np.sin(theta13)
    P1e = c12**2 * c13**2
    P2e = s12**2 * c13**2
    P3e = s13**2

    if mix == 'NH':
        # NH:
        Pe1m = 1
        # P1e, P2e, P3e = Pei_vec
        Pee = Pe1m*P11*P1e + Pe1m*P12*P2e + Pe1m*P13*P3e
    elif mix == 'IH':
        # IH:
        Pe3m = 1
        Pee = Pe3m*P33*P3e + Pe3m*P31*P1e + Pe3m*P32*P2e
    else:
        return
    return Pee


def Pij_conserved_E_interps_func(n, E, gamma_nat, ij, mix, nu_nubar):
    from scipy.interpolate import RectBivariateSpline
    g_exp = np.log10(gamma_nat)
    file_name = 'saved_Pij_sn_matter/P%s_%s_%s_n%i_%iE_%ig.npy'%(ij, mix, nu_nubar, n, 201, 201)
    Pij_load = np.load(file_name)
    E_list = np.linspace(0.1, 61, 201)

    if n == 0:
        g_list = np.linspace(-23, -9, 201)
        Pij_interp = RectBivariateSpline(E_list, g_list, np.fliplr(Pij_load))
    elif n == 2:
        g_list = np.linspace(-35, -19, 201)
        Pij_interp = RectBivariateSpline(E_list, g_list, Pij_load)
    elif n == 2.5:
        g_list = np.linspace(-40, -20, 201)
        Pij_interp = RectBivariateSpline(E_list, g_list, Pij_load)
    
    return Pij_interp(E, g_exp)[0]

#####################################################################################################



############################################# OQS+Loss ##############################################

def Pab_vacuum_nu_loss_E(n, E, D_kpc, gamma_nat, theta12, mix):
    E = E*1e6 #eV
    gamma_nat = gamma_nat * E**n
    gamma_kpc = gamma_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    Pii = np.exp(-gamma_kpc * D_kpc)

    theta13 = 8.62  #+-0.12 NuFIT 2021 https://arxiv.org/pdf/2111.03086.pdf
    theta23 = 49.2 #+1.0 -1.3 NuFIT 2021 https://arxiv.org/pdf/2111.03086.pdf
    theta13 = theta13*math.pi/180
    theta23 = theta23*math.pi/180
    c12=np.cos(theta12)
    s12=np.sin(theta12)
    c13=np.cos(theta13)
    s13=np.sin(theta13)
    c23=np.cos(theta23)
    s23=np.sin(theta23)

    U23 = np.array([ [1,0,0],[0,c23,s23],[0,-s23,c23] ])
    U13 = np.array([ [c13,0,s13],[0,1,0],[-s13,0,c13] ])
    U12 = np.array([ [c12,s12,0],[-s12,c12,0],[0,0,1] ])
    U3 = U23 @ U13 @ U12

    #Muon-tau mixing
    Umt = np.array([[c23,s23],[-s23,c23]])

    if mix == 'NH':
        Pee = Pii*U3[0,2]**2
        Pee_bar = Pii*U3[0,0]**2
        Pet_bar = Pii*U3[2,0]**2
        Pem_bar = Pii*U3[1,0]**2

        Pme = Umt[0,0]**2 * Pii * U3[0,0]**2 + Umt[0,1]**2 * Pii * U3[0,1]**2
        Pte = Umt[1,0]**2 * Pii * U3[0,0]**2 + Umt[1,1]**2 * Pii * U3[0,1]**2
        Pmm = Umt[0,0]**2 * Pii * U3[1,0]**2 + Umt[0,1]**2 * Pii * U3[1,1]**2
        Pmt = Umt[0,0]**2 * Pii * U3[2,0]**2 + Umt[0,1]**2 * Pii * U3[2,1]**2
        Ptt = Umt[1,0]**2 * Pii * U3[2,0]**2 + Umt[1,1]**2 * Pii * U3[2,1]**2
        Ptm = Umt[1,0]**2 * Pii * U3[1,0]**2 + Umt[1,1]**2 * Pii * U3[1,1]**2
        Pme_bar = Umt[0,0]**2 * Pii * U3[0,1]**2 + Umt[0,1]**2 * Pii * U3[0,2]**2
        Pte_bar = Umt[1,0]**2 * Pii * U3[0,1]**2 + Umt[1,1]**2 * Pii * U3[0,2]**2
        Pmm_bar = Umt[0,0]**2 * Pii * U3[1,1]**2 + Umt[0,1]**2 * Pii * U3[1,2]**2
        Pmt_bar = Umt[0,0]**2 * Pii * U3[2,1]**2 + Umt[0,1]**2 * Pii * U3[2,2]**2
        Ptt_bar = Umt[1,0]**2 * Pii * U3[2,1]**2 + Umt[1,1]**2 * Pii * U3[2,2]**2
        Ptm_bar = Umt[1,0]**2 * Pii * U3[1,1]**2 + Umt[1,1]**2 * Pii * U3[1,2]**2
    
    elif mix == 'IH':
        Pee = Pii*U3[0,1]**2
        Pee_bar = Pii*U3[0,2]**2
        Pem_bar = Pii*U3[1,2]**2
        Pet_bar = Pii*U3[2,2]**2

        Pme = Umt[0,0]**2 * Pii * U3[0,0]**2 + Umt[0,1]**2 * Pii * U3[0,2]**2
        Pte = Umt[1,0]**2 * Pii * U3[0,0]**2 + Umt[1,1]**2 * Pii * U3[0,2]**2
        Pmm = Umt[0,0]**2 * Pii * U3[1,0]**2 + Umt[0,1]**2 * Pii * U3[1,2]**2
        Pmt = Umt[0,0]**2 * Pii * U3[2,0]**2 + Umt[0,1]**2 * Pii * U3[2,2]**2
        Ptt = Umt[1,0]**2 * Pii * U3[2,0]**2 + Umt[1,1]**2 * Pii * U3[2,2]**2
        Ptm = Umt[1,0]**2 * Pii * U3[1,0]**2 + Umt[1,1]**2 * Pii * U3[1,2]**2
        Pme_bar = Umt[0,0]**2 * Pii * U3[0,0]**2 + Umt[0,1]**2 * Pii * U3[0,1]**2
        Pte_bar = Umt[1,0]**2 * Pii * U3[0,0]**2 + Umt[1,1]**2 * Pii * U3[0,1]**2
        Pmm_bar = Umt[0,0]**2 * Pii * U3[1,0]**2 + Umt[0,1]**2 * Pii * U3[1,1]**2
        Pmt_bar = Umt[0,0]**2 * Pii * U3[2,0]**2 + Umt[0,1]**2 * Pii * U3[2,1]**2
        Ptt_bar = Umt[1,0]**2 * Pii * U3[2,0]**2 + Umt[1,1]**2 * Pii * U3[2,1]**2
        Ptm_bar = Umt[1,0]**2 * Pii * U3[1,0]**2 + Umt[1,1]**2 * Pii * U3[1,1]**2

    else:
        return
    return Pee, Pee_bar, Pme, Pte, Pme_bar, Pte_bar, Pmm, Pmt, Ptt, Ptm, Pem_bar, Pet_bar, Pmm_bar, Pmt_bar, Ptt_bar, Ptm_bar



def Pee_deco_vac_basis_loss_E(n, E, D_kpc, gamma_nat, theta12, mix):
    E = E*1e6 #eV
    gamma_nat = gamma_nat * E**n
    gamma_kpc = gamma_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    P11 = np.exp(-gamma_kpc * D_kpc)
    P12 = 0
    P21 = 0
    P22 = np.exp(-gamma_kpc * D_kpc)
    P31 = 0
    P23 = 0
    P13 = 0
    P32 = 0
    P33 = np.exp(-gamma_kpc * D_kpc)

    #Theta is already in rad
    c12=np.cos(theta12)
    s12=np.sin(theta12)
    # c13=np.cos(theta13)
    # s13=np.sin(theta13)
    P1e = c12**2 * c13**2
    P2e = s12**2 * c13**2
    P3e = s13**2

    if mix == 'NH':
        # NH:
        Pe3m = 1
        Pee = Pe3m*P33*P3e + Pe3m*P32*P2e + Pe3m*P31*P1e
    elif mix == 'IH':
        # IH:
        Pe2m = 1
        Pee = Pe2m*P22*P2e + Pe2m*P21*P1e + Pe2m*P23*P3e
    else:
        return
    return Pee


#gamma_nat in eV
def Pee_bar_deco_vac_basis_loss_E(n, E, D_kpc, gamma_nat, theta12, mix):
    E = E*1e6 #eV
    gamma_nat = gamma_nat * E**n
    gamma_kpc = gamma_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    P11 = np.exp(-gamma_kpc * D_kpc)
    P12 = 0
    P21 = 0
    # P22 = np.exp(-2*gamma_kpc * D_kpc)
    P31 = 0
    P23 = 0
    P13 = 0
    P32 = 0
    P33 = np.exp(-gamma_kpc * D_kpc)

    #Theta is already in rad
    c12=np.cos(theta12)
    s12=np.sin(theta12)
    c13=np.cos(theta13)
    s13=np.sin(theta13)
    P1e = c12**2 * c13**2
    P2e = s12**2 * c13**2
    P3e = s13**2

    if mix == 'NH':
        # NH:
        Pe1m = 1
        Pee = Pe1m*P11*P1e + Pe1m*P12*P2e + Pe1m*P13*P3e
    elif mix == 'IH':
        # IH:
        Pe3m = 1
        Pee = Pe3m*P33*P3e + Pe3m*P31*P1e + Pe3m*P32*P2e
    else:
        return
    return Pee


def P11_loss_E_func(n, E, D_kpc, gamma_nat):
    E = E*1e6 #eV
    gamma_nat = gamma_nat * E**n
    gamma_kpc = gamma_nat / (0.197e9 * 1e-15) * 3.086e19 # kpc-1
    P11 = np.exp(-gamma_kpc * D_kpc)
    return P11

# P21 = P12 feito
# P22 = P11 feito
# P23 = P31 feito
# P13 = P31 feito
# P32 = P31 feito
#########################################################################################



##############################################################################################
from Pie_earth_interps import P1e_nh_nue_180_func, P1e_nh_nue_160_func, P1e_nh_nue_140_func, P1e_nh_nue_120_func, P2e_nh_nue_180_func, P2e_nh_nue_160_func, P2e_nh_nue_140_func, P2e_nh_nue_120_func, P1e_ih_nue_180_func, P1e_ih_nue_160_func, P1e_ih_nue_140_func, P1e_ih_nue_120_func, P2e_ih_nue_180_func, P2e_ih_nue_160_func, P2e_ih_nue_140_func, P2e_ih_nue_120_func
from Pie_earth_interps import P1e_nh_nuebar_180_func, P1e_nh_nuebar_160_func, P1e_nh_nuebar_140_func, P1e_nh_nuebar_120_func, P2e_nh_nuebar_180_func, P2e_nh_nuebar_160_func, P2e_nh_nuebar_140_func, P2e_nh_nuebar_120_func, P1e_ih_nuebar_180_func, P1e_ih_nuebar_160_func, P1e_ih_nuebar_140_func, P1e_ih_nuebar_120_func, P2e_ih_nuebar_180_func, P2e_ih_nuebar_160_func, P2e_ih_nuebar_140_func, P2e_ih_nuebar_120_func

from Pie_earth_interps import P1e_nh_nue_97_func, P1e_nh_nue_95_func, P1e_nh_nue_129_6_func, P1e_nh_nue_129_3_func, P2e_nh_nue_97_func, P2e_nh_nue_95_func, P2e_nh_nue_129_6_func, P2e_nh_nue_129_3_func, P1e_ih_nue_97_func, P1e_ih_nue_95_func, P1e_ih_nue_129_6_func, P1e_ih_nue_129_3_func, P2e_ih_nue_97_func, P2e_ih_nue_95_func, P2e_ih_nue_129_6_func, P2e_ih_nue_129_3_func
from Pie_earth_interps import P1e_nh_nuebar_97_func, P1e_nh_nuebar_95_func, P1e_nh_nuebar_129_6_func, P1e_nh_nuebar_129_3_func, P2e_nh_nuebar_97_func, P2e_nh_nuebar_95_func, P2e_nh_nuebar_129_6_func, P2e_nh_nuebar_129_3_func, P1e_ih_nuebar_97_func, P1e_ih_nuebar_95_func, P1e_ih_nuebar_129_6_func, P1e_ih_nuebar_129_3_func, P2e_ih_nuebar_97_func, P2e_ih_nuebar_95_func, P2e_ih_nuebar_129_6_func, P2e_ih_nuebar_129_3_func

from Pie_earth_interps import P1e_nh_nue_105_func, P1e_nh_nue_120_8_func, P1e_nh_nue_140_func, P1e_nh_nue_146_6_func, P2e_nh_nue_105_func, P2e_nh_nue_120_8_func, P2e_nh_nue_146_6_func, P1e_ih_nue_105_func, P1e_ih_nue_120_8_func, P1e_ih_nue_146_6_func, P2e_ih_nue_105_func, P2e_ih_nue_120_8_func, P2e_ih_nue_146_6_func
from Pie_earth_interps import P1e_nh_nuebar_105_func, P1e_nh_nuebar_120_8_func, P1e_nh_nuebar_146_6_func, P2e_nh_nuebar_105_func, P2e_nh_nuebar_120_8_func, P2e_nh_nuebar_146_6_func, P1e_ih_nuebar_105_func, P1e_ih_nuebar_120_8_func, P1e_ih_nuebar_146_6_func, P2e_ih_nuebar_105_func, P2e_ih_nuebar_120_8_func, P2e_ih_nuebar_146_6_func


def Pie_earth_interp_func(w21_i, theta21_i, nu_nubar, mix, thetaz_dune, detector, state):

    Pie_dict = {'nue':{'NH':{180:{'DUNE':{1:P1e_nh_nue_180_func(w21_i, theta21_i), 
                                          2:P2e_nh_nue_180_func(w21_i, theta21_i)},
                                  'HK':{1:P1e_nh_nue_129_3_func(w21_i, theta21_i), 
                                        2:P2e_nh_nue_129_3_func(w21_i, theta21_i)},
                                  'JN':{1:P1e_nh_nue_146_6_func(w21_i, theta21_i), 
                                        2:P2e_nh_nue_146_6_func(w21_i, theta21_i)},},
                            160:{'DUNE':{1:P1e_nh_nue_160_func(w21_i, theta21_i), 
                                         2:P2e_nh_nue_160_func(w21_i, theta21_i)},
                                  'HK':{1:P1e_nh_nue_129_6_func(w21_i, theta21_i), 
                                        2:P2e_nh_nue_129_6_func(w21_i, theta21_i)},
                                  'JN':{1:P1e_nh_nue_120_8_func(w21_i, theta21_i), 
                                        2:P2e_nh_nue_120_8_func(w21_i, theta21_i)},},
                            140:{'DUNE':{1:P1e_nh_nue_140_func(w21_i, theta21_i), 
                                         2:P2e_nh_nue_140_func(w21_i, theta21_i)},
                                  'HK':{1:P1e_nh_nue_95_func(w21_i, theta21_i), 
                                        2:P2e_nh_nue_95_func(w21_i, theta21_i)},
                                  'JN':{1:P1e_nh_nue_105_func(w21_i, theta21_i), 
                                        2:P2e_nh_nue_105_func(w21_i, theta21_i)},},
                            120:{'DUNE':{1:P1e_nh_nue_120_func(w21_i, theta21_i), 
                                         2:P2e_nh_nue_120_func(w21_i, theta21_i)},
                                  'HK':{1:P1e_nh_nue_97_func(w21_i, theta21_i), 
                                        2:P2e_nh_nue_97_func(w21_i, theta21_i)},
                                  'JN':{1:abs(U3[0,0]**2), 
                                        2:abs(U3[0,1]**2)},},
                            0:{'DUNE':{1:abs(U3[0,0]**2), 
                                      2:abs(U3[0,1]**2)},
                                'HK':{1:P1e_nh_nue_95_func(w21_i, theta21_i), 
                                      2:P2e_nh_nue_95_func(w21_i, theta21_i)},
                                'JN':{1:P1e_nh_nue_105_func(w21_i, theta21_i), 
                                      2:P2e_nh_nue_105_func(w21_i, theta21_i)},},
                            320:{'DUNE':{1:abs(U3[0,0]**2), 
                                         2:abs(U3[0,1]**2)},
                                'HK':{1:P1e_nh_nue_129_6_func(w21_i, theta21_i), 
                                      2:P2e_nh_nue_129_6_func(w21_i, theta21_i)},
                                'JN':{1:P1e_nh_nue_146_6_func(w21_i, theta21_i), 
                                      2:P2e_nh_nue_146_6_func(w21_i, theta21_i)},}},
                       'IH':{180:{'DUNE':{1:P1e_ih_nue_180_func(w21_i, theta21_i), 
                                          2:P2e_ih_nue_180_func(w21_i, theta21_i)},
                                  'HK':{1:P1e_ih_nue_129_3_func(w21_i, theta21_i), 
                                        2:P2e_ih_nue_129_3_func(w21_i, theta21_i)},
                                  'JN':{1:P1e_ih_nue_146_6_func(w21_i, theta21_i), 
                                        2:P2e_ih_nue_146_6_func(w21_i, theta21_i)},},
                            160:{'DUNE':{1:P1e_ih_nue_160_func(w21_i, theta21_i), 
                                         2:P2e_ih_nue_160_func(w21_i, theta21_i)},
                                  'HK':{1:P1e_ih_nue_129_6_func(w21_i, theta21_i), 
                                        2:P2e_ih_nue_129_6_func(w21_i, theta21_i)},
                                  'JN':{1:P1e_ih_nue_120_8_func(w21_i, theta21_i), 
                                        2:P2e_ih_nue_120_8_func(w21_i, theta21_i)},},
                            140:{'DUNE':{1:P1e_ih_nue_140_func(w21_i, theta21_i), 
                                         2:P2e_ih_nue_140_func(w21_i, theta21_i)},
                                  'HK':{1:P1e_ih_nue_95_func(w21_i, theta21_i), 
                                        2:P2e_ih_nue_95_func(w21_i, theta21_i)},
                                  'JN':{1:P1e_ih_nue_105_func(w21_i, theta21_i), 
                                        2:P2e_ih_nue_105_func(w21_i, theta21_i)},},
                            120:{'DUNE':{1:P1e_ih_nue_120_func(w21_i, theta21_i), 
                                         2:P2e_ih_nue_120_func(w21_i, theta21_i)},
                                  'HK':{1:P1e_ih_nue_97_func(w21_i, theta21_i), 
                                        2:P2e_ih_nue_97_func(w21_i, theta21_i)},
                                  'JN':{1:abs(U3[0,0]**2), 
                                        2:abs(U3[0,1]**2)},},
                            0:{'DUNE':{1:abs(U3[0,0]**2), 
                                      2:abs(U3[0,1]**2)},
                                'HK':{1:P1e_ih_nue_95_func(w21_i, theta21_i), 
                                      2:P2e_ih_nue_95_func(w21_i, theta21_i)},
                                'JN':{1:P1e_ih_nue_105_func(w21_i, theta21_i), 
                                      2:P2e_ih_nue_105_func(w21_i, theta21_i)},},
                            320:{'DUNE':{1:abs(U3[0,0]**2), 
                                         2:abs(U3[0,1]**2)},
                                'HK':{1:P1e_ih_nue_129_6_func(w21_i, theta21_i), 
                                      2:P2e_ih_nue_129_6_func(w21_i, theta21_i)},
                                'JN':{1:P1e_ih_nue_146_6_func(w21_i, theta21_i), 
                                      2:P2e_ih_nue_146_6_func(w21_i, theta21_i)},}}},
                'nuebar':{'NH':{180:{'DUNE':{1:P1e_nh_nuebar_180_func(w21_i, theta21_i), 
                                          2:P2e_nh_nuebar_180_func(w21_i, theta21_i)},
                                  'HK':{1:P1e_nh_nuebar_129_3_func(w21_i, theta21_i), 
                                        2:P2e_nh_nuebar_129_3_func(w21_i, theta21_i)},
                                  'JN':{1:P1e_nh_nuebar_146_6_func(w21_i, theta21_i), 
                                        2:P2e_nh_nuebar_146_6_func(w21_i, theta21_i)},},
                            160:{'DUNE':{1:P1e_nh_nuebar_160_func(w21_i, theta21_i), 
                                         2:P2e_nh_nuebar_160_func(w21_i, theta21_i)},
                                  'HK':{1:P1e_nh_nuebar_129_6_func(w21_i, theta21_i), 
                                        2:P2e_nh_nuebar_129_6_func(w21_i, theta21_i)},
                                  'JN':{1:P1e_nh_nuebar_120_8_func(w21_i, theta21_i), 
                                        2:P2e_nh_nuebar_120_8_func(w21_i, theta21_i)},},
                            140:{'DUNE':{1:P1e_nh_nuebar_140_func(w21_i, theta21_i), 
                                         2:P2e_nh_nuebar_140_func(w21_i, theta21_i)},
                                  'HK':{1:P1e_nh_nuebar_95_func(w21_i, theta21_i), 
                                        2:P2e_nh_nuebar_95_func(w21_i, theta21_i)},
                                  'JN':{1:P1e_nh_nuebar_105_func(w21_i, theta21_i), 
                                        2:P2e_nh_nuebar_105_func(w21_i, theta21_i)},},
                            120:{'DUNE':{1:P1e_nh_nuebar_120_func(w21_i, theta21_i), 
                                         2:P2e_nh_nuebar_120_func(w21_i, theta21_i)},
                                  'HK':{1:P1e_nh_nuebar_97_func(w21_i, theta21_i), 
                                        2:P2e_nh_nuebar_97_func(w21_i, theta21_i)},
                                  'JN':{1:abs(U3[0,0]**2), 
                                        2:abs(U3[0,1]**2)},},
                            0:{'DUNE':{1:abs(U3[0,0]**2), 
                                      2:abs(U3[0,1]**2)},
                                'HK':{1:P1e_nh_nuebar_95_func(w21_i, theta21_i), 
                                      2:P2e_nh_nuebar_95_func(w21_i, theta21_i)},
                                'JN':{1:P1e_nh_nuebar_105_func(w21_i, theta21_i), 
                                      2:P2e_nh_nuebar_105_func(w21_i, theta21_i)},},
                            320:{'DUNE':{1:abs(U3[0,0]**2), 
                                         2:abs(U3[0,1]**2)},
                                'HK':{1:P1e_nh_nuebar_129_6_func(w21_i, theta21_i), 
                                      2:P2e_nh_nuebar_129_6_func(w21_i, theta21_i)},
                                'JN':{1:P1e_nh_nuebar_146_6_func(w21_i, theta21_i), 
                                      2:P2e_nh_nuebar_146_6_func(w21_i, theta21_i)},}},
                       'IH':{180:{'DUNE':{1:P1e_ih_nuebar_180_func(w21_i, theta21_i), 
                                          2:P2e_ih_nuebar_180_func(w21_i, theta21_i)},
                                  'HK':{1:P1e_ih_nuebar_129_3_func(w21_i, theta21_i), 
                                        2:P2e_ih_nuebar_129_3_func(w21_i, theta21_i)},
                                  'JN':{1:P1e_ih_nuebar_146_6_func(w21_i, theta21_i), 
                                        2:P2e_ih_nuebar_146_6_func(w21_i, theta21_i)},},
                            160:{'DUNE':{1:P1e_ih_nuebar_160_func(w21_i, theta21_i), 
                                         2:P2e_ih_nuebar_160_func(w21_i, theta21_i)},
                                  'HK':{1:P1e_ih_nuebar_129_6_func(w21_i, theta21_i), 
                                        2:P2e_ih_nuebar_129_6_func(w21_i, theta21_i)},
                                  'JN':{1:P1e_ih_nuebar_120_8_func(w21_i, theta21_i), 
                                        2:P2e_ih_nuebar_120_8_func(w21_i, theta21_i)},},
                            140:{'DUNE':{1:P1e_ih_nuebar_140_func(w21_i, theta21_i), 
                                         2:P2e_ih_nuebar_140_func(w21_i, theta21_i)},
                                  'HK':{1:P1e_ih_nuebar_95_func(w21_i, theta21_i), 
                                        2:P2e_ih_nuebar_95_func(w21_i, theta21_i)},
                                  'JN':{1:P1e_ih_nuebar_105_func(w21_i, theta21_i), 
                                        2:P2e_ih_nuebar_105_func(w21_i, theta21_i)},},
                            120:{'DUNE':{1:P1e_ih_nuebar_120_func(w21_i, theta21_i), 
                                         2:P2e_ih_nuebar_120_func(w21_i, theta21_i)},
                                  'HK':{1:P1e_ih_nuebar_97_func(w21_i, theta21_i), 
                                        2:P2e_ih_nuebar_97_func(w21_i, theta21_i)},
                                  'JN':{1:abs(U3[0,0]**2), 
                                        2:abs(U3[0,1]**2)},},
                            0:{'DUNE':{1:abs(U3[0,0]**2), 
                                      2:abs(U3[0,1]**2)},
                                'HK':{1:P1e_ih_nuebar_95_func(w21_i, theta21_i), 
                                      2:P2e_ih_nuebar_95_func(w21_i, theta21_i)},
                                'JN':{1:P1e_ih_nuebar_105_func(w21_i, theta21_i), 
                                      2:P2e_ih_nuebar_105_func(w21_i, theta21_i)},},
                            320:{'DUNE':{1:abs(U3[0,0]**2), 
                                         2:abs(U3[0,1]**2)},
                                'HK':{1:P1e_ih_nuebar_129_6_func(w21_i, theta21_i), 
                                      2:P2e_ih_nuebar_129_6_func(w21_i, theta21_i)},
                                'JN':{1:P1e_ih_nuebar_146_6_func(w21_i, theta21_i), 
                                      2:P2e_ih_nuebar_146_6_func(w21_i, theta21_i)},}}}}
    return Pie_dict[nu_nubar][mix][thetaz_dune][detector][state]


########################## MSC diagonal for Gamma = Gamma(E**n) ###############################

#D_kpc in kpc, E in MeV, Gamma in eV, thetaz in degrees
# @jit(cache=True)
def P_combined_msc_E(n, D_kpc, E, Gamma3, Gamma8, mix, thetaz, oscil_par, detector):
    t12, Deltam21 = oscil_par
    c12=np.cos(t12)
    s12=np.sin(theta12)
    c13=np.cos(theta13)
    s13=np.sin(theta13)
    c23=np.cos(theta23)
    s23=np.sin(theta23)

    U23 = np.array([ [1,0,0],[0,c23,s23],[0,-s23,c23] ])
    U13 = np.array([ [c13,0,s13],[0,1,0],[-s13,0,c13] ])
    U12 = np.array([ [c12,s12,0],[-s12,c12,0],[0,0,1] ])
    U3 = U23 @ U13 @ U12

    D_kpc = D_kpc - 1.92594212046e-8 #removing SN radius

    sn_matter = 'no'

    if sn_matter == 'yes':
        D_kpc = D_kpc - 1.92594212046e-8 #removing SN radius

    P11 = P11_E_func(n, E, D_kpc, Gamma3, Gamma8)
    P12 = P12_E_func(n, E, D_kpc, Gamma3, Gamma8)
    P31 = P31_E_func(n, E, D_kpc, Gamma8)
    P33 = P33_E_func(n, E, D_kpc, Gamma8)

    P21 = P12
    P22 = P11
    P23 = P31
    P13 = P31
    P32 = P31

    w21 = Deltam21/(E*1e6)

    if mix == 'NH':
        Pe3m = 1
        P1e = Pie_earth_interp_func(w21, theta12, 'nue', mix, thetaz, detector, 1)
        P2e = Pie_earth_interp_func(w21, theta12, 'nue', mix, thetaz, detector, 2)
        P3e = abs(U3[0,2])**2

        Pe1m_bar = 1
        P1e_bar = Pie_earth_interp_func(w21, theta12, 'nuebar', mix, thetaz, detector, 1)
        P2e_bar = Pie_earth_interp_func(w21, theta12, 'nuebar', mix, thetaz, detector, 2)
        P3e_bar = abs(U3[0,2])**2
        
        Pee = Pe3m*(P33*P3e + P32*P2e + P31*P1e)
        Pee_bar = Pe1m_bar*(P11*P1e_bar + P12*P2e_bar + P13*P3e_bar)
    
    elif mix == 'IH':
        Pe2m = 1
        P1e = Pie_earth_interp_func(w21, theta12, 'nue', mix, thetaz, detector, 1)
        P2e = Pie_earth_interp_func(w21, theta12, 'nue', mix, thetaz, detector, 2)
        P3e = abs(U3[0,2])**2

        Pe3m_bar = 1
        P1e_bar = Pie_earth_interp_func(w21, theta12, 'nuebar', mix, thetaz, detector, 1)
        P2e_bar = Pie_earth_interp_func(w21, theta12, 'nuebar', mix, thetaz, detector, 2)
        P3e_bar = abs(U3[0,2])**2
        
        Pee = Pe2m*(P23*P3e + P22*P2e + P21*P1e)
        Pee_bar = Pe3m_bar*(P31*P1e_bar + P32*P2e_bar + P33*P3e_bar)
    
    else:
        return
    
    return Pee, Pee_bar
###################################################################################################



########################## MSC CONSERVED ENERGY diagonal for Gamma = Gamma(E**n) ###############################

#D_kpc in kpc, E in MeV, Gamma in eV, thetaz in degrees
# @jit(cache=True)
def P_combined_msc_conserved_E(n, E, Gamma, mix, thetaz, oscil_par, detector):
    t12, Deltam21 = oscil_par
    c12=np.cos(t12)
    s12=np.sin(theta12)
    c13=np.cos(theta13)
    s13=np.sin(theta13)
    c23=np.cos(theta23)
    s23=np.sin(theta23)

    U23 = np.array([ [1,0,0],[0,c23,s23],[0,-s23,c23] ])
    U13 = np.array([ [c13,0,s13],[0,1,0],[-s13,0,c13] ])
    U12 = np.array([ [c12,s12,0],[-s12,c12,0],[0,0,1] ])
    U3 = U23 @ U13 @ U12

    nu_nubar = 'nue'
    # P11 = Pij_conserved_E_interps_func(n, E, Gamma, '11', mix, nu_nubar)
    # P12 = Pij_conserved_E_interps_func(n, E, Gamma, '12', mix, nu_nubar)
    # P13 = Pij_conserved_E_interps_func(n, E, Gamma, '13', mix, nu_nubar)
    P21 = Pij_conserved_E_interps_func(n, E, Gamma, '21', mix, nu_nubar)
    P22 = Pij_conserved_E_interps_func(n, E, Gamma, '22', mix, nu_nubar)
    P23 = Pij_conserved_E_interps_func(n, E, Gamma, '23', mix, nu_nubar)
    P31 = Pij_conserved_E_interps_func(n, E, Gamma, '31', mix, nu_nubar)
    P32 = Pij_conserved_E_interps_func(n, E, Gamma, '32', mix, nu_nubar)
    P33 = Pij_conserved_E_interps_func(n, E, Gamma, '33', mix, nu_nubar)

    nu_nubar = 'nuebar'
    P11_bar = Pij_conserved_E_interps_func(n, E, Gamma, '11', mix, nu_nubar)
    P12_bar = Pij_conserved_E_interps_func(n, E, Gamma, '12', mix, nu_nubar)
    P13_bar = Pij_conserved_E_interps_func(n, E, Gamma, '13', mix, nu_nubar)
    # P21_bar = Pij_conserved_E_interps_func(n, E, Gamma, '21', mix, nu_nubar)
    # P22_bar = Pij_conserved_E_interps_func(n, E, Gamma, '22', mix, nu_nubar)
    # P23_bar = Pij_conserved_E_interps_func(n, E, Gamma, '23', mix, nu_nubar)
    P31_bar = Pij_conserved_E_interps_func(n, E, Gamma, '31', mix, nu_nubar)
    P32_bar = Pij_conserved_E_interps_func(n, E, Gamma, '32', mix, nu_nubar)
    P33_bar = Pij_conserved_E_interps_func(n, E, Gamma, '33', mix, nu_nubar)

    w21 = Deltam21/(E*1e6)

    if mix == 'NH':
        Pe3m = 1
        P1e = Pie_earth_interp_func(w21, theta12, 'nue', mix, thetaz, detector, 1)
        P2e = Pie_earth_interp_func(w21, theta12, 'nue', mix, thetaz, detector, 2)
        P3e = abs(U3[0,2])**2

        Pe1m_bar = 1
        P1e_bar = Pie_earth_interp_func(w21, theta12, 'nuebar', mix, thetaz, detector, 1)
        P2e_bar = Pie_earth_interp_func(w21, theta12, 'nuebar', mix, thetaz, detector, 2)
        P3e_bar = abs(U3[0,2])**2
        
        Pee = Pe3m*(P33*P3e + P32*P2e + P31*P1e)
        Pee_bar = Pe1m_bar*(P11_bar*P1e_bar + P12_bar*P2e_bar + P13_bar*P3e_bar)
    
    elif mix == 'IH':
        Pe2m = 1
        P1e = Pie_earth_interp_func(w21, theta12, 'nue', mix, thetaz, detector, 1)
        P2e = Pie_earth_interp_func(w21, theta12, 'nue', mix, thetaz, detector, 2)
        P3e = abs(U3[0,2])**2

        Pe3m_bar = 1
        P1e_bar = Pie_earth_interp_func(w21, theta12, 'nuebar', mix, thetaz, detector, 1)
        P2e_bar = Pie_earth_interp_func(w21, theta12, 'nuebar', mix, thetaz, detector, 2)
        P3e_bar = abs(U3[0,2])**2
        
        Pee = Pe2m*(P23*P3e + P22*P2e + P21*P1e)
        Pee_bar = Pe3m_bar*(P31_bar*P1e_bar + P32_bar*P2e_bar + P33_bar*P3e_bar)
    
    else:
        return
    
    return Pee, Pee_bar
###################################################################################################



########################## MSC non-diagonal for Gamma = Gamma(E**n) ###############################

#D_kpc in kpc, E in MeV, Gamma in eV, thetaz in degrees
def P_combined_msc_non_diag_E(n, D_kpc, E, Gamma, mix, thetaz, oscil_par, detector):
    t12, Deltam21 = oscil_par
    c12=np.cos(t12)
    s12=np.sin(theta12)
    c13=np.cos(theta13)
    s13=np.sin(theta13)
    c23=np.cos(theta23)
    s23=np.sin(theta23)

    U23 = np.array([ [1,0,0],[0,c23,s23],[0,-s23,c23] ])
    U13 = np.array([ [c13,0,s13],[0,1,0],[-s13,0,c13] ])
    U12 = np.array([ [c12,s12,0],[-s12,c12,0],[0,0,1] ])
    U3 = U23 @ U13 @ U12

    D_kpc = D_kpc - 1.92594212046e-8 #removing SN radius

    sn_matter = 'no'

    if sn_matter == 'yes':
        D_kpc = D_kpc - 1.92594212046e-8 #removing SN radius

    P11 = P11_non_diag_E_func(n, E, D_kpc, Gamma)
    P33 = P11
    P13 = P13_non_diag_E_func(n, E, D_kpc, Gamma)
    P31 = P13
    P32 = 0
    P22 = 1
    P21 = 0
    P23 = 0
    P12 = 0

    w21 = Deltam21/(E*1e6)

    if mix == 'NH':
        Pe3m = 1
        P1e = Pie_earth_interp_func(w21, theta12, 'nue', mix, thetaz, detector, 1)
        P2e = Pie_earth_interp_func(w21, theta12, 'nue', mix, thetaz, detector, 2)
        P3e = abs(U3[0,2])**2

        Pe1m_bar = 1
        P1e_bar = Pie_earth_interp_func(w21, theta12, 'nuebar', mix, thetaz, detector, 1)
        P2e_bar = Pie_earth_interp_func(w21, theta12, 'nuebar', mix, thetaz, detector, 2)
        P3e_bar = abs(U3[0,2])**2
        
        Pee = Pe3m*(P33*P3e + P32*P2e + P31*P1e)
        Pee_bar = Pe1m_bar*(P11*P1e_bar + P12*P2e_bar + P13*P3e_bar)
    
    elif mix == 'IH':
        Pe2m = 1
        P1e = Pie_earth_interp_func(w21, theta12, 'nue', mix, thetaz, detector, 1)
        P2e = Pie_earth_interp_func(w21, theta12, 'nue', mix, thetaz, detector, 2)
        P3e = abs(U3[0,2])**2

        Pe3m_bar = 1
        P1e_bar = Pie_earth_interp_func(w21, theta12, 'nuebar', mix, thetaz, detector, 1)
        P2e_bar = Pie_earth_interp_func(w21, theta12, 'nuebar', mix, thetaz, detector, 2)
        P3e_bar = abs(U3[0,2])**2
        
        Pee = Pe2m*(P23*P3e + P22*P2e + P21*P1e)
        Pee_bar = Pe3m_bar*(P31*P1e_bar + P32*P2e_bar + P33*P3e_bar)
    
    else:
        return
    
    return Pee, Pee_bar
###################################################################################################


#################################### nu-loss for Gamma = Gamma(E**n) ##############################

#D_kpc in kpc, E in MeV, Gamma in eV, thetaz in degrees
def P_combined_loss_E(n, D_kpc, E, Gamma, mix, thetaz, oscil_par, detector):
    t12, Deltam21 = oscil_par
    c12=np.cos(t12)
    s12=np.sin(theta12)
    c13=np.cos(theta13)
    s13=np.sin(theta13)
    c23=np.cos(theta23)
    s23=np.sin(theta23)

    U23 = np.array([ [1,0,0],[0,c23,s23],[0,-s23,c23] ])
    U13 = np.array([ [c13,0,s13],[0,1,0],[-s13,0,c13] ])
    U12 = np.array([ [c12,s12,0],[-s12,c12,0],[0,0,1] ])
    U3 = U23 @ U13 @ U12

    D_kpc = D_kpc - 1.92594212046e-8 #removing SN radius

    Pii = P11_loss_E_func(n, E, D_kpc, Gamma)

    w21 = Deltam21/(E*1e6)

    if mix == 'NH':

        Pee = Pii*U3[0,2]**2
        Pee_bar = Pii*Pie_earth_interp_func(w21, theta12, 'nuebar', mix, thetaz, detector, 1)
        Pme = Pii*Pie_earth_interp_func(w21, theta12, 'nue', mix, thetaz, detector, 1)
        Pte = Pii*Pie_earth_interp_func(w21, theta12, 'nue', mix, thetaz, detector, 2)
        Pme_bar = Pii*Pie_earth_interp_func(w21, theta12, 'nuebar', mix, thetaz, detector, 2)
        Pte_bar = Pii*U3[0,2]**2
        Pmm = Pii*U3[1,0]**2
        Pmt = Pii*U3[2,0]**2
        Ptt = Pii*U3[2,1]**2
        Ptm = Pii*U3[1,1]**2
        Pem_bar = Pii*U3[1,0]**2
        Pet_bar = Pii*U3[2,0]**2
        Pmm_bar = Pii*U3[1,1]**2
        Pmt_bar = Pii*U3[2,1]**2
        Ptt_bar = Pii*U3[2,2]**2
        Ptm_bar = Pii*U3[1,2]**2
    
    elif mix == 'IH':
        Pee = Pii*Pie_earth_interp_func(w21, theta12, 'nue', mix, thetaz, detector, 2)
        Pee_bar = Pii*U3[0,2]**2
        Pme = Pii*Pie_earth_interp_func(w21, theta12, 'nue', mix, thetaz, detector, 1)
        Pte = Pii*U3[0,2]**2
        Pme_bar = Pii*Pie_earth_interp_func(w21, theta12, 'nuebar', mix, thetaz, detector, 2)
        Pte_bar = Pii*Pie_earth_interp_func(w21, theta12, 'nuebar', mix, thetaz, detector, 1)
        Pmm = Pii*U3[1,0]**2
        Pmt = Pii*U3[2,0]**2
        Ptt = Pii*U3[2,2]**2
        Ptm = Pii*U3[1,2]**2
        Pem_bar = Pii*U3[1,2]**2
        Pet_bar = Pii*U3[2,2]**2
        Pmm_bar = Pii*U3[1,1]**2
        Pmt_bar = Pii*U3[2,1]**2
        Ptt_bar = Pii*U3[2,0]**2
        Ptm_bar = Pii*U3[1,0]**2
    
    else:
        return
    
    return Pee, Pee_bar, Pme, Pte, Pme_bar, Pte_bar, Pmm, Pmt, Ptt, Ptm, Pem_bar, Pet_bar, Pmm_bar, Pmt_bar, Ptt_bar, Ptm_bar

##################################################################################################