import numpy as np
import math
from scipy.linalg import expm
from numpy import array, sqrt, exp, trace, mean, pi, tan
import numpy as np
import math
import numpy.linalg as la
from numba import jit

Gf = 1.16632e-23 #eV^-2
data_ea = np.loadtxt('data/Earth_profile.txt', delimiter=',', skiprows=8)

# @jit(nopython=True, cache=True, fastmath=True)
#Energy in MeV, Gamma in eV
def Pje_Earth_f(w21, thetaz, mix, j, nu_nubar, theta12):
    # thetaz = thetaz*np.pi/180
    
    Deltam21 = 7.42e-5 #eV² +0.21 -0.20
    Deltam31 = 2.510e-3 #NH #eV² +-0.027 https://arxiv.org/pdf/2111.03086.pdf
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
    U23 = np.array([ [1.,0.,0.],[0.,c23,s23],[0,-s23,c23] ], dtype=np.complex128)
    U13 = np.array([ [c13,0.,s13],[0.,1.,0.],[-s13,0,c13] ], dtype=np.complex128)
    U12 = np.array([ [c12,s12,0.],[-s12,c12,0],[0.,0.,1.] ], dtype=np.complex128)
    U3 = U23 @ U13 @ U12
    U3_dag = np.transpose(U3)

    #Earth profile----------
    # data_ea = np.loadtxt('data/Earth_profile.txt', delimiter=',', skiprows=8)
    r_data_earth = data_ea[:,0] #km
    cm3_to_eV3 = (0.197e9*1e-15*100)**3
    Na = 6.022e23
    ne_earth = data_ea[:,1] * cm3_to_eV3 * Na #eV^3

    n = 1001
    # E = E*1e6 #eV
    Pee_list = []
    #imaginary number
    i = 1j
    k = 0

    #r in km, ne in eV^3
    def ne_func_earth(r):
        r_data_earth_rev = -np.flip(r_data_earth)
        ne_earth_rev = np.flip(ne_earth)
        r_data_earth_complete = np.concatenate((r_data_earth_rev, r_data_earth))
        ne_earth_complete = np.concatenate((ne_earth_rev, ne_earth))
        # ne_f = interp1d(r_data_earth_complete, ne_earth_complete, kind='linear', fill_value='extrapolate')
        ne_f = np.interp(r, r_data_earth_complete, ne_earth_complete)
        return ne_f #ne_array

    def l_earth(thetaz):
        alpha = 90 - thetaz
        alpha = alpha*pi/180
        x1 = 0
        y1 = max(r_data_earth)
        if thetaz >= 0 and thetaz <=90:
            x2 = 0
            y2 = max(r_data_earth)
        elif thetaz > 90 and thetaz <= 180:
            x2 = -2*max(r_data_earth)*tan(alpha)/(1+tan(alpha)**2)
            y2 = tan(alpha)*x2 + max(r_data_earth)
        else:
            print('Error: the function l_earth supports only angles between 90 and 180 degrees')
            return 0

        d = sqrt((x2-x1)**2 + (y2-y1)**2)

        def line_x_to_r_profile():
            xa = np.linspace(0,x2,n)
            a = (y2-y1)/(x2-x1)
            b = y1
            ya = a*xa + b
            rnew = sqrt(xa**2 + ya**2)
            return rnew

        r_new = line_x_to_r_profile()
        # return d,x1,x2,y1,y2,r_new
        return r_new

    if thetaz <= 90:
        if j == '1':
            return abs(U3[0,0])**2
        elif j == '2':
            return abs(U3[0,1])**2
        elif j == '3':
            return abs(U3[0,2])**2

    else:
        r_list = l_earth(thetaz)

    #Energy in MeV and rho in g cm^-3
    def diag(w21, Vcc0):
        # try:
        #Re-evaluating the M2 with the E parameter for the function
        if mix == 'NH':
            M2 = np.array([[0.,0.,0.],[0.,w21,0.],[0.,0.,Deltam31/Deltam21*w21]], dtype=np.complex128)
        elif mix == 'IH':
            M2 = np.array([[0.,0.,0.],[0.,w21,0.],[0.,0.,-Deltam31/Deltam21*w21]], dtype=np.complex128)
        # else:
        #     return 0,0,0,np.zeros((3,3)),np.zeros((3,3))

        if nu_nubar == 'nuebar':
            Vcc0 = -Vcc0
        # else:
        #     return 0,0,0,np.zeros((3,3)),np.zeros((3,3))

        #Hamiltonian in flavour basis
        # M2f = U3 @ M2 @ U3_dag
        M2f = np.dot(np.dot(U3, M2), U3_dag)
        
        #Potential in matter
        Vcc = np.array([[Vcc0,0.,0.],[0.,0.,0.],[0.,0.,0.]], dtype=np.complex128)

        Hf = 1/2 * M2f + Vcc

        #eigenvalues
        eigvals, eigvecs = la.eig(Hf)
        eigvals = eigvals.real

        #sorting eigenvalues list
        id_sor = np.argsort(np.abs(eigvals))

        #adding eigenvalues to a list
        eval1 = eigvals[id_sor[0]]
        eval2 = eigvals[id_sor[1]]
        eval3 = eigvals[id_sor[2]]

        #collecting eigenvectors from sorted eigenvalues
        eve1 = eigvecs[:,id_sor[0]]
        eve2 = eigvecs[:,id_sor[1]]
        eve3 = eigvecs[:,id_sor[2]]

        #Eigenvector for electron neutrino spectrum
        Ue1 = (eve1[0])
        Ue2 = (eve2[0])
        Ue3 = (eve3[0])
        #Eigenvector for muon neutrino spectrum
        Umu1 = (eve1[1])
        Umu2 = (eve2[1])
        Umu3 = (eve3[1])
        #Eigenvector for tau neutrino spectrum
        Utau1 = (eve1[2])
        Utau2 = (eve2[2])
        Utau3 = (eve3[2])

        Um3 = array([[Ue1,Ue2,Ue3],
                    [Umu1,Umu2,Umu3],
                    [Utau1,Utau2,Utau3]], dtype=np.complex128)
        Um3_dag = np.transpose(np.conjugate(Um3))

        return eval1, eval2, eval3, Um3, Um3_dag
        # except:
        #     return 0,0,0,np.zeros((3,3)),np.zeros((3,3))

    ############### for thetaz=180º ##############
    #initial radius of the Earth
    # r0 = -max(r_data_earth) #km
    # #final radius to calculate Pee
    # rf = max(r_data_earth) #km
    # r_list = np.linspace(r0, rf, n) #km
    ##############################################

    Pee_list = []
    k = 0

    #loop over all points in the SN potential
    for r in r_list[1:]:
        vcc0 = math.sqrt(2) * Gf * ne_func_earth(r)
        id = list(r_list).index(r)
        Delta_x_km = r - r_list[id-1]
        Delta_x_eV = Delta_x_km/(0.197e9 * 1e-15 / 1000) #eV^-1

        eval1, eval2, eval3, Um, Um_dag = diag(w21, vcc0)
        exp_Hm = np.array([[exp(-i*eval1*Delta_x_eV), 0, 0],
                            [0, exp(-i*eval2*Delta_x_eV), 0],
                            [0, 0, exp(-i*eval3*Delta_x_eV)]])
        if k == 0:
            U = np.dot(np.dot(Um, exp_Hm), Um_dag)
            nue_0 = array([1,0,0], dtype=np.complex128)
            numu_0 = array([0,1,0], dtype=np.complex128)
            nutau_0 = array([0,0,1], dtype=np.complex128)
            if j == '1':
                nu1_0 = Um_dag[0,0] * nue_0 + Um_dag[0,1] * numu_0 + Um_dag[0,2] * nutau_0
                nui = U @ nu1_0
            elif j == '2':
                nu2_0 = Um_dag[1,0] * nue_0 + Um_dag[1,1] * numu_0 + Um_dag[1,2] * nutau_0
                nui = U @ nu2_0
            elif j == '3':
                nu3_0 = Um_dag[2,0] * nue_0 + Um_dag[2,1] * numu_0 + Um_dag[2,2] * nutau_0
                nui = U @ nu3_0
            
        else:
            U = Um @ exp_Hm @ Um_dag
            nui = U @ nui

        nue_0 = array([1,0,0], dtype=np.complex128)
        Pie = abs(nui @ nue_0)**2
        Pee_list.append(Pie)
        k = k+1
    return Pie
Pje_Earth = np.vectorize(Pje_Earth_f)

##############################################################################################

# Pje_Earth(.1, .1, 'IH', 2, 'nue', .1)