import numpy as np
import math
from scipy.interpolate import interp1d

#CC cross section
#cm^2
def cross_CC(Enu):
    sigmaCC = 9e-44 * Enu**2 #cm^2
    return sigmaCC


#ES cross section
#cm^2
def cross_ES(Enu):
    sigmaES = 9e-45 * Enu #cm^2
    return sigmaES


#IBD differential cross section following (ZELLER) eq. (2) from https://arxiv.org/pdf/1305.7513.pdf
def dsigma_dcos_zeller(Ee, cos_theta):
    #Fermi constant
    Gf = 1.16637e-23 * 10**12 #MeV-²
    #11 element of CKM matrix
    Vud = 0.974
    #calculating positron momentum
    me = 0.510999 #MeV
    ve = np.sqrt(1 - me**2/Ee**2)
    gamma = 1/np.sqrt(1 - ve**2)
    pe = gamma * me * ve
    beta = pe/me
    #form factors
    fv = 1
    fa = -1/2 #fa = -ga (confirm later)
    #conversion factor from MeV-² to cm²
    conv = 1000**2 * (0.197e-15)**2 * 100**2
    dcross = Gf**2 * Vud**2 * Ee * pe/(2*math.pi) * (fv**2 * (1 + beta * cos_theta) + 3 * fa**2 * (1 - beta/3 * cos_theta)) * conv
    return dcross


#IBD differential cross section following (VOGEL and BEACON) (https://arxiv.org/pdf/hep-ph/9903554.pdf)
def dsigma_dcos_IBD(Enu, cos_theta):
    #Fermi constant
    Gf = 1.1663787e-11 #MeV-²
    #vector and axial-vector coupling constants
    f = 1
    g = 1.26
    mn = 939.56542052 #MeV  #PDG (2019)
    mp = 938.27208816 #MeV  #PDG (2019)
    me = 0.51099895000 #MeV  #PDG (2019)
    Delta = mn - mp
    Ee0 = Enu - Delta
    pe0 = np.sqrt(Ee0**2 - me**2)
    ve0 = pe0/Ee0
    M = (mp+mn)/2
    y_squared = (Delta**2 - me**2)/2
    Ee1 = Ee0 * (1 - Enu/M * (1 - ve0 * cos_theta)) - y_squared/M
    pe1 = np.sqrt(Ee1**2 - me**2)
    ve1 = pe1/Ee1
    cos_theta_C = 0.974
    Delta_R_inner = 0.024
    sigma0 = Gf**2 * cos_theta_C**2/math.pi * (1 + Delta_R_inner)
    f2 = 3.706
    Gamma = 2*(f + f2)*g * ((2*Ee0 + Delta) * (1 - ve0*cos_theta) - me**2/Ee0) + \
            (f**2 + g**2) * (Delta*(1 + ve0*cos_theta) + me**2/Ee0) + \
            (f**2 + 3*g**2) * ((Ee0 + Delta) * (1 - cos_theta/ve0) - Delta) + \
            (f**2 - g**2) * ((Ee0 + Delta) * (1 - cos_theta/ve0) - Delta) * ve0 * cos_theta
    
    hc_square = (0.197e-10)**2 #MeV^(-2) to cm^2
    dsigma_dcos = sigma0/2 * ((f**2 + 3*g**2) + (f**2 - g**2) * ve1 * cos_theta) * Ee1 * pe1 - sigma0/2 * Gamma/M * Ee0 * pe0
    dsigma_dcos = hc_square * dsigma_dcos #cm^2
    return dsigma_dcos


#neutrino-argon cross section following http://repositorio.unicamp.br/jspui/bitstream/REPOSIP/342830/1/RoseroGil_JhonAndersson_D.pdf
def sigma_nue_Ar_approximation(Enu):
    Q = 5.885 #MeV
    F = 1.56 #for x > 0.5MeV
    me = 0.510999 #MeV
    x = Enu - Q + me
    sigma = 1.702e-44 * x/me**2 * np.sqrt(x**2 - me**2) * F
    return sigma


#taken from snowglobes (cm^2)
def sigma_nue_Ar(Enu):
    data_file = np.loadtxt('data/xs_nue_Ar40.dat', delimiter=None)
    Energy = 1000 * 10**(data_file[2:,0]) #MeV
    cross = data_file[2:,1] / 1000 * 1e-38 # cm^2/MeV 
    cross_func = interp1d(Energy, cross, kind='cubic', bounds_error=False, fill_value='extrapolate')
    return cross_func(Enu) * Enu


#neutrino electron ES differential cross section following https://arxiv.org/abs/1912.06658
def dsigma_dEr_nu_e_ES(Enu, Ee, flavor):
    #Fermi constant
    Gf = 1.16637e-23 * 10**12 #MeV-²
    me = 0.51099895000 #MeV  #PDG (2019)
    sw_square = 0.23153 #https://arxiv.org/pdf/hep-ex/0509008.pdf

    if flavor == 'nue':
        g1 = 1/2 + sw_square
        g2 = sw_square
    elif flavor == 'numu' or flavor == 'nutau':
        g1 = -1/2 + sw_square
        g2 = sw_square
    elif flavor == 'nuebar':
        g1 = sw_square
        g2 = 1/2 + sw_square
    elif flavor == 'numubar' or flavor == 'nutaubar':
        g1 = sw_square
        g2 = -1/2 + sw_square
    
    # theta = 0
    # Ee = 2*me*Enu**2*np.cos(theta)**2/((me + Enu)**2 - Enu**2*np.cos(theta)**2)
    sigma0 = 2*Gf**2 * me**2/math.pi
    dsigma = sigma0/me * (g1**2 + g2**2*(1-Ee/Enu)**2 - g1*g2*me*Ee/Enu**2)
    
    # dsigma = 2 * Gf**2 * me/math.pi * (g1**2 + g2**2 * (1 - Ee/Enu)**2 - g1*g2*me*Ee/Enu**2)
    # dsigma = 1.72e-41 * (g1**2 + g2**2 * (1 - Ee/Enu)**2) / 1000
    MeV2_to_cm2 = (0.197e3 * 10**-15 * 100)**2
    dsigma = dsigma * MeV2_to_cm2
    return dsigma



#neutrino-proton ES differential cross section https://arxiv.org/pdf/1103.2768.pdf
def dsigma_dT_nu_proton_ES(Enu, Tk):
    #Tkp is the proton kinect energy and Enu is the neutrino energy
    #Tk goes from 0 to Tmax where neutrino momentum is reversed
    #Enu goes from sqrt(mp.Tk/2) until Tkmax
    dsigma = 4.83e-42 * (1 + 466*Tk/Enu**2) #cm^2/MeV
    return dsigma

#from Kemp analysis: NC nu-C12 cross section, v + 12C -> v'+12C* ; 12C* -> 12C + 15.11 MeV gamma
#valid from 16 to 100 MeV
def sigma_nu_C12_NC(Enu, nu_antinu):
    if nu_antinu == 'nu':
        a0 = 6.6050148625750e+00 #+/- 4,5365873953055e-01
        a1 = -8.1785983332850e-01 #+/- 4,2821081701704e-02
        a2 = 2.9704564808250e-02 #+/- 1,3480099720515e-03
        a3 = -2.4244820490160e-04 #+/- 1,7037975586186e-05
        a4 = 5.6329632876460e-07 #+/- 7,4355254574571e-08
        sigma = (a0 + a1*Enu + a2*Enu**2 + a3*Enu**3 + a4*Enu**4) * 1e-42 #cm^2/MeV
        return sigma
    if nu_antinu == 'antinu':
        a0 = 4.9202270053880e+00 #+/- 1,8889300386557e-01
        a1 = -6.6178961849680e-01 #+/- 1,7829708119692e-02
        a2 = 2.6320145416360e-02 #+/- 5,6128019176603e-04
        a3 = -2.6062022365150e-04 #+/- 7,0942193368431e-06
        a4 = 8.2263558566680e-07 #+/- 3,0959810647744e-08
        sigma = (a0 + a1*Enu + a2*Enu**2 + a3*Enu**3 + a4*Enu**4) * 1e-42 #cm^2/MeV
        return sigma

#Get an effective IDB cross section
def create_sigma_IBD():
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
        elif 'compute' in pc_name or 'masternode' in pc_name:
            data_path = 'data/database_compute.db'
        connection = sqlite3.connect(data_path)
    cursor = connection.cursor()
    cursor.execute('drop table ibd_cross')

    def create_database_ibd():
        cursor.execute("""create table ibd_cross(
                                            Enu real,
                                            cross real
        )""")
    create_database_ibd()
    from scipy.integrate import simpson
    cos_theta_list = np.linspace(-1,1,101)
    Enu_list = np.linspace(0.1,80,101)
    d_list = []
    for Enu in Enu_list:
        d = simpson(dsigma_dcos_IBD(Enu, cos_theta_list), cos_theta_list)
        cursor.execute("insert into ibd_cross values (?,?)",(Enu, d))
    connection.commit()
    connection.close()

def sigma_IBD(Enu):
    from scipy.interpolate import interp1d
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
        elif 'compute' in pc_name or 'masternode' in pc_name:
            data_path = 'data/database_compute.db'
        connection = sqlite3.connect(data_path)
    cursor = connection.cursor()
    cursor.execute("select * from ibd_cross")
    data = cursor.fetchall()
    data = np.array(data)
    Enu_data = data[:,0]
    cross_data = data[:,1]
    cross_func = interp1d(Enu_data, cross_data, kind='linear', fill_value='extrapolate')
    return cross_func(Enu)
