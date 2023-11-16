import numpy as np
from scipy.interpolate import interp1d
from math import sqrt, pi
from scipy.integrate import simpson
import sqlite3
from bins import create_bins
import socket
import platform
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

# data_path = 'data/database_neutrino7.db'
# connection = sqlite3.connect(data_path)
# cursor = connection.cursor()

#resolution of SK detector
def resolution_hk(Te_data, Enu):
    sigma = -0.0839 + 0.349 * np.sqrt(Te_data + 0.511) + 0.0397*(Te_data + 0.511)
    R = 1/(sqrt(2*pi)*sigma) * np.exp(-1/2*((Enu - Te_data)/sigma)**2)
    return R

def resolution_dune(Ee, Enu):
    sigma = 0.11*np.sqrt(Ee) + 0.02*Ee
    R = 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-1/2*(Enu - Ee)**2/sigma**2)
    return R

#resolution of JUNO detector
def resolution_jn(Te_data, Enu):
    sigma = 0.03 * np.sqrt(Te_data)
    R = 1/(sqrt(2*pi)*sigma) * np.exp(-1/2*((Enu - Te_data)/sigma)**2)
    return R

def elastic_data():
    cursor.execute("select bins from bins where detector = 'hk'")
    data = np.array(cursor.fetchall())[:,0]
    Enu_max = max(data)
    print(Enu_max)
    cursor.execute('delete from elastic_cross')
    me = 0.511 #MeV
    from interaction import dsigma_dEr_nu_e_ES
    Enu_list = np.arange(0.01, Enu_max + 0.1, 0.1)
    for Enu in Enu_list:
        Te_max = Enu**2/(Enu + me/2)
        Te_list = np.linspace(0, Te_max, 101)
        for Te in Te_list:
            dnue = dsigma_dEr_nu_e_ES(Enu, Te, 'nue')
            dnumu = dsigma_dEr_nu_e_ES(Enu, Te, 'numu')
            dantinue = dsigma_dEr_nu_e_ES(Enu, Te, 'nuebar')
            dantinumu = dsigma_dEr_nu_e_ES(Enu, Te, 'numubar')
            cursor.execute("insert into elastic_cross values (?,?,?,?,?,?)", (Enu, Te, dnue, dnumu, dantinue, dantinumu))
    

#this function creates a function to be multiplied to events rate
def create_work_function(detector, channel, flavor, trigger_l):
    #uncomment to create bins with 2 sigma_E
    bins_list_hk_nh, bins_list_dune_nh, bins_list_jn_nh = create_bins(trigger_l,'NH')
    bins_list_hk_ih, bins_list_dune_ih, bins_list_jn_ih = create_bins(trigger_l,'IH')

    if trigger_l == 'loss':
        table = 'work_function_loss'
    else:
        table = 'work_function'

    #work function for elastic scattering for HK
    def work_function_elastic(bin_i, Eei, Eef, mix):
        if flavor == 'nue':
            l = 2
        elif flavor == 'numu':
            l = 3
        elif flavor == 'antinue':
            l = 4
        elif flavor == 'antinumu':
            l = 5
        Eth = 3
        n = 250
        Ee_list = np.linspace(Eei, Eef, n) - 0.511
        integ_list_final = []
        sig_Enu = []
        cursor.execute("select round(Enu,2) from elastic_cross")
        Enu_data = cursor.fetchall()
        Enu_data = list(sorted(set([E[0] for E in Enu_data])))

        for Enu in Enu_data:
            cursor.execute("select * from elastic_cross where round(Enu,2) = %f"%(Enu))
            data = np.array(cursor.fetchall())
            Te_data, sigma_data = data[:,1], data[:,l]
            integ_list = []
            sigma_func = interp1d(Te_data, sigma_data, kind='cubic', fill_value='extrapolate')
            Te_list = np.linspace(0, max(Te_data), n)
            for Ee_value in Ee_list:
                f = sigma_func(Te_list) * resolution_hk(Te_list, Ee_value)
                integ1 = simpson(f, Te_list)
                integ_list.append(integ1)
            integ2 = simpson(integ_list, Ee_list)
            integ_list_final.append(integ2)

            sig_Enu.append(simpson(sigma_func(Te_data), Te_data))
            cursor.execute("insert into %s values (?,?,?,?,?,?,?,?,?,?)"%table, (bin_i, detector, mix, channel, flavor, Eth, Enu, Eei, Eef, integ2))
            connection.commit()
          
    #work function for IBD for HK
    def work_function_ibd(bin_i, Eei, Eef, mix):
        Eth = 3
        Enu_list = np.linspace(Eth, 61, 601)
        Te_list = np.linspace(Eei, Eef, 601)

        for Enu in Enu_list:
            res = simpson(resolution_hk(Te_list, Enu), Te_list)
            cursor.execute("insert into %s values (?,?,?,?,?,?,?,?,?,?)"%table, (bin_i, detector, mix, channel, flavor, Eth, Enu, Eei, Eef, res))
    
    #work function for DUNE
    def work_function_nueAr(bin_i, Eei, Eef, mix):
        Eth = 4.5
        Enu_list = np.linspace(Eth, 61, 601)
        Ee_list = np.linspace(Eei, Eef, 601)

        for Enu in Enu_list:
            res = simpson(resolution_dune(Ee_list, Enu), Ee_list)
            cursor.execute("insert into %s values (?,?,?,?,?,?,?,?,?,?)"%table, (bin_i, detector, mix, channel, flavor, Eth, Enu, Eei, Eef, res))
    
    #work function for IBD for JUNO
    def work_function_jn_ibd(bin_i, Eei, Eef, mix):
        Eth = 3
        Enu_list = np.linspace(Eth, 61, 601)
        Te_list = np.linspace(Eei, Eef, 601)

        for Enu in Enu_list:
            res = simpson(resolution_jn(Te_list, Enu), Te_list)
            cursor.execute("insert into %s values (?,?,?,?,?,?,?,?,?,?)"%table, (bin_i, detector, mix, channel, flavor, Eth, Enu, Eei, Eef, res))
    
    def run_elastic_hk(mix):
        if mix == 'NH':
            bins_list_hk = bins_list_hk_nh
        elif mix == 'IH':
            bins_list_hk = bins_list_hk_ih

        max_bins_hk = max(bins_list_hk)
        # create work function for hk
        for i in range(len(bins_list_hk)-1):
            Eei = bins_list_hk[i]
            Eef = bins_list_hk[i+1]
            if Eei > max_bins_hk and Eef > max_bins_hk:
                break
            bin_i = i+1
            print(flavor, bin_i)
            work_function_elastic(bin_i, Eei, Eef, mix)
    
    def run_ibd_hk(mix):
        if mix == 'NH':
            bins_list_hk = bins_list_hk_nh
        elif mix == 'IH':
            bins_list_hk = bins_list_hk_ih
        
        max_bins_hk = max(bins_list_hk)

        # create work function for hk
        for i in range(len(bins_list_hk)-1):
            Eei = bins_list_hk[i]
            Eef = bins_list_hk[i+1]
            if Eei > max_bins_hk and Eef > max_bins_hk:
                break
            bin_i = i+1
            work_function_ibd(bin_i, Eei, Eef, mix)
    
    def run_dune(mix):
        if mix == 'NH':
            bins_list_dune = bins_list_dune_nh
        elif mix == 'IH':
            bins_list_dune = bins_list_dune_ih

        max_bins = max(bins_list_dune)+0.001
        # create work function for dune
        for i in range(len(bins_list_dune)-1):
            Eei = bins_list_dune[i]
            Eef = bins_list_dune[i+1]
            # if Eei > 50 and Eef > 50:
            if Eei > max_bins and Eef > max_bins:
                break
            # print(Eei, Eef)
            bin_i = i+1
            work_function_nueAr(bin_i, Eei, Eef, mix)
    
    def run_ibd_jn(mix):
        if mix == 'NH':
            bins_list_jn = bins_list_jn_nh
        elif mix == 'IH':
            bins_list_jn = bins_list_jn_ih
        
        max_bins_jn = max(bins_list_jn)

        # create work function for jn
        for i in range(len(bins_list_jn)-1):
            Eei = bins_list_jn[i]
            Eef = bins_list_jn[i+1]
            if Eei > max_bins_jn and Eef > max_bins_jn:
                break
            bin_i = i+1
            work_function_jn_ibd(bin_i, Eei, Eef, mix)
    
    if detector == 'hk':
        if channel == 'elastic':
            cursor.execute("delete from %s where channel = 'elastic' and flavor = '%s'"%(table, flavor))
            run_elastic_hk('NH')
            run_elastic_hk('IH')
        elif channel == 'ibd':
            cursor.execute("delete from %s where channel = 'ibd' and detector = 'hk'"%table)
            run_ibd_hk('NH')
            run_ibd_hk('IH')
    elif detector == 'dune':
        cursor.execute("delete from %s where channel = 'nue-Ar'"%table)
        run_dune('NH')
        run_dune('IH')
    elif detector == 'jn':
        cursor.execute("delete from %s where channel = 'ibd' and detector = 'jn'"%table)
        run_ibd_jn('NH')
        run_ibd_jn('IH')
    
    print('Finished:', detector, channel, flavor)

trigger_l = 'loss'
# trigger_l = 0

# elastic_data()
create_work_function('hk','elastic','nue', trigger_l)
create_work_function('hk','elastic','numu', trigger_l)
create_work_function('hk','elastic','antinue', trigger_l)
create_work_function('hk','elastic','antinumu', trigger_l)
create_work_function('hk','ibd','antinue', trigger_l)
# create_work_function('dune','nue-Ar','nue', trigger_l)
# create_work_function('jn','ibd','antinue', trigger_l)

connection.commit()
connection.close()