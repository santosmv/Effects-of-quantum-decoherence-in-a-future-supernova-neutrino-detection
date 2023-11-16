import numpy as np
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

# connection = sqlite3.connect('data/database_neutrino7.db')
# cursor = connection.cursor()

def read_cross_data():
    cursor.execute("select * from elastic_cross")
    d = cursor.fetchall()
    Enu_data = np.array(d)[:,0]
    Te_data = np.array(d)[:,1]
    sigma_e_data = np.array(d)[:,2]
    sigma_mu_data = np.array(d)[:,3]
    sigma_antie_data = np.array(d)[:,4]
    sigma_antimu_data = np.array(d)[:,5]
    return Enu_data, Te_data, sigma_e_data, sigma_mu_data, sigma_antie_data, sigma_antimu_data


def read_bins(detector, mix):
    cursor.execute("select bins from bins where detector = ? and hierarchy = ?",(detector, mix,))
    bins = cursor.fetchall()
    bins = np.array(bins)[:,0]
    return bins


def read_work_function(bin_i, detector, channel, mix, trigger_l):
    if trigger_l == 'loss':
        query = "select * from work_function_loss where detector = ? and hierarchy = ? and channel = ? and bin_i = ?"
    else:
        query = "select * from work_function where detector = ? and hierarchy = ? and channel = ? and bin_i = ?"

    if detector == 'hk':
        if channel == 'elastic':
            cursor.execute(query + " and flavor = 'nue'",(detector, mix, channel, bin_i,))
            data = cursor.fetchall()
            # Enuf = np.array(data)[:,-2]
            work_nue = np.array(data)[:,-1]
            work_nue = work_nue.astype(np.float)
            
            cursor.execute(query + " and flavor = 'numu'",(detector, mix, channel, bin_i,))
            data = cursor.fetchall()
            work_numu = np.array(data)[:,-1]
            work_numu = work_numu.astype(np.float)

            cursor.execute(query + " and flavor = 'antinue'",(detector, mix, channel, bin_i,))
            data = cursor.fetchall()
            work_antinue = np.array(data)[:,-1]
            work_antinue = work_antinue.astype(np.float)

            cursor.execute(query + " and flavor = 'antinumu'",(detector, mix, channel, bin_i,))
            data = cursor.fetchall()
            work_antinumu = np.array(data)[:,-1]
            work_antinumu = work_antinumu.astype(np.float)

            Enu_hk_es = np.array(data)[:,6]
            Enu_hk_es = Enu_hk_es.astype(np.float)
            return Enu_hk_es, work_nue, work_numu, work_antinue, work_antinumu

        elif channel == 'ibd':
            cursor.execute(query, (detector, mix, channel, bin_i,))
            data = cursor.fetchall()
            work_ibd = np.array(data)[:,-1]
            work_ibd = work_ibd.astype(np.float)
            Enu_hk_ibd = np.array(data)[:,6]
            Enu_hk_ibd = Enu_hk_ibd.astype(np.float)
            return Enu_hk_ibd, work_ibd
    
    elif detector == 'dune':
        cursor.execute(query, (detector, mix, channel, bin_i,))
        data = cursor.fetchall()
        work_nueAr = np.array(data)[:,-1]
        work_nueAr = work_nueAr.astype(np.float)
        Enu_dune = np.array(data)[:,6]
        Enu_dune = Enu_dune.astype(np.float)
        return Enu_dune, work_nueAr
    
    elif detector == 'jn':
        cursor.execute(query, (detector, mix, channel, bin_i,))
        data = cursor.fetchall()
        work_ibd = np.array(data)[:,-1]
        work_ibd = work_ibd.astype(np.float)
        Enu_juno = np.array(data)[:,6]
        Enu_juno = Enu_juno.astype(np.float)
        return Enu_juno, work_ibd
