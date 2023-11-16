import numpy as np
from iminuit import Minuit
import math
from sys import exit
from importlib import reload
from re import sub
from datetime import datetime
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
        data_path = '/data/database_compute.db'

    connection = sqlite3.connect(data_path)
cursor = connection.cursor()

# connection = sqlite3.connect('data/database_neutrino7.db')
# cursor = connection.cursor()

def minimize_chi_square_pi(config_list):
    from chi_square_pi import chi_square
    from time import time

    # 80 x 80 = 6400 -> 43.41 hours
    type_stat, detector, model = config_list[3], config_list[2], config_list[0]
    D = config_list[1] #kpc
    ai, af = -4*0.4, 4*0.4
    t12i, t12f = (33.45-0.75)*math.pi/180, (33.45+0.77)*math.pi/180

    if type_stat == 'contour_pull':
        min_chi = Minuit(chi_square, pid=1, pih=0.4, pjj=1, pjh=0.5, theta12=33.45*math.pi/180, a=0)
        min_chi.limits['pid'] = (0,1)
        min_chi.limits['pih'] = (0,1)
        min_chi.limits['pjj'] = (0,1)
        min_chi.limits['pjh'] = (0,1)
        min_chi.limits['theta12'] = (t12i, t12f)
        min_chi.limits['a'] = (ai, af)
        min_chi.errordef = Minuit.LEAST_SQUARES
        min_chi.migrad()
        bf = min_chi.values
        print(min_chi)

        #100 x 100 -> 91h
        ti = time()
        from scan_manual_contour_pi import two_dim_scan_v2
        info_i = 'pid', 0.001, 1
        info_j = 'pih', 0.001, 1
        info_k = 'pjj', 0.001, 1
        info_l = 'pjh', 0.001, 1
        # p1_list, p2_list, chi_list, k_aux, flag_aux = two_dim_scan_v2(min_chi, info_i, info_j, iterate=20, subtract_min=False)
        result = two_dim_scan_v2(min_chi, info_i, info_j, info_k, info_l, iterate=20, subtract_min=False)
        print('Time spent:',(time()-ti)/60/60,'h')
        name_file = 'results/contour_pid_pih_pjj_pjh_' + config_list[0] + '_' + str(D) +'kpc_' + detector + '_' + config_list[5] + '_scan.npy'
        name_file = sub(' ','_', name_file)
        name_file = sub('-','_', name_file)
        contour = np.array(result, dtype=object)
        np.save(name_file, contour)
        exit()


def minimize_chi_square_pi_t(config_list):
    from chi_square_pi_t import chi_square

    # 80 x 80 = 6400 -> 43.41 hours
    nsize = 100
    scan = True
    
    type_stat, detector, model = config_list[3], config_list[2], config_list[0]
    D = config_list[1] #kpc
    ai, af = -4*0.4, 4*0.4
    t12i, t12f = (33.45-0.75)*math.pi/180, (33.45+0.77)*math.pi/180

    if type_stat == 'contour_pull':
        
        min_chi = Minuit(chi_square, P1=0.5, P2=0.5, theta12=33.45*math.pi/180, a=0.5)
        min_chi.limits['P1'] = (0,1)
        min_chi.limits['P2'] = (0,1)
        min_chi.limits['theta12'] = (t12i, t12f)
        min_chi.limits['a'] = (ai, af)
        min_chi.errordef = Minuit.LEAST_SQUARES
        min_chi.migrad()
        bf = min_chi.values

        if scan == True:
            from time import time
            #100 x 100 -> 91h
            ti = time()
            from scan_manual_contour_pi_t import two_dim_scan_v2
            info_i = 'P1', 0, 1, nsize
            info_j = 'P2', 0, 1, nsize
            # p1_list, p2_list, chi_list, k_aux, flag_aux = two_dim_scan_v2(min_chi, info_i, info_j, iterate=20, subtract_min=False)
            result = two_dim_scan_v2(min_chi, info_i, info_j, iterate=20, subtract_min=False)
            print('Time spent:',(time()-ti)/60/60,'h')
            name_file = 'results/contour_p1_p2_t_' + config_list[0] + '_' + str(D) +'kpc_' + detector + '_' + config_list[5] + '_scan.npy'
            name_file = sub(' ','_', name_file)
            name_file = sub('-','_', name_file)
            contour = np.array(result, dtype=object)
            np.save(name_file, contour)
            exit()

        else:
            cl_value = 4
            mn = min_chi.mncontour('P1', 'P2', cl=cl_value, size=nsize, interpolated=200)
            p1 = mn[:,0]
            p2 = mn[:,1]
            P1_str, P2_str = 'p1','p2'
            par_name = P1_str + '-' + P2_str + '-' + str(cl_value)
            for i in range(len(p1)):
                pa_value = p1[i]
                pb_value = p2[i]
                cursor.execute("insert into contours values (NULL,NULL,?,?,?,?,?,?,?,?,?,?)", (config_list[0],config_list[1],config_list[2],config_list[3],config_list[4],config_list[5],config_list[6], par_name, pa_value, pb_value))
            if '(m)' in config_list[5]:
                P1_str = P1_str + '_m'
                P2_str = P2_str + '_m'
            
            name_file = 'results/contour_'+ model + '_' + P1_str + '_' + P2_str + '_' + str(D) +'kpc_' + detector + '_' + str(cl_value) + '_' + config_list[5] + '.npy'
            name_file = sub(' ','_', name_file)
            name_file = sub('-','_', name_file)
            np.save(name_file, [p1,p2,bf])
    else:
        print('Not implemented yet! (And probably will not, sorry...)')


def minimize_chi_square_Pi(config_list):
    from old.chi_square_Pi import chi_square

    nsize = 100
    
    type_stat, detector, model = config_list[3], config_list[2], config_list[0]
    D = config_list[1] #kpc
    ai, af = -4*0.4, 4*0.4

    if type_stat == 'contour_pull':

        if config_list[5] == 'NH vs free Px1,Px2,Pe3 in NH' or config_list[5] == 'IH vs free Px1,Px2,Pe3 in NH + OQS' or config_list[5] == 'IH vs free Px1,Px2,Pe3 in NH + OQS-loss' or config_list[5] == 'NH(m) vs free Px1,Px2,Pe3 in NH' or config_list[5] == 'IH(m) vs free Px1,Px2,Pe3 in NH + OQS' or config_list[5] == 'IH(m) vs free Px1,Px2,Pe3 in NH + OQS-loss':
            Pa_str = 'Px1'
            Pb_str = 'Pe3'
        elif config_list[5] == 'NH vs free Px1,Px3,Pe2 in IH + OQS-loss' or config_list[5] == 'IH vs free Px1,Px3,Pe2 in IH' or config_list[5] == 'NH vs free Px1,Px3,Pe2 in IH + OQS' or config_list[5] == 'IH(m) vs free Px1,Px3,Pe2 in IH' or config_list[5] == 'NH(m) vs free Px1,Px3,Pe2 in IH + OQS' or config_list[5] == 'NH(m) vs free Px1,Px3,Pe2 in IH + OQS-loss':
            Pa_str = 'Px1'
            Pb_str = 'Pe2'
        
        min_chi = Minuit(chi_square, P1=0.5, P2=0.5, a=0.5)
        min_chi.limits['P1'] = (0,1)
        min_chi.limits['P2'] = (0,1)
        min_chi.limits['a'] = (ai, af)
        min_chi.errordef = Minuit.LEAST_SQUARES
        min_chi.migrad()

        cl_value = 0.9
        mn = min_chi.mncontour('P1', 'P2', cl=cl_value, size=nsize)
        pa = mn[:,0]
        pb = mn[:,1]
        # print(pa,pb)
        par_name = Pa_str + '-' + Pb_str
        for i in range(len(pa)):
            pa_value = pa[i]
            pb_value = pb[i]
            cursor.execute("insert into contours values (NULL,NULL,?,?,?,?,?,?,?,?,?,?)", (config_list[0],config_list[1],config_list[2],config_list[3],config_list[4],config_list[5],config_list[6], par_name, pa_value, pb_value))
        if '(m)' in config_list[5]:
            Pa_str = Pa_str + '_m'
            Pb_str = Pb_str + '_m'
        
        name_file = 'results/contour_'+ model + '_' + Pa_str +'_'+ Pb_str +'_'+ str(D) +'kpc_' + detector + '_' + str(cl_value) + '_' + config_list[5] + '.npy'
        name_file = sub(' ','_', name_file)
        name_file = sub('-','_', name_file)
        np.save(name_file, [pa,pb])
    else:
        print('Not implemented yet! (And probably will not, sorry...)')  



def minimize_chi_square_Pee(config_list):
    from chi_square_Pee import chi_square
    nsize = 100
    type_stat, detector = config_list[3], config_list[2]
    ai, af = -4*0.4, 4*0.4
    cl = 0.95

    #values to confidence level
    cl_1, cl_2, cl_3, cl_4, cl_5 = 0.682689492137086, 0.954499736103642, 0.997300203936740, 0.999936657516334, 0.999999426696856
    
    D = config_list[1] #kpc

    
    min_chi = Minuit(chi_square, Pee=0.5, Pee_bar=0.5, a=0.5)
    
    min_chi.limits['Pee'] = (0,1)
    min_chi.limits['Pee_bar'] = (0,1)
    min_chi.limits['a'] = (ai, af)

    min_chi.errordef = Minuit.LEAST_SQUARES
    min_chi.migrad()
    # print(min_chi)
    
    if type_stat == 'contour_pull':
        p1, p2  = min_chi.mncontour("Pee", "Pee_bar", cl=cl, size=nsize)
        np.save('results/contour_Pee_Pee_bar_'+ str(D) +'kpc_' + detector + '_' + str(int(cl*100)) + 'CL_' + config_list[5] + '.npy', [p1,p2])

        par_name = "Pee-Pee_bar"
        for i in range(len(p1)):
            p1_value = p1[i]
            p2_value = p2[i]
            cursor.execute("insert into contours values (NULL,NULL,?,?,?,?,?,?,?,?,?,?)", (config_list[0],config_list[1],config_list[2],config_list[3],config_list[4],config_list[5],config_list[6], par_name, p1_value, p2_value))
    
    elif type_stat == 'profile_pull':
        par_name = "Pee"

        p, chi, ok = min_chi.mnprofile(par_name, bound=(0,1), size=nsize)
        profile_Pee = p, chi, ok
        np.save('results/profile_' + config_list[0] + '_' + par_name + '_'+ str(D) + 'kpc_' + detector + '_' + config_list[5] + '_thetaz_' + str(config_list[6]) + '.npy', profile_Pee)

        for i in range(len(p)):
            p_value = p[i]
            chi_value = chi[i]
            cursor.execute("insert into profiles values (NULL,NULL,?,?,?,?,?,?,?,?,?,?)", (config_list[0],config_list[1],config_list[2],config_list[3],config_list[4],config_list[5],config_list[6], par_name, p_value, chi_value))


#################### OQS (Gamma3, Gamma8), OQS-loss (Gamma) and OQS with Gamma = Gamma(E) ######################

def minimize_chi_square_oqs(config_list):
    from chi_square import chi_square_non_diag
    from chi_square import chi_square_D
    from chi_square import chi_square, chi_square_m
    from chi_square import chi_square_loss, chi_square_loss_m

    # nsize = 19 #5kpc, 11.2
    nsize = 50

    D, type_anal, type_stat, detector = config_list[1], config_list[4], config_list[3], config_list[2]

    def g_exp_limits(D_kpc, type_anal):
        D_eV = D_kpc * 1/(0.197e9 * 1e-15) * 3.086e19
        Ei,Ef = 3e6,50e6
        E_list = [Ei,Ef]
        gamma0_list = []
        
        if type_anal == 'OQS-loss-E-2.5':
            n = 2.5
            li,lf = 1e-5, 5e-1
        elif type_anal == 'OQS-loss-E':
            li,lf = 1e-4, 1e-1
            n = 2
        elif type_anal == 'OQS-loss':
            li,lf = 1e-3, 3
            n = 0

        for E in E_list:
            gamma0 = li/(E**n * D_eV)
            gamma0_list.append(gamma0)

        g_min = min(gamma0_list)
        g0i = np.log10(g_min)

        gamma0_list = []
        for E in E_list:
            gamma0 = lf/(E**n * D_eV)
            gamma0_list.append(gamma0)

        g_max = max(gamma0_list)
        g0f = np.log10(g_max)

        return g0i, g0f


    if type_anal == 'OQS-E' or type_anal == 'OQS-E ND':
        g_expi, g_expf = -52,-32
        # g_expi, g_expf = -56,-26 #1 kpc
        # g_expi, g_expf = -33-12,-27-12 #1 kpc
        g_expi, g_expf = -33-12,-22-12 #1 kpc
    elif type_anal == 'OQS-loss':
        # g_expi, g_expf = -33,-26
        # g_expi, g_expf = -33,-27.2
        g_expi, g_expf = g_exp_limits(D, type_anal)
        g_expi = -35
        # g_expi, g_expf = -40,-24
        g_expi, g_expf = -30,-26
    elif type_anal == 'OQS-loss-E':
        # g_expi, g_expf = -55,-39
        g_expi, g_expf = g_exp_limits(D, type_anal)
        g_expi = -50
        g_expi, g_expf = -48,-39
        g_expi, g_expf = -48,-40
    elif type_anal == 'OQS-E-2.5' or type_anal == 'OQS-E-2.5 ND':
        g_expi, g_expf = -55,-35
        g_expi, g_expf = -58,-42
        g_expi, g_expf = -33-15,-28.5-15 #1kpc
        g_expi, g_expf = -31-15,-21-15
    elif type_anal == 'OQS-loss-E-2.5':
        # g_expi, g_expf = -52,-42
        g_expi, g_expf = g_exp_limits(D, type_anal)
        g_expi = -55
        g_expi, g_expf = -52,-43
    elif type_anal == 'OQS conserved':
        g_expi, g_expf = -19, -9
        g_expi, g_expf = -19, -15 #betelgeuse
    elif type_anal == 'OQS-E conserved':
        g_expi, g_expf = -38, -20
        g_expi, g_expf = -33, -21
    elif type_anal == 'OQS-E-2.5 conserved':
        g_expi, g_expf = -38, -20
        g_expi, g_expf = -34, -24
    elif type_anal == 'OQS conserved D':
        g_expi, g_expf = -22, -10
        Di, Df = 0.001, 20
    else:
        if type_stat == 'profile_pull':
            g_expi, g_expf = -29,-22.9 #5kpc, 11.2
            g_expi, g_expf = -29,-22.9
            g_expi, g_expf = -28.5,-22.9
            g_expi, g_expf = -31,-26.5 #betelgeuse
        elif type_stat == 'contour_pull':
            g_expi, g_expf = -32,-24
    
    guess = (g_expi + g_expf)/2

    t12i, t12f = (33.45-0.75)*math.pi/180, (33.45+0.77)*math.pi/180
    t13i, t13f = (8.62-0.12)*math.pi/180, (8.62+0.12)*math.pi/180
    d21i, d21f = (7.42-0.20)*1e-5, (7.42+0.21)*1e-5 #eV^2
    d31i, d31f = (2.510-0.027)*1e-3, (2.510+0.027)*1e-3 #eV^2
    ai, af = -4*0.4, 4*0.4
    g = list()
    
    #no matter effects
    if '(m)' not in config_list[5]:
        zenith_str = ''
        if config_list[4] == 'OQS' or config_list[4] == 'OQS-E' or config_list[4] == 'OQS-E-2.5':
            par_name = 'g8_exp'
            # guess = -44 # OQS-E and OQS-E-2.5
            guess = g_expi + 0.5
            min_chi = Minuit(chi_square, g3_exp=guess, g8_exp=guess, theta12=33.45*math.pi/180, a=0.0)
            min_chi.limits['g3_exp'] = (g_expi, g_expf)
            # min_chi.limits['g3_exp'] = (-50,-42) #1 kpc

        elif config_list[4] == 'OQS conserved' or config_list[4] == 'OQS-E conserved' or config_list[4] == 'OQS-E-2.5 conserved':
            #Here I use the chi square for non-diag model, since it is the same for the proposed conserved energy model
            par_name = 'g_exp'
            min_chi = Minuit(chi_square_non_diag, g_exp=guess, theta12=33.45*math.pi/180, a=0.5)
            min_chi.limits['g_exp'] = (g_expi, g_expf)

        elif config_list[4] == 'OQS conserved D' or config_list[4] == 'OQS-E conserved D' or config_list[4] == 'OQS-E-2.5 conserved D':
            #Here I use the chi square for variable distance D
            par_name = 'D_g_exp'
            min_chi = Minuit(chi_square_D, D=0.1, g_exp=guess, theta12=33.45*math.pi/180, a=0.5)
            min_chi.limits['g_exp'] = (g_expi, g_expf)
            min_chi.limits['D'] = (Di, Df)

        elif config_list[4] == 'OQS ND' or config_list[4] == 'OQS-E ND' or config_list[4] == 'OQS-E-2.5 ND':
            par_name = 'g_exp'
            min_chi = Minuit(chi_square_non_diag, g_exp=guess, theta12=33.45*math.pi/180, a=0.5)
            min_chi.limits['g_exp'] = (g_expi, g_expf)

        elif config_list[4] == 'OQS-loss' or config_list[4] == 'OQS-loss-E' or config_list[4] == 'OQS-loss-E-2.5':
            par_name = 'g_exp'
            min_chi = Minuit(chi_square_loss, g_exp=guess, theta12=33.45*math.pi/180, a=0.5)

        elif config_list[4] == 'Hierarchy':
            par_name = 'theta12'
            min_chi = Minuit(chi_square, g3_exp=-100, g8_exp=-100, theta12=33.45*math.pi/180, a=0.5)
            t12i, t12f = 27.0*math.pi/180, 40.0*math.pi/180 #more than 3 sigma range
            min_chi.limits['theta12'] = (t12i, t12f)
            min_chi.fixed['g3_exp'] = True
            min_chi.fixed['g8_exp'] = True

    #with matter effects
    else:
        zenith_str = '_zenith_' + str(int(config_list[6]))
        if config_list[4] == 'OQS' or config_list[4] == 'OQS-E' or config_list[4] == 'OQS-E-2.5':
            par_name = 'g8_exp'
            min_chi = Minuit(chi_square_m, g3_exp=guess, g8_exp=guess, theta12=33.45*math.pi/180, Deltam21=7.42e-5, a=0.5)
            min_chi.limits['g3_exp'] = (g_expi, g_expf)
        
        elif config_list[4] == 'OQS-loss' or config_list[4] == 'OQS-loss-E' or config_list[4] == 'OQS-loss-E-2.5':
            par_name = 'g_exp'
            min_chi = Minuit(chi_square_loss_m, g_exp=guess, theta12=33.45*math.pi/180, Deltam21=7.42e-5, a=0.5)
        
        min_chi.limits['Deltam21'] = (d21i, d21f)
    
    if par_name == 'g8_exp' and type_stat == 'contour_pull':
        g_expi, g_expf = -35,-27

    if config_list[4] != 'Hierarchy' and config_list[4] != 'OQS conserved D':
        min_chi.limits[par_name] = (g_expi, g_expf)
        min_chi.limits['theta12'] = (t12i, t12f)
        min_chi.limits['a'] = (ai, af)
        min_chi.errordef = Minuit.LEAST_SQUARES
        min_chi.migrad(iterate=30)
        # min_chi.minos()
        print(min_chi)
    else:
        min_chi.limits['theta12'] = (t12i, t12f)
        min_chi.limits['a'] = (ai, af)
        min_chi.errordef = Minuit.LEAST_SQUARES
        min_chi.migrad(iterate=30)

    if type_stat == 'contour_pull':
        scan = True
        if scan == False:
            cl_list = [0.9,2,3]
            for cl in cl_list:
                print(min_chi)
                # cl = 2
                cl_str = str(cl)
                # contour_g = min_chi.mncontour('g8_exp', 'g3_exp', cl=cl, size=80, interpolated=200)
                contour_g = min_chi.mncontour('g8_exp', 'g3_exp', cl=cl, size=80)
                
                name_file = 'results/contour_' + par_name + '_' + config_list[0] + '_' + str(D) +'kpc_' + cl_str + 'cl_' + detector + '_' + config_list[5] + '.npy'
                name_file = sub(' ','_', name_file)
                name_file = sub('-','_', name_file)
                np.save(name_file, contour_g)

                for i in range(len(contour_g)):
                    g8_value = contour_g[i][0]
                    g3_value = contour_g[i][1]
                    cursor.execute("insert into contours values (NULL,NULL,?,?,?,?,?,?,?,?,?,?)", (config_list[0],config_list[1],config_list[2],config_list[3],config_list[4],config_list[5],config_list[6], par_name, g8_value, g3_value))

        else:
            from time import time
            #100 x 100 -> 91h
            #20 x 20 -> 21h
            ti = time()
            from scan_manual_contour import two_dim_scan_v2
            if config_list[4] == 'OQS conserved D' or config_list[4] == 'OQS-E conserved D' or config_list[4] == 'OQS-E-2.5 conserved D':
                info_i = 'D', Di, Df, 100
                info_j = 'g_exp', -22, -10, 100
                name_file = 'results/contour_' + par_name + '_' + config_list[0] + '_' + str(D) +'kpc_' + detector + '_' + config_list[5] + '_scan.npy'
                name_file = sub(' ','_', name_file)
                name_file = sub('-','_', name_file)
                D_list, g_list, chi_list, chi_only = two_dim_scan_v2(min_chi, info_i, info_j, iterate=20, subtract_min=False)
                contour_g = np.array([D_list, g_list, chi_list, chi_only], dtype=object)
                np.save(name_file, contour_g)
            else:
                info_i = 'g8_exp', -36, -20, 80
                info_j = 'g3_exp', -33, -17, 80
                name_file = 'results/contour_' + par_name + '_' + config_list[0] + '_' + str(D) +'kpc_' + detector + '_' + config_list[5] + '_scan.npy'
                name_file = sub(' ','_', name_file)
                name_file = sub('-','_', name_file)
                g8_list, g3_list, chi_list, k_aux, flag_aux = two_dim_scan_v2(min_chi, info_i, info_j, iterate=20, subtract_min=False)
                contour_g = np.array([g8_list, g3_list, chi_list], dtype=object)
                np.save(name_file, contour_g)
            print('Time spent:',(time()-ti)/60/60,'h')
            exit()

    elif type_stat == 'profile_pull':
        if config_list[4] != 'Hierarchy':
            g, chi, ok = min_chi.mnprofile(par_name, bound=(g_expi,g_expf), size=nsize)
        else:
            g, chi, ok = min_chi.mnprofile(par_name, bound=(t12i,t12f), size=nsize)
        # g, chi, ok = min_chi.mnprofile(par_name, bound=5, size=nsize)
        profile_g = g, chi, ok
        
        name_file = 'results/profile_' + par_name + '_' + config_list[0] + '_' + str(D) +'kpc_' + detector + '_' + config_list[5] + zenith_str + '.npy'
        name_file = sub(' ','_', name_file)
        name_file = sub('-','_', name_file)
        np.save(name_file, profile_g)

        for i in range(len(g)):
            g_value = g[i]
            chi_value = chi[i]
            cursor.execute("insert into profiles values (NULL,NULL,?,?,?,?,?,?,?,?,?,?)", (config_list[0],config_list[1],config_list[2],config_list[3],config_list[4],config_list[5],config_list[6], par_name, g_value, chi_value))


def run_minimize(config_list):
    if config_list[4] == 'OQS' or config_list[4] == 'OQS-loss' or config_list[4] == 'OQS-E' or config_list[4] == 'OQS-loss-E' or config_list[4] == 'OQS-E-2.5' or config_list[4] == 'OQS-loss-E-2.5' or config_list[4] == 'OQS ND' or config_list[4] == 'OQS-E ND' or config_list[4] == 'OQS-E-2.5 ND' or config_list[4] == 'Hierarchy' or 'conserved' in config_list[4]:
        minimize_chi_square_oqs(config_list)
    elif config_list[4] == 'Pee':
        minimize_chi_square_Pee(config_list)
    elif config_list[4] == 'Pi':
        minimize_chi_square_Pi(config_list)
    elif config_list[4] == 'pi':
        minimize_chi_square_pi(config_list)

def reload_config(module_name):
    reload(module_name)

def call_function():
    import import_config
    reload_config(import_config)
    from import_config import config_list
    try:
        model, D_kpc, exper, type_stat, type_anal, scenario, zenith = config_list
    except:
        print('All possibilities were matched. The End! (run....py)')
        exit()
    config_list = [model, D_kpc, exper, type_stat, type_anal, scenario, zenith]
    return config_list

def call_individual():
    from menu import run_function
    config_list = run_function()
    return config_list

# choice = input('Do you want a (1) automated or (2) manual analysis? (1 or 2): ')
choice = 1

if int(choice) == 1:
    config_list = [0,0]
    while True:
        config_list = call_function()
        print('config_list =', config_list)
        run_minimize(config_list)
        now = datetime.now() # current date and time
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        model, D_kpc, exper, type_stat, type_anal, scenario, zenith = config_list
        cursor.execute("insert into runs values (NULL,?,?,?,?,?,?,?,?)",(date_time, model, D_kpc, exper, type_stat, type_anal, scenario, zenith))
        connection.commit()

elif int(choice) == 2:
    print('This option has to be fixed before usage!')
    exit()
    # from datetime import datetime
    # from menu import run_function
    # config_list = run_function()
    config_list = call_individual()
    # now = datetime.now()
    # date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    # config_list = call_function()
    # model, D_kpc, exper, type_stat, type_anal, scenario, zenith = config_list
    # cursor.execute("insert into runs values (NULL,?,?,?,?,?,?,?,?)",(date_time, model, D_kpc, exper, type_stat, type_anal, scenario, zenith))
    # connection.commit()
    run_minimize(config_list)

connection.commit()
connection.close()
