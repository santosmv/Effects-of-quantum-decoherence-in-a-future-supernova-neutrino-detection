import numpy as np
import sqlite3
import platform
import socket
import sys
import inspect
import os

pc_name = socket.gethostname()
#DOCS: https://docs.python.org/3/library/sqlite3.html
if platform.system() == 'Windows':
    connection = sqlite3.connect('data/database_windows.db')
else:
    data_path = 'data/database.db'
    connection = sqlite3.connect(data_path)
cursor = connection.cursor()


def run_function():
    def run_loop():
        type_anal = 0
        D_kpc = 0
        type_stat = 0
        model = 0
        exper = 0
        zenith = 0

        models = ['LS220-s11.2c','LS220-s27.0c','LS220-s27.0co','Shen-s11.2c']
        expers = ['dune','hk','dune_hk']
        type_stats = ['profile_pull','contour_pull']
        type_anals = ['OQS','Pee','Pi','OQS-loss']
        zenith_list = [120,140,160,180]
            
        while model not in (1,2,3,4):
            model = input('Select a model:\n 1.' + models[0] + '\n 2.' + models[1] + '\n 3.' + models[2] + '\n 4.' + models[3] + '\nValue:')
            model = int(model)
        
        while D_kpc < 0.00001 or D_kpc > 500:
            D_kpc = input('Enter with a Supernova distance in kpc:')
            D_kpc = float(D_kpc)
        
        while exper not in (1,2,3):
            exper = input(' 1.Dune\n 2.HK\n 3.DUNE+HK\nValue:')
            exper = int(exper)
        
        while type_stat not in (1,2):
            type_stat = input(' 1.Chi square profile\n 2.Contour of two parameters\nValue:')
            type_stat = int(type_stat)
        
        while type_anal not in (1,2,3,4):
            type_anal = input(' 1.Parameters of OQS\n 2.Calculating a free Pee\n 3.Calculating a free Pi_alpha to triangle plot\n 4.Parameters of OQS with nu loss\nValue:')
            type_anal = int(type_anal)
        
        scenario = 0

        if type_anal == 1:
            scenarios = ['NH vs NH + OQS','NH(m) vs NH(m) + OQS','IH vs IH + OQS','IH(m) vs IH(m) + OQS','IH vs NH + OQS','IH(m) vs NH(m) + OQS']
            while scenario not in (1,2,3,4,5,6):
                scenario = input('Select a scenario to be investigated:\n \
        1.NH vs NH + OQS\n \
        2.NH(m) vs NH(m) + OQS\n \
        3.IH vs IH + OQS\n \
        4.IH(m) vs IH(m) + OQS\n \
        5.IH vs NH + OQS\n \
        6.IH(m) vs NH(m) + OQS \n\n\
        Obs: (m) means matter effects.\n\
        Value:')
                scenario = int(scenario)
            scenario = scenarios[scenario-1]
        
        elif type_anal == 2:
            scenarios = ['NH','IH','NH(m)','IH(m)']
            while scenario not in (1,2,3,4):
                scenario = input('Select a scenario to be investigated:\n \
        1.NH vs a free Pee and Pee_bar\n \
        2.IH vs a free Pee and Pee_bar\n \
        3.NH(m) vs a free Pee and Pee_bar\n \
        4.IH(m) vs a free Pee and Pee_bar\n \
        Obs: (m) means matter effects.\n\
        Value:')
                scenario = int(scenario)
            scenario = scenarios[scenario-1]
        
        elif type_anal == 3:
            scenarios = ['NH vs free Px1,Px2,Pe3 in NH','IH vs free Px1,Px3,Pe2 in IH','NH(m) vs free Px1,Px2,Pe3 in NH','IH(m) vs free Px1,Px3,Pe2 in IH', 'NH vs free Px1,Px3,Pe2 in IH + OQS', 'IH vs free Px1,Px2,Pe3 in NH + OQS', 'NH(m) vs free Px1,Px3,Pe2 in IH + OQS', 'IH(m) vs free Px1,Px2,Pe3 in NH + OQS','NH vs free Px1,Px3,Pe2 in IH + OQS-loss','IH vs free Px1,Px2,Pe3 in NH + OQS-loss','NH(m) vs free Px1,Px3,Pe2 in IH + OQS-loss','IH(m) vs free Px1,Px2,Pe3 in NH + OQS-loss']

            while scenario not in (1,2,3,4,5,6,7,8,9,10,11,12):
                scenario = input('Select a scenario to be investigated:\n \
        1.NH vs free Px1,Px2,Pe3 in NH \n \
        2.IH vs free Px1,Px3,Pe2 in IH \n \
        3.NH(m) vs free Px1,Px2,Pe3 in NH \n \
        4.IH(m) vs free Px1,Px3,Pe2 in IH \n \
        5.NH vs free Px1,Px3,Pe2 in IH + OQS \n \
        6.IH vs free Px1,Px2,Pe3 in NH + OQS \n \
        7.NH(m) vs free Px1,Px3,Pe2 in IH + OQS \n \
        8.IH(m) vs free Px1,Px2,Pe3 in NH + OQS \n \
        9.NH vs free Px1,Px3,Pe2 in IH + OQS-loss \n \
        10.IH vs free Px1,Px2,Pe3 in NH + OQS-loss \n \
        11.NH(m) vs free Px1,Px3,Pe2 in IH + OQS-loss \n \
        12.IH(m) vs free Px1,Px2,Pe3 in NH + OQS-loss \n \
        Obs: (m) means matter effects.\n\
        Value:')
                scenario = int(scenario)
            scenario = scenarios[scenario-1]

        elif type_anal == 4:
            scenarios = ['NH vs NH + OQS-loss','NH(m) vs NH(m) + OQS-loss','IH vs IH + OQS-loss','IH(m) vs IH(m) + OQS-loss','IH vs NH + OQS-loss','IH(m) vs NH(m) + OQS-loss']
            while scenario not in (1,2,3,4,5,6):
                scenario = input('Select a scenario to be investigated:\n \
        1.NH vs NH + OQS with nu loss\n \
        2.NH(m) vs NH(m) + OQS with nu loss\n \
        3.IH vs IH + OQS with nu loss\n \
        4.IH(m) vs IH(m) + OQS with nu loss\n \
        5.IH vs NH + OQS with nu loss\n \
        6.IH(m) vs NH(m) + OQS with nu loss \n\n\
        Obs: (m) means matter effects.\n\
        Value:')
                scenario = int(scenario)
            scenario = scenarios[scenario-1]

        model = models[model-1]
        type_stat = type_stats[type_stat-1]
        type_anal = type_anals[type_anal-1]
        exper = expers[exper-1]

        if '(m)' in scenario:
            zenith_opt = 0
            while zenith_opt not in (1,2,3,4):
                zenith_opt = input('What is the zenith angle for DUNE?\n \
        1. 120 deg\n \
        2. 140 deg\n \
        3. 160 deg\n \
        4. 180 deg\n \
        Value:')
                zenith_opt = int(zenith_opt)
            zenith = zenith_list[zenith_opt-1]

        return model, D_kpc, exper, type_stat, type_anal, scenario, zenith

    model, D_kpc, exper, type_stat, type_anal, scenario, zenith = run_loop()
    
    c_runs = None
    tuple_config = (model, D_kpc, exper, type_stat, type_anal, scenario, zenith)
    
    try:
        cursor.execute("select * from runs where model = ? and D_kpc = ? and exper = ? and type_stat = ? and type_anal = ? and scenario = ? and zenith is ? ", (model, D_kpc, exper, type_stat, type_anal, scenario, zenith))
        c_runs = cursor.fetchall()[0][2:]
    except:
        print('This option has not been calculated at this computer yet!')
    
    if c_runs == tuple_config:
        answer = '0'
        while answer != ('y' or 'yes' or 'n' or 'no'):
            answer = input('This data was already taken! Are you sure you want to run it again? (yes/no): ')
            if answer == 'n' or answer == 'no':
                exit()
    
    from datetime import datetime
    now = datetime.now() # current date and time
    date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    cursor.execute("insert into runs values (NULL,?,?,?,?,?,?,?,?)",(date_time, model, D_kpc, exper, type_stat, type_anal, scenario, zenith))
    connection.commit()
    cursor.execute("select id from runs where date = '%s'"%date_time)
    id_value = cursor.fetchall()[0][0]

    return model, D_kpc, exper, type_stat, type_anal, scenario, zenith#, id_value, date_time


def run_function_automated():
    cursor.execute('delete from D_done')
    connection.commit()

    D_list = [1,5,7,10,20,50]
    job_id = os.getpid()
    
    cursor.execute("select job_id from D_done")
    jobs = np.array(cursor.fetchall())

    if len(jobs) == 0:
        D_kpc = D_list[0]
        cursor.execute("insert into D_done values (?,?)",(job_id, D_kpc,))
        connection.commit()
    
    else:
        jobs = jobs[:,0]
        if job_id in jobs:
            cursor.execute("select D_kpc from D_done where job_id = ?",(job_id,))
            D_kpc = np.array(cursor.fetchall())[0][0]
        else:
            if len(jobs) >= len(D_list):
                D_kpc = None
                print('All D_kpc values were tested. The End!')
                exit()
            else:
                D_kpc = D_list[len(jobs)]
                cursor.execute("insert into D_done values (?,?)",(job_id, D_kpc,))
                connection.commit()

    choice_m = 1
    models = ['LS220-s11.2c','LS220-s27.0c','LS180-s40.0','LS180-s17.8']
    type_stats = ['profile_pull']
    type_anals = ['Hierarchy','OQS','OQS ND','OQS-loss','OQS-E','OQS-E ND','OQS-loss-E','OQS-E-2.5','OQS-E-2.5 ND','OQS-loss-E-2.5','OQS conserved','OQS-E conserved','OQS-E-2.5 conserved']
    expers = ['dune','hk','jn','dune_hk_jn']

    # msc e
    # models = ['LS220-s11.2c']
    # type_stats = ['profile_pull']
    # type_anals = ['OQS']
    # expers = ['hk']
    # D_kpc = 0.1681 #kpc
    # D_kpc = 1.0 #kpc

    # expers = ['dune_hk_jn']
    
    #conserved D
    models = ['LS180-s40.0']
    type_stats = ['contour_pull']
    expers = ['hk']
    type_anals = ['OQS conserved D']

    #conserved
    # models = ['LS180-s40.0']
    # expers = ['hk']
    # type_anals = ['OQS conserved']
    # D_kpc = 0.1681 #kpc

    #OQS ND
    # type_anals = ['OQS ND','OQS-E ND','OQS-E-2.5 ND']

    # loss
    # models = ['LS220-s27.0c']
    # type_stats = ['profile_pull']
    # type_anals = ['OQS-loss','OQS-loss-E','OQS-loss-E-2.5']
    # type_anals = ['OQS-loss']
    # expers = ['dune']
    # D_kpc = 10

    # type_anals = ['OQS']
    # expers = ['hk']


    #hierarchy theta12
    # type_anals = ['Hierarchy']
    # expers = ['dune_hk_jn']
    # models = ['LS220-s27.0c']
    # D_kpc = 10.0

    #matter effects
    # type_anals = ['OQS','OQS-E']
    # expers = ['dune_hk_jn']
    # models = ['LS220-s27.0c']
    
    #contour
    # type_stats = ['contour_pull']
    # expers = ['dune_hk_jn']
    # models = ['LS220-s27.0c']
    # type_anals = ['OQS']
    # D_kpc = 10.0

    #pi analysis
    # models = ['LS220-s11.2c']
    # expers = ['dune_hk_jn']
    # type_stats = ['contour_pull']
    # type_anals = ['pi']
    # D_kpc = 10.0
    
    if int(choice_m) == 1:
        scenarios_oqs = ['NH vs NH + OQS','IH vs IH + OQS','IH vs NH + OQS']
        scenarios_oqs = ['IH vs IH + OQS'] #remover
        scenarios_oqs_nd = ['NH vs NH + OQS ND','IH vs IH + OQS ND','IH vs NH + OQS ND']
        # scenarios_loss = ['NH vs NH + OQS-loss','IH vs IH + OQS-loss','NH vs IH + OQS-loss']
        scenarios_loss = ['IH vs IH + OQS-loss']
        scenarios_Pee = ['NH','IH']
        scenarios_oqs_E = ['NH vs NH + OQS-E','IH vs IH + OQS-E','IH vs NH + OQS-E']
        scenarios_oqs_E = ['IH vs IH + OQS-E']
        scenarios_oqs_E = ['NH vs NH + OQS-E']
        scenarios_oqs_E_nd = ['NH vs NH + OQS-E ND','IH vs IH + OQS-E ND','IH vs NH + OQS-E ND']
        # scenarios_loss_E = ['NH vs NH + OQS-loss-E','IH vs IH + OQS-loss-E','NH vs IH + OQS-loss-E']
        scenarios_loss_E = ['IH vs IH + OQS-loss-E']
        scenarios_oqs_E_2_5 = ['NH vs NH + OQS-E-2.5','IH vs IH + OQS-E-2.5','IH vs NH + OQS-E-2.5']
        scenarios_oqs_E_2_5 = ['IH vs IH + OQS-E-2.5']
        scenarios_oqs_E_2_5_nd = ['NH vs NH + OQS-E-2.5 ND','IH vs IH + OQS-E-2.5 ND','IH vs NH + OQS-E-2.5 ND']
        # scenarios_loss_E_2_5 = ['NH vs NH + OQS-loss-E-2.5','IH vs IH + OQS-loss-E-2.5','NH vs IH + OQS-loss-E-2.5']
        scenarios_loss_E_2_5 = ['IH vs IH + OQS-loss-E-2.5']
        scenarios_Pi = ['NH vs free Px1,Px2,Pe3 in NH','IH vs free Px1,Px3,Pe2 in IH', 'NH vs free Px1,Px3,Pe2 in IH + OQS', 'IH vs free Px1,Px2,Pe3 in NH + OQS', 'NH vs free Px1,Px3,Pe2 in IH + OQS-loss','IH vs free Px1,Px2,Pe3 in NH + OQS-loss']
        scenarios_pi = ['NH','IH']
        scenarios_hier = ['NH vs IH']
        scenarios_oqs_conserved = ['NH vs NH + OQS conserved','IH vs IH + OQS conserved', 'IH vs NH + OQS conserved']
        scenarios_oqs_conserved = ['IH vs IH + OQS conserved']
        scenarios_oqs_conserved_E = ['NH vs NH + OQS-E conserved','IH vs IH + OQS-E conserved', 'IH vs NH + OQS-E conserved']
        scenarios_oqs_conserved_E_2_5 = ['NH vs NH + OQS-E-2.5 conserved','IH vs IH + OQS-E-2.5 conserved', 'IH vs NH + OQS-E-2.5 conserved']
        scenarios_oqs_conserved_D = ['IH vs IH + OQS conserved D']
        zenith_list = [0]

    elif int(choice_m) == 2:
        scenarios_oqs = ['NH(m) vs NH(m) + OQS','IH(m) vs NH(m) + OQS']
        scenarios_Pee = ['NH(m)','IH(m)']
        scenarios_loss = ['IH(m) vs IH(m) + OQS-loss','NH(m) vs IH(m) + OQS-loss']
        scenarios_oqs_E = ['NH(m) vs NH(m) + OQS-E','IH(m) vs IH(m) + OQS-E','IH(m) vs NH(m) + OQS-E']
        scenarios_loss_E = ['NH(m) vs NH(m) + OQS-loss-E','IH(m) vs IH(m) + OQS-loss-E','NH(m) vs IH(m) + OQS-loss-E']
        scenarios_oqs_E_2_5 = ['NH(m) vs NH(m) + OQS-E-2.5','IH(m) vs NH(m) + OQS-E-2.5']
        scenarios_loss_E_2_5 = ['IH(m) vs IH(m) + OQS-loss-E-2.5','NH(m) vs IH(m) + OQS-loss-E-2.5']
        scenarios_Pi = ['NH(m) vs free Px1,Px2,Pe3 in NH','IH(m) vs free Px1,Px3,Pe2 in IH', 'NH(m) vs free Px1,Px3,Pe2 in IH + OQS', 'IH(m) vs free Px1,Px2,Pe3 in NH + OQS', 'NH(m) vs free Px1,Px3,Pe2 in IH + OQS-loss','IH(m) vs free Px1,Px2,Pe3 in NH + OQS-loss']

    else:
        print('Are you kidding me???')
        sys.exit()
    
    # if pc_name == 'neutrino7':
    #     #model = 'LS220-s27.0c'
    #     model = 'LS220-s11.2c'
    # elif pc_name == 'drcpc65':
    #     model = 'LS220-s27.0c'
    # elif pc_name == 'Marconis':
    #     #model = 'LS220-s11.2c'
    #     model = 'LS220-s27.0c'
    # else:
    #     model = 'LS220-s27.0c'
    #     #model = 'LS220-s11.2c'
    
    config_lists = []

    for model in models:
        for exper in expers:
            for type_stat in type_stats:
                for type_anal in type_anals:
                    if type_anal == 'OQS':
                        scenarios = scenarios_oqs
                    elif type_anal == 'OQS ND':
                        scenarios = scenarios_oqs_nd
                    elif type_anal == 'Pee':
                        scenarios = scenarios_Pee
                    elif type_anal == 'OQS-loss':
                        scenarios = scenarios_loss
                    elif type_anal == 'OQS-E':
                        scenarios = scenarios_oqs_E
                    elif type_anal == 'OQS-E ND':
                        scenarios = scenarios_oqs_E_nd
                    elif type_anal == 'OQS-loss-E':
                        scenarios = scenarios_loss_E
                    elif type_anal == 'OQS-E-2.5':
                        scenarios = scenarios_oqs_E_2_5
                    elif type_anal == 'OQS-E-2.5 ND':
                        scenarios = scenarios_oqs_E_2_5_nd
                    elif type_anal == 'OQS-loss-E-2.5':
                        scenarios = scenarios_loss_E_2_5
                    elif type_anal == 'Pi':
                        scenarios = scenarios_Pi
                    elif type_anal == 'pi':
                        scenarios = scenarios_pi
                    elif type_anal == 'Hierarchy':
                        scenarios = scenarios_hier
                    elif type_anal == 'OQS conserved':
                        scenarios = scenarios_oqs_conserved
                    elif type_anal == 'OQS-E conserved':
                        scenarios = scenarios_oqs_conserved_E
                    elif type_anal == 'OQS-E-2.5 conserved':
                        scenarios = scenarios_oqs_conserved_E_2_5
                    elif type_anal == 'OQS conserved D':
                        scenarios = scenarios_oqs_conserved_D
                    else:
                        sys.exit()
                    
                    if choice_m == 2:
                        if exper == 'dune':
                            zenith_list = [120,180]
                        elif exper == 'dune_hk_jn':
                            zenith_list = [120,180,320]
                        else:
                            zenith_list = [180,320]

                    for scenario in scenarios:
                        for zenith in zenith_list:
                            config_list = [model, D_kpc, exper, type_stat, type_anal, scenario, zenith]
                            config_lists.append(config_list)
    
    i = 0
    check_list = [0,0]
    while True:
        config_list = config_lists[i]
        model, D_kpc, exper, type_stat, type_anal, scenario, zenith = config_list
        #print(config_list)
        # cursor.execute("delete from runs")
        # cursor.execute("delete from runs where type_anal='pi_t'")
        # connection.commit()

        cursor.execute("select * from runs where model = ? and D_kpc = ? and exper = ? and type_stat = ? and type_anal = ? and scenario == ? and zenith = ?", (model, D_kpc, exper, type_stat, type_anal, scenario, zenith,))
        check_list = np.array(cursor.fetchall())
        i = i+1
        if len(check_list) == 0:
            return config_list

        if i > len(config_lists)-1:
            print('All possibilities were matched. The End! (menu....py)')
            connection.commit()
            exit()

# if bool_module:
#     print('entrou')
#     config_list = run_function_automated()

config_list = run_function_automated()

# if bool_module:
#     config_list_individual = run_function()
# else:
#     config_list_individual = None
