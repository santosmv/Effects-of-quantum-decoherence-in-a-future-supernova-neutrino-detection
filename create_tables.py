import sqlite3
import platform
import socket
pc_name = socket.gethostname()

#DOCS: https://docs.python.org/3/library/sqlite3.html
if platform.system() == 'Windows':
    connection = sqlite3.connect('data/database_windows.db')
else:
    data_path = 'data/database.db'
    connection = sqlite3.connect(data_path)
cursor = connection.cursor()

def create_database_ec():
    cursor.execute("""create table elastic_cross(
                                        Enu real,
                                        Te real,
                                        nue real,
                                        numu real,
                                        antinue real,
                                        antinumu real
    )""")

def create_database_bin():
    cursor.execute("DROP TABLE bins")
    cursor.execute("""create table bins(
                                        detector text,
                                        hierarchy text,
                                        Eth real,
                                        bins real
    )""")

def create_database_work():
    cursor.execute("DROP TABLE work_function")
    cursor.execute("""create table if not exists work_function(
                                        bin_i real,
                                        detector text,
                                        hierarchy text,
                                        channel text,
                                        flavor text,
                                        Eth real,
                                        Enu real,
                                        Enui real,
                                        Enuf real,
                                        work real
    )""")

def create_database_work_loss():
    cursor.execute("DROP TABLE work_function_loss")
    cursor.execute("""create table if not exists work_function_loss(
                                        bin_i real,
                                        detector text,
                                        hierarchy text,
                                        channel text,
                                        flavor text,
                                        Eth real,
                                        Enu real,
                                        Enui real,
                                        Enuf real,
                                        work real
    )""")

def create_runs_table():
    cursor.execute('drop table runs')
    cursor.execute("""create table if not exists runs(
                                        id INTEGER PRIMARY KEY,
                                        date text,
                                        model text,
                                        D_kpc real,
                                        exper text,
                                        type_stat text,
                                        type_anal text,
                                        scenario text,
                                        zenith real
    )""")

def create_profiles_table():
    cursor.execute('drop table profiles')
    cursor.execute("""create table if not exists profiles(
                                        id INTEGER,
                                        date text,
                                        model text,
                                        D_kpc real,
                                        exper text,
                                        type_stat text,
                                        type_anal text,
                                        scenario text,
                                        zenith real,
                                        par_name text,
                                        par_value real,
                                        chi real
    )""")

def create_contours_table():
    cursor.execute('drop table contours')
    cursor.execute("""create table if not exists contours(
                                        id INTEGER,
                                        date text,
                                        model text,
                                        D_kpc real,
                                        exper text,
                                        type_stat text,
                                        type_anal text,
                                        scenario text,
                                        zenith real,
                                        par_names text,
                                        par1 real,
                                        par2 real
    )""")

def create_smearing_table():
    cursor.execute("""create table if not exists smearing(
                                        name_file text,
                                        Enu_reco real,
                                        Enu_true real
    )""")

def create_D_table():
    cursor.execute("create table if not exists D_done(job_id real, D_kpc real)")


# create_runs_table()
# create_profiles_table()
# create_contours_table()
# create_D_table()
# create_database_bin()
# create_database_work()
create_database_work_loss()

connection.commit()
connection.close()