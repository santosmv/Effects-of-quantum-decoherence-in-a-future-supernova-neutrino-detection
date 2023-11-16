from importlib import reload

def reload_config(module_name):
    reload(module_name)

import menu
reload_config(menu)
from menu import config_list

# from menu import config_list_individual
# print(config_list_individual)
