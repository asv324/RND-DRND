""" import configparser

config = configparser.ConfigParser()
# config.read('./config.conf')
config.read('config.conf')

# ---------------------------------
default = 'DEFAULT'
# ---------------------------------
default_config = config[default] """
import configparser
import os

config = configparser.ConfigParser()

# Get the current directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config_DRND.conf')

# Read the config file
if not config.read(config_path):
    raise FileNotFoundError(f"Configuration file not found at: {config_path}")

default = 'DEFAULT'
default_config = config[default]
