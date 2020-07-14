# -*- coding: utf-8 -*-

"""
This file contains the function to read arguments from a config file and command line.
    load_arguments - Function to load arguments from the config file and command line.
    print_arguments - Function to print loaded arguments.
"""


# Built-in/Generic Imports
import sys
from argparse import ArgumentParser
from configparser import ConfigParser


__author__ = ["Jacob Carse", "Stephen Hogg"]
__credits__ = ["Jacob Carse", "Stephen Hogg"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


def load_arguments(description):
    """
    Loads arguments from a config file and command line.
    Arguments from command line overrides arguments from the config file.
    The config file will be loaded from the default location ./config.ini and can be overridden from the command line.
    :param description: The description of the application.
    :return: ArgumentParser Namespace object.
    """

    # Creates a ArgumentParser to read command line arguments.
    argument_parser = ArgumentParser(description=description)

    # Creates a ConfigParser to read the config file.
    config_parser = ConfigParser()

    # Loads either a specified config file or default config file.
    if len(sys.argv) > 1:
        if sys.argv[1] == "--config_file":
            config_parser.read(sys.argv[2])
        else:
            config_parser.read("config.ini")
    else:
        config_parser.read("config.ini")

    # Standard Arguments
    argument_parser.add_argument("--experiment", type=str,
                                 default=config_parser["standard"]["experiment"],
                                 help="String representing the name of the current experiment.")

    # Returns the Argument Parser Namespace.
    return argument_parser.parse_args()


def print_arguments(arguments):
    """
    Print all the arguments to the command line.
    :param arguments: ArgumentParser Namespace object.
    """

    # Cycles through all the arguments within the Namespace object.
    for argument in vars(arguments):
        print(f"{argument: <24}: {getattr(arguments, argument)}")
