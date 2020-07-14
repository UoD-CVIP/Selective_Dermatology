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

# Own Modules Imports
from utils import str_to_bool


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
    argument_parser.add_argument("--config_file", type=str,
                                 default="config.ini",
                                 help="String representing the file path to the config file.")
    argument_parser.add_argument("--experiment", type=str,
                                 default=config_parser["standard"]["experiment"],
                                 help="String representing the name of the current experiment.")
    argument_parser.add_argument("--seed", type=int,
                                 default=int(config_parser["standard"]["seed"]),
                                 help="Integer used to set the random seed. Set to -1 for a random seed.")

    # Dataset Arguments
    argument_parser.add_argument("--dataset_dir", type=str,
                                 default=config_parser["dataset"]["dataset_dir"],
                                 help="Directory path for the dataset.")
    argument_parser.add_argument("--image_x", type=int,
                                 default=int(config_parser["dataset"]["image_x"]),
                                 help="Integer for the x dimension of the image after resizing.")
    argument_parser.add_argument("--image_y", type=int,
                                 default=int(config_parser["dataset"]["image_y"]),
                                 help="Integer for the y dimension of the image after resizing.")
    argument_parser.add_argument("--augmentation", type=str_to_bool,
                                 default=config_parser["dataset"]["augmentation"].lower() == "true",
                                 help="Boolean value for if augmentation should be applied to training data.")

    # Returns the Argument Parser Namespace.
    arguments = argument_parser.parse_args()
    return vars(arguments)


def print_arguments(arguments):
    """
    Print all the arguments to the command line.
    :param arguments: ArgumentParser Namespace object.
    """

    # Cycles through all the arguments within the Namespace object.
    for key, value in arguments.items():
        print(f"{key: <24}: {value}")
