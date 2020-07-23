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


__author__ = ["Jacob Carse", "Stephen Hogg", "Stephen McKenna"]
__copyright__ = "Copyright 2020, Selective Dermatology"
__credits__ = ["Jacob Carse", "Stephen Hogg", "Stephen McKenna"]
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
    argument_parser.add_argument("--tensorboard_dir", type=str,
                                 default=config_parser["standard"]["tensorboard_dir"],
                                 help="String representing the Directory path for the TensorBoard output.")

    # Debug Arguments
    argument_parser.add_argument("--fast_dev_run", type=str_to_bool,
                                 default=config_parser["debug"]["fast_dev_run"].lower() == "true",
                                 help="Boolean for if the model should run in debug mode.")

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
    argument_parser.add_argument("--val_split", type=float,
                                 default=float(config_parser["dataset"]["val_split"]),
                                 help="Floating point value for the dataset validation split.")
    argument_parser.add_argument("--test_split", type=float,
                                 default=float(config_parser["dataset"]["test_split"]),
                                 help="Floating point value for the dataset testing split.")

    # Performance Arguments
    argument_parser.add_argument("--efficient_net", type=int,
                                 default=int(config_parser["performance"]["efficient_net"]),
                                 help="Integer representing the compound coefficient of the EfficientNet.")
    argument_parser.add_argument("--precision", type=int,
                                 default=int(config_parser["performance"]["precision"]),
                                 help="Integer for the level of precision used to train the model. 16 or 32.")
    argument_parser.add_argument("--num_gpus", type=int,
                                 default=int(config_parser["performance"]["num_gpus"]),
                                 help="Integer for the number of GPUs to use to train the model.")
    argument_parser.add_argument("--data_workers", type=int,
                                 default=int(config_parser["performance"]["data_workers"]),
                                 help="Integer for the number of data loaders used to load the data.")
    argument_parser.add_argument("--distributed_backend", type=str,
                                 default=config_parser["performance"]["distributed_backend"],
                                 help="String for the type of distributed learning method to be used.")

    # Training Arguments
    argument_parser.add_argument("--starting_lr", type=float,
                                 default=float(config_parser["training"]["starting_lr"]),
                                 help="Floating point value for the starting learning rate. -1 for auto.")
    argument_parser.add_argument("--max_lr", type=float,
                                 default=float(config_parser["training"]["max_lr"]),
                                 help="Floating point value for the maximum learning rate. -1 for auto.")
    argument_parser.add_argument("--batch_size", type=int,
                                 default=int(config_parser["training"]["batch_size"]),
                                 help="Integer for the batch size. -1 for auto.")

    # Early Stopping Arguments
    argument_parser.add_argument("--min_delta", type=float,
                                 default=float(config_parser["early_stopping"]["min_delta"]),
                                 help="Floating point value for the minimum delta for early stopping.")
    argument_parser.add_argument("--patience", type=int,
                                 default=int(config_parser["early_stopping"]["patience"]),
                                 help="Integer for the patience used for early stopping.")
    argument_parser.add_argument("--max_epochs", type=int,
                                 default=int(config_parser["early_stopping"]["max_epochs"]),
                                 help="Integer for the maximum number of epochs used in training.")
    argument_parser.add_argument("--min_epochs", type=int,
                                 default=int(config_parser["early_stopping"]["min_epochs"]),
                                 help="Integer for the minimum number of epochs used in training.")

    # Selective Arguments
    argument_parser.add_argument("--alpha", type=float,
                                 default=float(config_parser["selective"]["alpha"]),
                                 help="Floating point value for the alpha value of the Selective loss.")
    argument_parser.add_argument("--lambda", type=int,
                                 default=int(config_parser["selective"]["lambda"]),
                                 help="Floating point value for the lambda value of the Selective loss.")
    argument_parser.add_argument("--coverage", type=float,
                                 default=float(config_parser["selective"]["coverage"]),
                                 help="Floating point value for the target coverage of the Selective loss.")
    argument_parser.add_argument("--drop_rate", type=float,
                                 default=float(config_parser["selective"]["drop_rate"]),
                                 help="Floating point value for the drop rate for the model's dropout layers.")
    argument_parser.add_argument("--drop_iterations", type=int,
                                 default=int(config_parser["selective"]["drop_iterations"]),
                                 help="Integer for the number of iterations to perform for MC Dropout.")

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
