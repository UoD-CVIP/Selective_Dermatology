# -*- coding: utf-8 -*-


"""
This file contains the function used to handle loading a configuration file and command line arguments.
    load_configurations - Function to load configurations from a configurations file and command line arguments.
    print_arguments - Function to print the loaded arguments.
"""


# Built-in/Generic Imports
import sys
from argparse import ArgumentParser
from configparser import ConfigParser

# Own Modules Import
from utils import log, str_to_bool


__author__    = ["Jacob Carse"]
__copyright__ = "Copyright 2020, Selective Dermatology"
__credits__   = ["Jacob Carse", "Stephen Hogg", "Stephen McKenna"]
__license__   = "MIT"
__version__   = "3.0.0"
__maintainer  = "Jacob Carse"
__email__     = "j.carse@dundee.ac.uk"
__status__    = "Development"


def load_configurations(description):
    """
    Loads arguments from a configuration file and command line.
    Arguments from the command line override arguments from the configuration file.
    The config file will be loaded from the default location "./config.ini" but can be overridden from the command line.
    :param description: The description of the application that is shown when using the "--help" command.
    :return: ArgumentParser Namespace object containing the loaded configurations.
    """

    # Creates an ArgumentParser to read the command line arguments.
    argument_parser = ArgumentParser(description=description)

    # Creates a ConfigParser to read configurations file arguments.
    config_parser = ConfigParser()

    # Loads wither a specified configurations file or file from the default location.
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
                                 help="String - File path to the config file.")
    argument_parser.add_argument("--task", type=str,
                                 default=config_parser["standard"]["task"],
                                 help="String - Task for the application to run.")
    argument_parser.add_argument("--seed", type=int,
                                 default=int(config_parser["standard"]["seed"]),
                                 help="Integer - Seed used to generate random numbers.")
    argument_parser.add_argument("--experiment", type=str,
                                 default=config_parser["standard"]["experiment"],
                                 help="String - The name of the current experiment.")

    # Logging Arguments
    argument_parser.add_argument("--verbose", type=str_to_bool,
                                 default=config_parser["logging"]["verbose"].lower() == "true",
                                 help="Boolean - Should outputs should be printed on the terminal.")
    argument_parser.add_argument("--log_dir", type=str,
                                 default=config_parser["logging"]["log_dir"],
                                 help="String - Directory path for where log files are stored.")
    argument_parser.add_argument("--log_interval", type=int,
                                 default=int(config_parser["logging"]["log_interval"]),
                                 help="Integer - Number of batches before printing to the console.")
    argument_parser.add_argument("--tensorboard_dir", type=str,
                                 default=config_parser["logging"]["tensorboard_dir"],
                                 help="String - Directory path for where the TensorBoard logs will be saved.")
    argument_parser.add_argument("--model_dir", type=str,
                                 default=config_parser["logging"]["model_dir"],
                                 help="String - Directory path for where the models will be saved.")
    argument_parser.add_argument("--output_dir", type=str,
                                 default=config_parser["logging"]["output_dir"],
                                 help="String - Directory path for where the testing outputs will be saved.")
    argument_parser.add_argument("--plot_dir", type=str,
                                 default=config_parser["logging"]["plot_dir"],
                                 help="String - Directory path for where the plots will be saved.")

    # Dataset Arguments
    argument_parser.add_argument("--dataset_dir", type=str,
                                 default=config_parser["dataset"]["dataset_dir"],
                                 help="String - Directory path for where the dataset is located.")
    argument_parser.add_argument("--image_x", type=int,
                                 default=int(config_parser["dataset"]["image_x"]),
                                 help="Integer - Width of the image that should be resized to.")
    argument_parser.add_argument("--image_y", type=int,
                                 default=int(config_parser["dataset"]["image_y"]),
                                 help="Integer - Height of the image that should be resized to.")
    argument_parser.add_argument("--augmentation", type=str_to_bool,
                                 default=config_parser["dataset"]["augmentation"].lower() == "true",
                                 help="Boolean - Should if the training data should be augmented.")
    argument_parser.add_argument("--val_split", type=float,
                                 default=float(config_parser["dataset"]["val_split"]),
                                 help="Floating Point Value - The percentage of data to be used for validation.")

    # Performance Arguments
    argument_parser.add_argument("--precision", type=int,
                                 default=int(config_parser["performance"]["precision"]),
                                 help="Integer - The level of precision used during training. 32 or 16.")
    argument_parser.add_argument("--gpu", type=str_to_bool,
                                 default=config_parser["performance"]["gpu"].lower() == "true",
                                 help="Boolean - Should training use GPU acceleration.")
    argument_parser.add_argument("--data_workers", type=int,
                                 default=int(config_parser["performance"]["data_workers"]),
                                 help="Integer - The number of data workers used to load data.")

    # Training Arguments
    argument_parser.add_argument("--efficient_net", type=int,
                                 default=int(config_parser["training"]["efficient_net"]),
                                 help="Integer - The compound coefficient used to scale the EfficientNet encoder.")
    argument_parser.add_argument("--starting_lr", type=float,
                                 default=float(config_parser["training"]["starting_lr"]),
                                 help="Floating Point Value - Starting learning rate. -1 for auto learning rate.")
    argument_parser.add_argument("--maximum_lr", type=float,
                                 default=float(config_parser["training"]["maximum_lr"]),
                                 help="Floating Point Value - The maximum learning rate. -1 for auto.")
    argument_parser.add_argument("--batch_size", type=int,
                                 default=int(config_parser["training"]["batch_size"]),
                                 help="Integer - The sizes of the batches during training.")

    # Early Stopping Arguments
    argument_parser.add_argument("--window", type=int,
                                 default=int(config_parser["early_stopping"]["window"]),
                                 help="Integer - The size of the window used on the validation losses.")
    argument_parser.add_argument("--stop_target", type=float,
                                 default=float(config_parser["early_stopping"]["stop_target"]),
                                 help="Floating Point Value - The target to stop training the model.")
    argument_parser.add_argument("--min_epochs", type=int,
                                 default=int(config_parser["early_stopping"]["min_epochs"]),
                                 help="Integer - The minimum number of epochs during training.")
    argument_parser.add_argument("--max_epochs", type=int,
                                 default=int(config_parser["early_stopping"]["max_epochs"]),
                                 help="Integer - The maximum number of epochs during training.")

    # Temperature
    argument_parser.add_argument("--temperature", type=float,
                                 default=float(config_parser["temperature"]["temperature"]),
                                 help="Floating Point Value - The temperature used to improve calibration.")

    # MC Dropout Arguments
    argument_parser.add_argument("--drop_rate", type=float,
                                 default=float(config_parser["mc_dropout"]["drop_rate"]),
                                 help="Floating Point Value - The drop rate applied to the model.")
    argument_parser.add_argument("--drop_iterations", type=int,
                                 default=int(config_parser["mc_dropout"]["drop_iterations"]),
                                 help="Integer - The number of forward passes for mc dropout.")

    # SelectiveNet Arguments
    argument_parser.add_argument("--alpha", type=float,
                                 default=float(config_parser["selective_net"]["alpha"]),
                                 help="")
    argument_parser.add_argument("--lamda", type=int,
                                 default=int(config_parser["selective_net"]["lamda"]),
                                 help="")
    argument_parser.add_argument("--target", type=float,
                                 default=float(config_parser["selective_net"]["target"]),
                                 help="")

    # Debug Arguments
    argument_parser.add_argument("--batches_per_epoch", type=int,
                                 default=int(config_parser["debug"]["batches_per_epoch"]),
                                 help="Integer - The number of batches should be run each epoch.")

    # Returns the argument parser.
    return argument_parser.parse_args()


def print_arguments(arguments):
    """
    Prints all arguments in a ArgumentParser Namespace.
    :param arguments:  ArgumentParser Namespace object containing arguments.
    """

    # Cycles through all the arguments within the ArgumentParser Namespace.
    for argument in vars(arguments):
        log(arguments, f"{argument: <24}: {getattr(arguments, argument)}")

    # Adds a blank line after printing arguments.
    log(arguments, "\n")
