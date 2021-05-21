# -*- coding: utf-8 -*-


"""
The file contains the following utility functions for the application:
    str_to_bool - Function to convert an input string to a boolean value.
    log - Function to print and/or log messages to the console or logging file.
    set_random_seed - Function used to set the random seed for all libraries used to generate random numbers.
"""


# Built-in/Generic Imports
import os
import random
from argparse import ArgumentTypeError

# Library Imports
import torch
import numpy as np


__author__    = ["Jacob Carse"]
__copyright__ = "Copyright 2020, Selective Dermatology"
__credits__   = ["Jacob Carse", "Stephen Hogg", "Stephen McKenna"]
__license__   = "MIT"
__version__   = "3.0.0"
__maintainer  = "Jacob Carse"
__email__     = "j.carse@dundee.ac.uk"
__status__    = "Development"


def str_to_bool(argument):
    """
    Function to convert a string to a boolean value.
    :param argument: String to be converted.
    :return: Boolean value.
    """

    # Checks if the argument is already a boolean value.
    if isinstance(argument, bool): return argument

    # Returns boolean depending on the input string.
    if argument.lower() == "true" or argument.lower() == 't':
        return True
    elif argument.lower() == "false" or argument.lower() == 'f':
        return False

    # Returns an error if the value is not converted to a boolean value.
    return ArgumentTypeError(f"Boolean value expected. Got \"{argument}\".")


def log(arguments, message):
    """
    Logging function that will both print and log an input message.
    :param arguments: ArgumentParser object containing 'log_dir' and 'experiment'.
    :param message: String containing the message to be printed and/or logged.
    """

    # Prints the message to console if verbose is set to True.
    if arguments.verbose:
        print(message)

    # Logs the message within a specific log file is defined.
    if arguments.log_dir != '':
        # Creates the directory for the log files.
        os.makedirs(arguments.log_dir, exist_ok=True)

        # Logs the message to the log file.
        print(message, file=open(os.path.join(arguments.log_dir, f"{arguments.experiment}_log.txt"), 'a'))


def set_random_seed(seed):
    """
    Sets the random sed or all libraries that are used to generate random numbers.
    :param seed: Integer for the seed that will be used.
    """

    # Sets the seed for the inbuilt Python functions.
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Sets the seed for the NumPy library.
    np.random.seed(seed)

    # Sets the seed for the PyTorch library.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_device(arguments):
    """
    Sets the device that will be used to training and testing.
    :param arguments: A ArgumentParser Namespace containing "gpu".
    :return: A PyTorch device.
    """

    # Checks if the GPU is available to be used and sets the .
    if arguments.gpu and torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.device_count() - 1}")

    # Sets the device to CPU.
    else:
        return torch.device("cpu")
