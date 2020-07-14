# -*- coding: utf-8 -*-

"""
This file contains the following utility functions:
    str_to_bool - Function to convert a string to a boolean value.
    set_random_seed - Function to set random seed across the application.
"""


# Built-in/Generic Imports
from argparse import ArgumentTypeError

# Library Imports
import pytorch_lightning as pl


__author__ = ["Jacob Carse", "Stephen Hogg"]
__credits__ = ["Jacob Carse", "Stephen Hogg"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


def str_to_bool(argument):
    """
    Function to convert a string to a boolean.
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

    # Returns an error if the value is not converted to boolean.
    raise ArgumentTypeError("Boolean value expected.")


def set_random_seed(seed):
    """
    Sets the random seed for all libraries that are used to generate random numbers.
    :param seed: Integer for the seed to be set as the random seed.
    """

    pl.seed_everything(seed)
