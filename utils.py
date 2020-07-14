# -*- coding: utf-8 -*-

"""
This file contains the following utility functions:
    set_random_seed - Function to set random seed across the application.
"""


# Library Imports
import pytorch_lightning as pl


__author__ = ["Jacob Carse", "Stephen Hogg"]
__credits__ = ["Jacob Carse", "Stephen Hogg"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


def set_random_seed(seed):
    """
    Sets the random seed for all libraries that are used to generate random numbers.
    :param seed: Integer for the seed to be set as the random seed.
    """

    pl.seed_everything(seed)
