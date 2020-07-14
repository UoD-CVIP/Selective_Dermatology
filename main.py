#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
This file is the main executable for Selective Dermatology.
This file loads the arguments, sets random seed, initialises and trains a model.
"""


# Own Modules Imports
from config import *


__author__ = ["Jacob Carse", "Stephen Hogg"]
__credits__ = ["Jacob Carse", "Stephen Hogg"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


if __name__ == "__main__":
    # Loads the arguments from a config file and the command line.
    description = "Selective Dermatology"
    arguments = load_arguments(description)

    # Displays the given arguments.
    print("Loaded Arguments:")
    print_arguments(arguments)
