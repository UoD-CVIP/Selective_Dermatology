#! /usr/bin/python3
# -*- coding: utf-8 -*-


"""

"""


# Own Modules Imports
from utils import *
from config import *
from train_sn import *
from train_cnn import *


__author__    = ["Jacob Carse"]
__copyright__ = "Copyright 2020, Selective Dermatology"
__credits__   = ["Jacob Carse", "Stephen Hogg", "Stephen McKenna"]
__license__   = "MIT"
__version__   = "3.0.0"
__maintainer  = "Jacob Carse"
__email__     = "j.carse@dundee.ac.uk"
__status__    = "Development"


if __name__ == "__main__":
    # Loads the arguments from configurations file and command line.
    description = "Selective Dermatology Experiments"
    arguments = load_configurations(description)

    # Displays the loaded arguments.
    log(arguments, "Loaded Arguments:")
    print_arguments(arguments)

    # Sets the random seed if specified.
    if arguments.seed != -1:
        set_random_seed(arguments.seed)
        log(arguments, f"Set Random Seed to {arguments.seed}\n")

    # Sets the default device to be used.
    device = get_device(arguments)
    log(arguments, f"Device set to {device}\n")

    # Runs the selected task.
    if arguments.task == "cnn":

        # Trains and tests the CNN model.
        arguments.temperature = train_cnn(arguments, device)
        test_cnn(arguments, device)

    elif arguments.task == "train_cnn":

        # Trains the CNN model.
        train_cnn(arguments, device)

    elif arguments.task == "test_cnn":

        # Tests the CNN model.
        test_cnn(arguments, device)

    elif arguments.task == "sn":

        # Trains and tests the SelectiveNet model.
        arguments.temperature = train_sn(arguments, device)
        test_sn(arguments, device)

    elif arguments.task == "train_sn":

        # Trains the SelectiveNet model.
        train_sn(arguments, device)

    elif arguments.task == "test_sn":

        # Tests the SelectiveNet model.
        test_sn(arguments, device)

    else:

        # Asks the user to enter a valid task.
        log(arguments, "Enter a valid task \'cnn\' or \'sn\'.")
