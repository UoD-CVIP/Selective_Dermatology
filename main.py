#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
This file is the main executable for Selective Dermatology.
This file loads the arguments, sets random seed, initialises and trains a model.
"""


# Built-in/Generic Imports
import os

# Own Modules Imports
from utils import *
from config import *
from selective_net import *


__author__ = ["Jacob Carse", "Stephen Hogg", "Stephen McKenna"]
__copyright__ = "Copyright 2020, Selective Dermatology"
__credits__ = ["Jacob Carse", "Stephen Hogg", "Stephen McKenna"]
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

    # Sets the random seed if specified.
    if arguments["seed"] != -1:
        set_random_seed(arguments["seed"])
        print(f"\nSet Random Seed to {arguments['seed']}")

    # Loads the SelectiveNet Module.
    module = SelectiveNetModule(arguments)

    # Gets the PyTorch Lightning Trainer used to train and test the model.
    trainer = get_trainer(arguments)

    # Trains the model using the trainer.
    trainer.fit(module)

    # Loads the trained model weights
    filenames = os.listdir(arguments["experiment"])
    filenames.sort(reverse=True)
    module = module.load_from_checkpoint(os.path.join(arguments["experiment"], filenames[0]))

    # Tests the trained model with the trainer.
    trainer.test(module)
