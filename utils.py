# -*- coding: utf-8 -*-

"""
This file contains the following utility functions:
    str_to_bool - Function to convert a string to a boolean value.
    set_random_seed - Function to set random seed across the application.
    get_trainer - Function to get the PyTorch Lightning Trainer used to train and test the model.
"""


# Built-in/Generic Imports
from argparse import ArgumentTypeError

# Library Imports
import torch
import pytorch_lightning as pl


__author__ = ["Jacob Carse", "Stephen Hogg", "Stephen McKenna"]
__copyright__ = "Copyright 2020, Selective Dermatology"
__credits__ = ["Jacob Carse", "Stephen Hogg", "Stephen McKenna"]
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


def get_trainer(arguments):
    """
    Function to get a PyTorch Lightning Trainer used for training and testing of a model.
    :param arguments: Dictionary of arguments.
    :return: A PyTorch Lightning Trainer.
    """

    # Gets the number of GPUs that should be used for the trainer.
    if arguments["num_gpus"] == 0:
        num_gpus = None
    else:
        if torch.cuda.device_count() < arguments["num_gpus"]:
            num_gpus = torch.cuda.device_count()
        else:
            num_gpus = arguments["num_gpus"]

    # Determines if a distributed backend should be used.
    if arguments["distributed_backend"].lower() == "none" or num_gpus == 0:
        distributed_backend = None
    else:
        distributed_backend = arguments["distributed_backend"].lower()

    # Gets the level of precision that should be used.
    if arguments["num_gpus"] == 0:
        precision = 32
    else:
        precision = 16 if arguments["precision"] == -1 else None

    # Determines if a batch size finder should be used to determine the maximum batch size.
    batch_size = "binsearch" if arguments["batch_size"] == -1 else None

    # Determins if a learning rate finder should be used to determine the optimal starting learning rate.
    lr = "starting_lr" if arguments["starting_lr"] == -1 else False

    # Sets up the TensorBoard Logger used to track training metrics.
    tb_logger = pl.loggers.TensorBoardLogger(save_dir=arguments["tensorboard_dir"], name=arguments["experiment"])

    # Sets up the EarlyStopping callback.
    early_stopping = pl.callbacks.EarlyStopping("val_loss", arguments["min_delta"], arguments["patience"], verbose=True)

    # Sets up the checkpoint callback.
    checkpoint_callback = pl.callbacks.model_checkpoint.ModelCheckpoint("Models/", monitor="val_loss", save_top_k=1)

    # Creates the PyTorch Lightning Trainer with values from arguments.
    return pl.Trainer(distributed_backend=distributed_backend, gpus=num_gpus, precision=precision,
                      max_epochs=arguments["max_epochs"], min_epochs=arguments["min_epochs"], logger=tb_logger,
                      early_stop_callback=early_stopping, fast_dev_run=arguments["fast_dev_run"],
                      num_sanity_val_steps=0, deterministic=True, auto_scale_batch_size=batch_size,
                      auto_lr_find=lr, checkpoint_callback=checkpoint_callback)
