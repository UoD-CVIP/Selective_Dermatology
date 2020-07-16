# -*- coding: utf-8 -*-

"""
This file contains the class defining the SelectiveNet model:
    SelectiveNet - Class for the SelectiveNet Model.
"""


# Library Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet


__author__ = ["Jacob Carse", "Stephen Hogg", "Stephen McKenna"]
__copyright__ = "Copyright 2020, Selective Dermatology"
__credits__ = ["Jacob Carse", "Stephen Hogg", "Stephen McKenna"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


class SelectiveNet(nn.Module):
    """
    Class for the SelectiveNet model with a EfficientNet encoder, containing the methods:
        init - Initialiser for the model to initialise the model.
        forward - Forward propagation for the SelectiveNet model.
    """

    def __init__(self, arguments):
        """
        Initialiser for the model that initialises the models layers.
        :param arguments: Dictionary of arguments.
        :param n: Integer for the type of EfficientNet to be loaded.
        """

        # Calls the super for the nn.Module.
        super(SelectiveNet, self).__init__()

        # Loads the EfficientNet encoder.
        self.encoder = EfficientNet.from_pretrained(f"efficientnet-b{str(arguments['efficient_net'])}")
        self.encoder_pool = nn.AdaptiveAvgPool2d(1)

        # Gets the dimension of the encoder output.
        with torch.no_grad():
            temp_input = torch.zeros(1, 3, arguments["image_x"], arguments["image_y"])
            encoder_size = self.encoder.extract_features(temp_input).shape[2]

        # Initialises the classification head for generating predictions.
        self.classifier = nn.Linear(encoder_size, 2)

        # Initialises the selective head for generating selective scores.
        self.selective_hidden = nn.Linear(encoder_size, 512)
        self.selective_batch_norm = nn.BatchNorm1d(512)
        self.selective_regression = nn.Linear(512, 1)

        # Initialises the auxiliary head used during training.
        self.auxiliary_output = nn.Linear(encoder_size, 2)

    def forward(self, x, drop_rate=None):
        """
        Performs forward propagation with the SelectiveNet.
        :param x: The input image batch.
        :param drop_rate: Floating point value for the dropout rate to be applied to the encoder output.
        :return: Predictive, Selective and Auxiliary outputs.
        """

        # Performs the forward propagation with the encoder.
        x = self.encoder.extract_features(x)
        x = self.encoder_pool(x)
        x = x.view(x.shape[0], -1)

        # Applies dropout to the encoder outputs if drop rate is None.
        if drop_rate is not None:
            x = F.dropout(x, drop_rate)

        # Gets the predictive output of the model.
        x1 = self.classifier(x)

        # Gets the selective output of the model.
        x2 = F.relu((self.selective_hidden(x)))
        x2 = self.selective_batch_norm(x2)
        x2 = torch.sigmoid(self.selective_regression(x2))

        # Gets the auxiliary output of the model.
        x3 = self.auxiliary_output(x)

        # Returns the outputs of the model.
        return x1, x2, x3
