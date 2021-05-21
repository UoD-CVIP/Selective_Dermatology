# -*- coding: utf-8 -*-


"""
The file for the definition of the SelectiveNet and Classifier models.
    Classifier - Class for a EfficientNet Classifier Model.
    SelectiveNet - Class for the SelectiveNet Model using a EfficientNet encoder.
"""


# Built-in/Generic Imports
import os

# Library Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet


__author__    = ["Jacob Carse"]
__copyright__ = "Copyright 2020, Selective Dermatology"
__credits__   = ["Jacob Carse", "Stephen Hogg", "Stephen McKenna"]
__license__   = "MIT"
__version__   = "3.0.0"
__maintainer  = "Jacob Carse"
__email__     = "j.carse@dundee.ac.uk"
__status__    = "Development"


class Classifier(nn.Module):
    """
    Class for the Classifier model that uses an EfficientNet encoder with optional drop out.
    """

    def __init__(self, b=0, drop_rate=0.5, pretrained=True):
        """
        Initiliser for the model that initialises the models layers.
        :param b: The compound coefficient of the EfficientNet model to be loaded.
        :param drop_rate: The drop rate for the optional dropout layers.
        :param pretrained: Boolean if pretrained weights should be loaded.
        """

        # Calls the super for the nn.Module.
        super(Classifier, self).__init__()

        # Sets the drop rate for the dropout layers.
        self.drop_rate = drop_rate

        # Loads the EfficientNet encoder.
        if pretrained:
            self.encoder = EfficientNet.from_pretrained(f"efficientnet-b{str(b)}")
        else:
            self.encoder = EfficientNet.from_name(f"efficientnet-b{str(b)}")

        self.encoder_pool = nn.AdaptiveAvgPool2d(1)

        # Defines a hidden layer.
        self.hidden = nn.Linear(2560, 512)

        # Defines the output layer of the neural network.
        self.classifier = nn.Linear(512, 2)

    def forward(self, x, dropout=False):
        """
        Performs forward propagation with the Classifier.
        :param x: Input image batch.
        :param dropout: Boolean if dropout should be applied.
        :return: A PyTorch Tensor of logits.
        """

        # Performs forward propagation with the encoder.
        x = self.encoder.extract_features(x)
        x = self.encoder_pool(x)
        x = x.view(x.shape[0], -1)

        # Applies dropout to the model is selected.
        if dropout:
            x = F.dropout(x, self.drop_rate)

        # Performs forward propagation with the hidden layer.
        x = self.hidden(x)

        # Applies dropout to the model is selected.
        if dropout:
            x = F.dropout(x, self.drop_rate)

        # Gets the output logits from the output layer.
        return self.classifier(x)

    def save_model(self, path, name, epoch="best"):
        """
        Method for saving the model.
        :param path: Directory path to save the model.
        :param name: The name of the experiment to be saved.
        :param epoch: Integer for the current epoch to be included in the save name.
        """

        # Checks if the save directory exists and if not creates it.
        os.makedirs(path, exist_ok=True)

        # Saves the model to the save directory.
        torch.save(self.state_dict(), os.path.join(path, f"{name}_cnn_{str(epoch)}.pt"))


class SelectiveNet(nn.Module):
    """
    Class for the SelectiveNet model that uses a EfficientNet encoder and optional drop out.
    """

    def __init__(self, b=0, drop_rate=0.5, pretrained=True):
        """
        Initiliser for the SelectiveNet model that Initialises the models layers.
        :param b: The compound coefficient of the EfficientNet model to be loaded.
        :param drop_rate: The drop rate for the optional dropout layers.
        :param pretrained: Boolean if pretrained weights should be loaded.
        """

        # Calls the super for the nn.Module.
        super(SelectiveNet, self).__init__()

        # Loads the EfficientNet encoder.
        if pretrained:
            self.encoder = EfficientNet.from_pretrained(f"efficientnet-b{str(b)}")
        else:
            self.encoder = EfficientNet.from_name(f"efficientnet-b{str(b)}")
        self.encoder_pool = nn.AdaptiveAvgPool2d(1)

        # Defines a hidden layer.
        self.hidden = nn.Linear(2560, 512)

        # Initialises the classifier for generating predictions.
        self.classifier = nn.Linear(512, 2)

        # Initialises the selective branch for generating selection scores.
        self.selective_hidden = nn.Linear(512, 512)
        self.selective_batch_norm = nn.BatchNorm1d(512)
        self.selective_regression = nn.Linear(512, 1)

        # Initialises the auxiliary output used by the model during training.
        self.auxiliary_output = nn.Linear(512, 2)

        # Stores the dropout rate in the object.
        self.drop_rate = drop_rate

    def forward(self, x, dropout=False):
        """
        Performs forward propagation with SelectiveNet.
        :param x: The input image batch.
        :param dropout: Boolean if dropout should be applied.
        :return: Predictions, Selective score and auxiliary output.
        """

        # Performs forward propagation with the EfficientNet encoder.
        x = self.encoder.extract_features(x)
        x = self.encoder_pool(x)
        x = x.view(x.shape[0], -1)

        # Applies dropout if selected.
        if dropout:
            x = F.dropout(x, self.drop_rate)

        # Uses the hidden layer.
        x = self.hidden(x)

        # Applies dropout if selected.
        if dropout:
            x = F.dropout(x, self.drop_rate)

        # Gets the predictive output of the model.
        x1 = self.classifier(x)

        # Gets the selective output of the model.
        x2 = F.relu(self.selective_hidden(x))
        x2 = self.selective_batch_norm(x2)
        x2 = torch.sigmoid(self.selective_regression(x2))

        # Gets the auxiliary output of the model.
        x3 = self.auxiliary_output(x)

        # Returns the outputs of the model.
        return x1, x2, x3

    def save_model(self, path, name, epoch="best"):
        """
        Method for saving the model.
        :param path: Directory path to save the model.
        :param name: The name of the experiment to be saved.
        :param epoch: Integer for the current epoch to be included in the save name.
        """

        # Checks if the save directory exists and if not creates it.
        os.makedirs(path, exist_ok=True)

        # Saves the model to the save directory.
        torch.save(self.state_dict(), os.path.join(path, f"{name}_sn_{str(epoch)}.pt"))
