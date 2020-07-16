# -*- coding: utf-8 -*-

"""
This file defines the PyTorch Lightning Module used to train and test the SelectiveNet model.
    SelectiveNetModule - Class for SelectiveNet Module.
"""


# Built-in/Generic Imports

# Library Imports
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, lr_scheduler

# Own Modules
from model import SelectiveNet
from dataset import get_datasets


__author__ = ["Jacob Carse", "Stephen Hogg", "Stephen McKenna"]
__copyright__ = "Copyright 2020, Selective Dermatology"
__credits__ = ["Jacob Carse", "Stephen Hogg", "Stephen McKenna"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


class SelectiveNetModule(pl.LightningModule):
    """
    Class for handling the training, validation and testing of the SelectiveNet Model.
        init - Initialiser for the Module, used to initialise the Model and other parameters.
        configure_optimizers - Method for getting the optimiser and learning rate scheduler to train the model.
        loss - Calculates the selective loss for the SelectiveNet model.
        forward - Forward propagation for the SelectiveNet model.
        training_step - Method for performing a training step.
        validation_step - Method for performing a validation step.
        validation_epoch_end - Method for summarises the validation steps.
        test_step -
        test_epoch_end -
        prepare_data - Method for preparing the Dataset objects.
        train_dataloader - Gets the data loader for the training data.
        val_dataloader - Gets the data loader for the validation data.
        test_dataloader - Gets the data loader for the testing data.
    """

    def __init__(self, hparams):
        """
        Initiliser for the SelectiveNet module.
        :param hparams: Dictionary of arguments.
        """

        # Calls the PyTorch Lightning Module initiliser.
        super(SelectiveNetModule, self).__init__()

        # Stores the hyper parameters for the module in the object.
        self.hparams = hparams

        # Defines the lists used to collect the testing predictions.
        self.test_predictions, self.test_labels = [], []
        self.test_csv = []

        # Initialises the SelectiveNet model.
        self.model = SelectiveNet(hparams)

    def configure_optimizers(self):
        """
        Configure the optimiser and learning rate scheduler used during training.
        :return: PyTorch optimiser and PyTorch learning rate scheduler.
        """

        # Defines the optimiser used for the training the model.
        optimiser = SGD(params=self.model.parameters(), lr=self.hparams["starting_lr"])

        # Defines the learning rate scheduler used for altering the learning rate during training.
        max_lr = self.hparams["starting_lr"] * 100 if self.hparams["max_lr"] == -1 else self.hparams["max_lr"]
        scheduler = lr_scheduler.CyclicLR(optimiser, base_lr=self.hparams["starting_lr"], max_lr=max_lr)

        # Returns the optimiser and learning rate scheduler.
        return [optimiser], [scheduler]

    def loss(self, labels, predictions, selections, auxiliary):
        """
        Calculates the loss and the empirical selective risk and coverage.
        :param labels: The labels for the batch of images.
        :param predictions: The output of the predictive head of the model.
        :param selections: The output of the selective head of the model.
        :param auxiliary: The output of the auxiliary head of the model.
        :return: PyTorch Tensors containing loss, selective risk and coverage.
        """

        # Gets the empirical coverage.
        emp_coverage = selections.mean()

        # Gets the selective risk by combining the predictive outputs and the selective outputs.
        selective_log_prob = - 1.0 * F.log_softmax(predictions, 1) * selections
        selective_risk = selective_log_prob.gather(1, labels.unsqueeze(1))
        selective_risk = selective_risk.mean() / emp_coverage

        # Gets the loss for the auxiliary output.
        auxiliary_log_prob = - 1.0 * F.log_softmax(auxiliary, 1)
        auxiliary_loss = auxiliary_log_prob.gather(1, labels.unsqueeze(1))
        auxiliary_loss = auxiliary_loss.mean()

        # Gets the loss using the selective risk, coverage and auxiliary loss.
        loss = self.hparams["alpha"] * (selective_risk + self.hparams["lamda"] *
                                        max(self.hparams["coverage"] - emp_coverage, 0) ** 2) +\
                                        (1 - self.hparams["alpha"]) * auxiliary_loss

        # Returns the loss, selective risk and coverage.
        return loss, selective_risk, emp_coverage

    def forward(self, x, drop_rate=None):
        """
        Forward propagation for the SelectiveNet model.
        :param x: Image batch.
        :param drop_rate: The drop rate for dropout in the model.
        :return: Model predictions, selective scores and auxiliary output.
        """

        return self.model(x, drop_rate)

    def training_step(self, batch, *args, **kwargs):
        """
        Performs a training step for the SelectiveNet model.
        :param batch: The input batch of images and labels.
        :param args: Input arguments.
        :param kwargs: Input keyword arguments.
        :return: PyTorch Lightning output dictionary.
        """

        # Splits the images and labels from the batch.
        images, labels = batch

        # Performs forward propagation with the SelectiveNet model.
        predictions, selective_scores, auxiliary = self.forward(images)

        # Gets the loss, selective risk and coverage of the model outputs.
        loss, selective_risk, coverage = self.loss(labels, predictions, selective_scores, auxiliary)

        # Calculates the accuracy of the image.
        acc = torch.sum(torch.argmax(predictions, 1) == labels.data, dtype=torch.float) / labels.shape[0]

        # Returns the PyTorch Lightning dictionary output.
        return {
            "loss": loss,
            "progress_bar": {"train_loss": loss, "train_acc": acc, "sel_risk": selective_risk, "coverage": coverage},
            "log": {"train_loss": loss, "train_acc": acc, "sel_risk": selective_risk, "coverage": coverage}
        }

    def validation_step(self, batch, *args, **kwargs):
        """
        Performs a validation step for the SelectiveNet model.
        :param batch: The input batch of images and labels.
        :param args: Input arguments.
        :param kwargs: Input keyword arguments.
        :return: Dictionary of validation outputs.
        """

        # Splits the images and labels from the batch.
        images, labels = batch

        # Performs forward propagation with the SelectiveNet model.
        predictions, selective_scores, auxiliary = self.forward(images)

        # Gets the loss, selective risk and coverage of the model outputs.
        loss, selective_risk, coverage = self.loss(labels, predictions, selective_scores, auxiliary)

        # Calculates the accuracy of the image.
        acc = torch.sum(torch.argmax(predictions, 1) == labels.data, dtype=torch.float) / labels.shape[0]

        # Returns the dictionary of validation metrics for validation step.
        return {
            "val_loss": loss,
            "val_sel_risk": selective_risk,
            "val_coverage": coverage,
            "val_acc": acc
        }

    def validation_epoch_end(self, outputs):
        """
        Summarises the validation epoch by averaging the validation steps.
        :param outputs: The outputs of all the validation steps.
        :return: PyTorch Lighting output dictionary.
        """

        # Defines the variables for each validation metric.
        val_loss_mean, val_sel_risk_mean, val_coverage_mean, val_acc_mean = 0, 0, 0, 0

        # Sums all the outputs of the validation steps.
        for output in outputs:
            val_loss_mean += output["val_loss"]
            val_sel_risk_mean += output["val_sel_risk"]
            val_coverage_mean += output["val_coverage"]
            val_acc_mean += output["val_acc"]

        # Divides the outputs by the number of validation steps.
        val_loss_mean /= len(outputs)
        val_sel_risk_mean /= len(outputs)
        val_coverage_mean /= len(outputs)
        val_acc_mean /= len(outputs)

        # PyTorch Lightning dictionary output.
        return {
            "progress_bar": {"val_loss": val_loss_mean, "val_sel_risk": val_sel_risk_mean,
                             "val_coverage": val_coverage_mean, "val_acc": val_acc_mean},
            "log": {"val_loss": val_loss_mean, "val_sel_risk": val_sel_risk_mean,
                    "val_coverage": val_coverage_mean, "val_acc": val_acc_mean}
        }

    def test_step(self, batch, *args, **kwargs):
        """
        Performs a testing step for the SelectiveNet model.
        :param batch: The input batch of images and labels.
        :param args: Input arguments.
        :param kwargs: Input keyword arguments.
        :return: Dictionary of testing outputs.
        """

        pass

    def test_epoch_end(self, outputs):
        pass

    def prepare_data(self):
        """
        Loads the dataset objects for training, validation and testing.
        """

        self.train_data, self.val_data, self.test_data = get_datasets(self.hparams)

    def train_dataloader(self):
        """
        Gets the data loader for the training data.
        :return: PyTorch DataLoader for the training data.
        """

        return DataLoader(self.train_data, batch_size=self.hparams["batch_size"], shuffle=True,
                          num_workers=self.hparams["data_workers"], pin_memory=False)

    def val_dataloader(self):
        """
        Gets the data loader for the validation data.
        :return: PyTorch DataLoader for the validation data.
        """

        return DataLoader(self.val_data, batch_size=self.hparams["batch_size"], shuffle=False,
                          num_workers=self.hparams["data_workers"], pin_memory=False)

    def test_dataloader(self):
        """
        Gets the data loader for the testing data.
        :return: PyTorch DataLoader for the testing data.
        """

        return DataLoader(self.test_data, batch_size=self.hparams["batch_size"], shuffle=False,
                          num_workers=self.hparams["data_workers"], pin_memory=False)
