# -*- coding: utf-8 -*-


"""
This file contains implementations of the functions used to train a CNN model:
    train_cnn - Function used to facilitate the training of the Convolutinal Neural Network model.
    test_cnn - Function used to facilitate the testing of the Convolutinal Neural Network model.
"""


# Built-in/Generic Imports
import os
import time

# Library Imports
import torch
import numpy as np
import pandas as pd
from torch.cuda import amp
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD, LBFGS, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

# Own Modules
from utils import log
from model import Classifier
from dataset import get_datasets


__author__    = ["Jacob Carse"]
__copyright__ = "Copyright 2020, Selective Dermatology"
__credits__   = ["Jacob Carse", "Stephen Hogg", "Stephen McKenna"]
__license__   = "MIT"
__version__   = "3.0.0"
__maintainer  = "Jacob Carse"
__email__     = "j.carse@dundee.ac.uk"
__status__    = "Development"


def train_cnn(arguments, device):
    """
    Function for training of the Convolutional neural network.
    :param arguments: ArgumentParser Namespace object with arguments used for training.
    :param device: PyTorch device that will be used for training.
    :return: Lists of training and validation losses and an integer for the best performing epoch.
    """

    # Loads a TensorBoard Summary Writer.
    if arguments.tensorboard_dir != "":
        writer = SummaryWriter(os.path.join(arguments.tensorboard_dir, arguments.task, arguments.experiment))

    # Loads the training and validation data.
    train_data, val_data, _ = get_datasets(arguments)

    # Creates the training data loader using the dataset objects.
    training_data_loader = DataLoader(train_data, batch_size=arguments.batch_size,
                                      shuffle=True, num_workers=arguments.data_workers,
                                      pin_memory=False, drop_last=False)

    # Creates the validation data loader using the dataset objects.
    validation_data_loader = DataLoader(val_data, batch_size=arguments.batch_size,
                                        shuffle=False, num_workers=arguments.data_workers,
                                        pin_memory=False, drop_last=False)

    log(arguments, "Loaded Datasets\n")

    # Initialises the classifier model.
    classifier = Classifier(arguments.efficient_net)

    # Sets the classifier to training mode.
    classifier.train()

    # Moves the classifier to the selected device.
    classifier.to(device)

    # Initialises the optimiser used to optimise the parameters of the model.
    optimiser = SGD(params=classifier.parameters(), lr=arguments.starting_lr)

    # Initialises the learning rate scheduler to adjust the learning rate during training.
    scheduler = lr_scheduler.CyclicLR(optimiser, base_lr=arguments.starting_lr, max_lr=arguments.maximum_lr)

    # Initialises the gradient scaler used for 16 but precision.
    if arguments.precision == 16 and device != torch.device("cpu"):
        scaler = amp.GradScaler()

    log(arguments, "Models Initialised")

    # Declares the main logging variables for the training.
    start_time = time.time()
    losses, validation_losses, temperatures = [], [], []
    best_loss, best_epoch, total_batches = 1e10, 0, 0

    log(arguments, "Training Timer Started\n")

    # The beginning of the main training loop.
    for epoch in range(1, arguments.max_epochs + 1):

        # Declares the logging variables for the epoch.
        epoch_acc, epoch_loss, epoch_risk, epoch_coverage,num_batches = 0, 0, 0, 0, 0

        # Loops through the training data batches.
        for images, labels in training_data_loader:

            # Moves the images and labels to the selected device.
            images = images.to(device)
            labels = labels.to(device)

            # Resets the gradients in the model.
            optimiser.zero_grad()

            # Perform training with 16 bit precision.
            if arguments.precision == 16 and device != torch.device("cpu"):
                with amp.autocast():

                    # Performs forward propagation with the model.
                    logits = classifier(images, dropout=True)

                    # Calculates the loss.
                    loss = F.cross_entropy(logits, labels)

                # Using the gradient scaler performs backward propagation.
                scaler.scale(loss).backward()

                # Update the weights of the model using the optimiser.
                scaler.step(optimiser)

                # Updates the scale factor of the gradient scaler.
                scaler.update()

            # Performs training with 32 bit precision.
            else:
                # Performs forward propagation with the model.
                logits = classifier(images, dropout=True)

                # Calculates the loss.
                loss = F.cross_entropy(logits, labels)

                # Performs backward propagation.
                loss.backward()

                # Update the weights of the model using the optimiser.
                optimiser.step()

            # Updates the learning rate scheduler.
            scheduler.step()

            # Calculates the accuracy of the batch.
            batch_accuracy = (logits.max(dim=1)[1] == labels).sum().double() / labels.shape[0]

            # Calculates the selection scores for the validation predictions.
            selections = torch.max(F.softmax(logits), 1)[0]

            # Calculates the coverage for the batch.
            batch_coverage = selections.mean()

            # Calculates the log probability for the predictions and selections.
            log_prob = -1. * F.log_softmax(logits, 1) * selections.view([labels.shape[0], 1])

            # Calculates the selective risk for the batch using the selections and predictions.
            batch_risk = log_prob.gather(1, labels.unsqueeze(1)).mean() / batch_coverage

            # Adds the number of batches, loss and accuracy to epoch sum.
            num_batches += 1
            epoch_loss += loss.item()
            epoch_acc += batch_accuracy
            epoch_coverage += batch_coverage
            epoch_risk += batch_risk

            # Writes the batch loss and accuracy to TensorBoard logger.
            if arguments.tensorboard_dir != "":
                writer.add_scalar("Loss/batch", loss.item(), num_batches + total_batches)
                writer.add_scalar("Accuracy/batch", batch_accuracy, num_batches + total_batches)

            # Logs the details of the epoch progress.
            if num_batches % arguments.log_interval == 0:
                log(arguments, "Time: {}s\tTrain Epoch: {} [{}/{}] ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.6f}".format(
                    str(int(time.time() - start_time)).rjust(6, '0'), str(epoch).rjust(2, '0'),
                    str(num_batches * arguments.batch_size).rjust(len(str(len(train_data))), '0'),
                    len(train_data), 100. * num_batches / (len(train_data) / arguments.batch_size),
                    epoch_loss / num_batches, epoch_acc / num_batches))

            # If the number of batches have been reached end epoch.
            if num_batches == arguments.batches_per_epoch:
                break

        # Updates the total number of batches (used for logging).
        total_batches += num_batches

        # Writes epoch loss and accuracy to TensorBoard.
        if arguments.tensorboard_dir != "":
            writer.add_scalar("Loss/train", epoch_loss / num_batches, epoch)
            writer.add_scalar("Accuracy/train", epoch_acc / num_batches, epoch)
            writer.add_scalar("Coverage/train", epoch_coverage / num_batches, epoch)
            writer.add_scalar("Selective Risk/train", epoch_risk / num_batches, epoch)

        # Declares the logging variables for validation.
        validation_acc, validation_loss, validation_risk, validation_coverage, validation_batches = 0, 0, 0, 0, 0

        logit_list, label_list = [], []

        temperature = torch.nn.Parameter(torch.ones(1, device=device))
        temp_optimiser = LBFGS([temperature], lr=0.01, max_iter=1000, line_search_fn="strong_wolfe")

        # Performs the validation epoch with no gradient calculations.
        with torch.no_grad():

            # Loops through the training data batches.
            for images, labels in validation_data_loader:

                # Moves the images and labels to the selected device.
                images = images.to(device)
                labels = labels.to(device)

                # Performs forward propagation using 16 bit precision.
                if arguments.precision == 16 and device != torch.device("cpu"):
                    with amp.autocast():

                        # Performs forward propagation with the model.
                        logits = classifier(images, dropout=False)

                        # Calculates the loss.
                        loss = F.cross_entropy(logits, labels)

                # Performs forward propagation using 32 bit precision.
                else:

                    # Performs forward propagation with the model.
                    logits = classifier(images, dropout=True)

                    # Calculates the loss.
                    loss = F.cross_entropy(logits, labels)

                logit_list.append(logits)
                label_list.append(labels)

                # Calculates the accuracy of the batch.
                batch_accuracy = (logits.max(dim=1)[1] == labels).sum().double() / labels.shape[0]

                # Calculates the selection scores for the validation predictions.
                selections = torch.max(F.softmax(logits), 1)[0]

                # Calculates the coverage for the batch.
                batch_coverage = selections.mean()

                # Calculates the log probability for the predictions and selections.
                log_prob = -1. * F.log_softmax(logits, 1) * selections.view([labels.shape[0], 1])

                # Calculates the selective risk for the batch using the selections and predictions.
                batch_risk = log_prob.gather(1, labels.unsqueeze(1)).mean() / batch_coverage

                # Adds the number of batches, loss and accuracy to validation sum.
                validation_batches += 1
                validation_loss += loss.item()
                validation_acc += batch_accuracy
                validation_coverage += batch_coverage
                validation_risk += batch_risk

                # If the number of batches have been reached end validation.
                if validation_batches == arguments.batches_per_epoch:
                    break

        logit_list = torch.cat(logit_list).to(device)
        label_list = torch.cat(label_list).to(device)

        def _eval():
            temp_loss = F.cross_entropy(torch.div(logit_list, temperature), label_list)
            temp_loss.backward()
            return temp_loss

        temp_optimiser.step(_eval)
        temperatures.append(temperature.item())

        # Writes validation loss and accuracy to TensorBoard.
        if arguments.tensorboard_dir != "":
            writer.add_scalar("Loss/validation", validation_loss / validation_batches, epoch)
            writer.add_scalar("Accuracy/validation", validation_acc / validation_batches, epoch)
            writer.add_scalar("Coverage/validation", validation_coverage / validation_batches, epoch)
            writer.add_scalar("Selective Risk/validation", validation_risk / validation_batches, epoch)

        # Adds the training and validation losses to their respective lists.
        losses.append(epoch_loss / num_batches)
        validation_losses.append(validation_loss / validation_batches)

        # Logs the details of the training epoch.
        log(arguments, "\nEpoch: {}\Training Loss: {:.6f}\tTraining Accuracy: {:.6f}\t"
                       "Training Coverage: {:.6f}\tTraining Selective Risk: {:.6f}\n"
                       "Validation Loss: {:.6f}\tValidation Accuracy: {:.6f}\t"
                       "Validation Coverage: {:.6f}\tValidation Selective Risk: {:.6f}\n".
            format(epoch, losses[-1], epoch_acc / num_batches, epoch_coverage / num_batches, epoch_risk / num_batches,
                   validation_losses[-1], validation_acc / validation_batches,
                   validation_coverage / validation_batches, validation_risk / validation_batches))

        # If the current epoch has the best validation loss then save the model with the prefix best.
        if validation_losses[-1] < best_loss:
            best_loss = validation_losses[-1]
            best_epoch = epoch
            classifier.save_model(arguments.model_dir, arguments.experiment)

        # Saves the model with the current epoch as the prefix.
        classifier.save_model(arguments.model_dir, arguments.experiment, str(epoch))

        # Checks if the training has performed the minimum number of epochs.
        if epoch >= arguments.min_epochs:

            # Calculates the generalised validation loss.
            g_loss = 100 *((validation_losses[-1] / min(validation_losses[:-1])) - 1)

            # Calculates the training progress using a window over the training losses.
            t_progress = 1000 * ((sum(losses[-(arguments.window + 1): - 1]) /
                                  (arguments.window * min(losses[-(arguments.window + 1): - 1]))) - 1)

            # Compares the generalised loss and training progress against a selected target value.
            if g_loss / t_progress > arguments.stop_target:
                break

    # Logs the final training information.
    log(arguments, f"\nTraining finished after {epoch} epochs in {int(time.time() - start_time)}s")
    log(arguments, f"Best Epoch {best_epoch} with a temperature of {temperatures[best_epoch - 1]}")

    # Returns the loss values from training and validation epochs and the best epoch.
    return temperatures[best_epoch - 1]


def test_cnn(arguments, device):
    """
        Function for testing the Convolutional neural network and generate csv files with all predictions.
        :param arguments: ArgumentParser Namespace object with arguments used for training.
        :param device: PyTorch device that will be used for training.
        :return: Lists of training and validation losses and an integer for the best performing epoch.
        """

    # Loads the training and validation data.
    _, _, test_data = get_datasets(arguments)

    # Creates the validation data loader using the dataset objects.
    testing_data_loader = DataLoader(test_data, batch_size=arguments.batch_size,
                                     shuffle=False, num_workers=arguments.data_workers,
                                     pin_memory=False, drop_last=False)

    log(arguments, "Loaded Datasets\n")

    # Initialises the classifier model.
    classifier = Classifier(arguments.efficient_net, pretrained=False)

    # Loads the trained model.
    classifier.load_state_dict(torch.load(os.path.join(arguments.model_dir, f"{arguments.experiment}_cnn_best.pt"), map_location=torch.device("cuda")))

    # Sets the classifier to evaluation mode.
    classifier.eval()

    # Moves the classifier to the selected device.
    classifier.to(device)

    test_labels, testing_batches = [], 0
    test_sr_mal, test_sr_ben, test_sr_selections = [], [], []
    test_tmp_mal, test_tmp_ben, test_tmp_selections = [], [], []
    test_mc_mal, test_mc_ben, test_mc_selections = [], [], []

    with torch.no_grad():

        for images, labels in testing_data_loader:

            images = images.to(device)

            labels = labels.cpu().numpy()

            # Performs forward propagation using 16 bit precision.
            if arguments.precision == 16 and device != torch.device("cpu"):
                with amp.autocast():

                    # Performs forward propagation with the model.
                    logits = classifier(images, dropout=False)

            # Performs forward propagation using 32 bit precision.
            else:

                # Performs forward propagation with the model.
                logits = classifier(images, dropout=False)

            sr_predictions = F.softmax(logits, dim=1).cpu().numpy()

            sr_selections = np.amax(sr_predictions, axis=1)

            test_sr_mal += sr_predictions[:, 0].tolist()
            test_sr_ben += sr_predictions[:, 1].tolist()
            test_sr_selections += sr_selections.tolist()

            tmp_predictions = F.softmax(torch.div(logits, arguments.temperature), dim=1).cpu().numpy()

            tmp_selections = np.amax(tmp_predictions, axis=1)

            test_tmp_mal += tmp_predictions[:, 0].tolist()
            test_tmp_ben += tmp_predictions[:, 1].tolist()
            test_tmp_selections += tmp_selections.tolist()

            mc_predictions = []

            if arguments.precision == 16 and device != torch.device("cpu"):
                with amp.autocast():
                    for _ in range(arguments.drop_iterations):
                        mc_predictions.append(classifier(images, dropout=True))
            else:
                for _ in range(arguments.drop_iterations):
                    mc_predictions.append(classifier(images, dropout=True))

            mc_predictions = torch.stack(mc_predictions)

            mc_predictions = F.softmax(mc_predictions, dim=2).cpu().numpy()

            mc_selections = [np.var(mc_predictions[:, i, 0]) for i in range(mc_predictions.shape[1])]

            mc_predictions = np.mean(mc_predictions, 0)

            test_mc_mal += mc_predictions[:, 0].tolist()
            test_mc_ben += mc_predictions[:, 1].tolist()
            test_mc_selections += mc_selections

            test_labels += labels.tolist()
            testing_batches += 1

            # If the number of batches have been reached end validation.
            if testing_batches == arguments.batches_per_epoch:
                break

    filenames = [os.path.basename(file_path)[:-4] for file_path in test_data.filenames]

    sr_output = pd.DataFrame({"image": filenames[:len(test_labels)],
                              "label": test_labels,
                              "mal": test_sr_mal,
                              "ben": test_sr_ben,
                              "sel": test_sr_selections})

    tmp_output = pd.DataFrame({"image": filenames[:len(test_labels)],
                               "label": test_labels,
                               "mal": test_tmp_mal,
                               "ben": test_tmp_ben,
                               "sel": test_tmp_selections})

    mc_output = pd.DataFrame({"image": filenames[:len(test_labels)],
                              "label": test_labels,
                              "mal": test_mc_mal,
                              "ben": test_mc_ben,
                              "sel": test_mc_selections})

    os.makedirs(arguments.output_dir, exist_ok=True)

    sr_output.to_csv(os.path.join(arguments.output_dir, f"{arguments.experiment}_sr_output.csv"), index=False)
    tmp_output.to_csv(os.path.join(arguments.output_dir, f"{arguments.experiment}_tmp_output.csv"), index=False)
    mc_output.to_csv(os.path.join(arguments.output_dir, f"{arguments.experiment}_mc_output.csv"), index=False)
