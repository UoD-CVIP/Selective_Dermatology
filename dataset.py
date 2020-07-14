# -*- coding: utf-8 -*-

"""
This file contains the following dataset handling functions and classes:
    get_datasets - Function to get the training, validation and testing dataset objects.
    Dataset - Class for the handling and dynamic loading and augmenting of images and labels.
"""


# Built-in/Generic Imports
import os

# Library Imports
import numpy as np
import pandas as pd
from torch.utils import data
from PIL import Image, ImageFile
from torchvision import transforms
from sklearn.model_selection import train_test_split


__author__ = ["Jacob Carse", "Stephen Hogg"]
__credits__ = ["Jacob Carse", "Stephen Hogg"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


def get_datasets(arguments):
    """
    Get the training, validation and testing dataset objects.
    :param arguments: Dictionary containing arguments.
    :return: Dataset objects for training, validation and testing.
    """

    # Reads the training data csv file containing filenames and labels.
    csv_file = pd.read_csv(os.path.join(arguments["dataset_dir"], "ISIC_2019_Training_GroundTruth.csv"))

    # Gets the filenames and labels from the csv file.
    filenames = csv_file["image"].tolist()
    labels = np.argmax(np.array(csv_file.drop(["image", "UNK"], axis=1)), axis=1)

    # Adds the dataset directory path for each image file path.
    for i in range(len(filenames)):
        filenames[i] = f"{arguments['dataset_dir']}/ISIC_2019_Training_Input/{filenames[i]}.jpg"

    # Splits the 8 labels into benign, malignant and unknown.
    for i in range(len(labels)):
        if labels[i] in [0, 2]:
            labels[i] = 0
        elif labels[i] == 1:
            labels[i] = 1
        else:
            labels[i] = 2

    # Gets the indices for all the unknown images.
    indices = [i for i, x in enumerate(labels) if x == 2]

    # Gets the unknown filenames and labels from the training data.
    test_filenames = np.take(filenames, indices)
    test_labels = np.take(labels, indices)

    # Removes the unknown filenames and labels from their arrays.
    filenames = np.delete(filenames, indices)
    labels = np.delete(labels, indices)

    # Splits the dataset into training and testing.
    test_split = arguments["val_split"] + arguments["test_split"]
    filenames, temp_filenames, labels, temp_labels = train_test_split(filenames, labels,
                                                                      test_size=test_split,
                                                                      random_state=arguments["seed"])

    # Splits the testing dataset into validation and testing.
    val_split = arguments["val_split"] / (arguments["val_split"] + arguments["test_split"])
    temp_filenames, val_filenames, temp_labels, val_labels = train_test_split(temp_filenames, test_labels,
                                                                              test_size=val_split,
                                                                              random_state=arguments["seed"])

    # Adds additional testing images with unknown image label.
    for file in os.listdir(os.path.join(arguments["dataset_dir"], "Test_Images")):
        if file.endswith(".jpg"):
            temp_filenames = np.append(temp_filenames, f"{arguments['dataset_dir']}/Test_Images/{file}")
            temp_labels = np.append(temp_labels, 2)

    # Adds the additional testing images and labels to the testing dataset.
    test_filenames = np.append(test_filenames, temp_filenames)
    test_labels = np.append(test_labels, temp_labels)

    # Creates the training, validation and testing dataset objects.
    train_data = Dataset(arguments, "train", filenames, labels)
    val_data = Dataset(arguments, "validation", val_filenames, val_labels)
    test_data = Dataset(arguments, "test", test_filenames, test_labels)

    # Returns the dataset objects.
    return train_data, val_data, test_data


class Dataset(data.Dataset):
    """
    This class for handling datasets contains the methods:
        init - The intiliser for the class.
        len - Gets the length of the dataset.
        getitem - Gets an item from the dataset based on a index.
        augment - Augments a given input image.
    """

    def __init__(self, arguments, mode, filenames, labels):
        """
        Initiliser for the class that sets the filenames and labels.
        :param arguments: Dictionary containing arguments.
        :param mode: String specifying the type of dataset, "train", "validation" and "test".
        :param filenames: NumPy array of filenames.
        :param labels: NumPy array of labels.
        """

        # Calls the PyTorch Dataset Initiliser.
        super(Dataset, self).__init__()

        # Stores the arguments and mode in the object.
        self.arguments = arguments
        self.mode = mode

        # Stores the filenames and labels in the object.
        self.filenames = filenames
        self.labels = labels

        # Configures PIL to load truncated images.
        ImageFile.LOAD_TRUNCATED_IMAGES = True

    def __len__(self):
        """
        Get the length of the dataset.
        :return: Integer for the length of the dataset.
        """

        return len(self.filenames)

    def __getitem__(self, index):
        """
        Gets a given item from the dataset based on a given index.
        :param index: Integer representing the index of the data from the dataset.
        :return: A PyTorch Tensor containing the extracted image and a integer of the corresponding label.
        """

        # Loads and augments the image.
        image = Image.open(self.filenames[index])
        image = self.augment(image)

        # Returns the image and label.
        return image, int(self.labels[index])

    def augment(self, image):
        """
        Method for augmenting a given input image into a tensor and applying additional augmentations.
        :param image: A PIL image.
        :return: A image Tensor.
        """

        # Mean and Standard Deviation used for normalising the dataset.
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

        # Declare the standard transforms to be applied to the image tensor.
        transformations = [transforms.Resize((self.arguments["image_x"], self.arguments["image_y"]), Image.LANCZOS),
                           transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]

        # Additional augmentations applied if in training mode.
        if self.arguments["augmentation"] and self.mode == "train":
            # Random rotation transform.
            class RandomRotation:
                def __init__(self, angles): self.angles = angles
                def __call__(self, x): return transforms.functional.rotate(x, np.random.choice(self.angles))

            # Adds additional transforms to the list of transformations.
            transformations = [transforms.RandomVerticalFlip(),
                               transforms.RandomHorizontalFlip(),
                               RandomRotation([0, 90, 180, 270])] + transformations

        # Applies the set of transforms to the input image.
        return transforms.Compose(transformations)(image)
