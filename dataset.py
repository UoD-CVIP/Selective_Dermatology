# -*- coding: utf-8 -*-


"""
The file contains the code used for handling the dataset used to train and test the model.
    Dataset - Class for handling the dynamic loading and augmentation of data.
    get_datasets - Function to load the training, validation and testing datasets.
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


__author__    = ["Jacob Carse"]
__copyright__ = "Copyright 2020, Selective Dermatology"
__credits__   = ["Jacob Carse", "Stephen Hogg", "Stephen McKenna"]
__license__   = "MIT"
__version__   = "3.0.0"
__maintainer  = "Jacob Carse"
__email__     = "j.carse@dundee.ac.uk"
__status__    = "Development"


class Dataset(data.Dataset):
    """
    Class for handling the dataset used to train and test the model.
        init - The initiliser for the class.
        len - Gets the size of the dataset.
        getitem - Gets a individual item from the dataset by index.
        augment - Function to augment an input image.
    """

    def __init__(self, arguments, mode, filenames, labels):
        """
        Initiliser for the class that stores the filenames and labels for the model.
        :param arguments: ArgumentParser Namespace containing arguments.
        :param mode: String specifying the type of data loaded, "train", "validation" and "test".
        :param filenames: Array of filenames.
        :param labels: Array of labels.
        """

        # Calls the PyTorch Dataset Initiliser.
        super(Dataset, self).__init__()

        # Stores the arguments and mode in the object.
        self.arguments = arguments
        self.mode = mode

        # Sets the Pillow library to load truncated images.
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        # Stores the filenames and labels in the object.
        self.filenames = filenames
        self.labels = labels

    def __len__(self):
        """
        Gets the length of the dataset.
        :return: Integer for the length of the dataset.
        """

        return len(self.filenames)

    def __getitem__(self, index):
        """
        Gets a given image and label from the datasets based on a given index.
        :param index: Integer representing the index of the data from the dataset.
        :return: A PyTorch tensor with the augmented image and a Integer for the label.
        """

        # Loads and augments the image.
        image = Image.open(self.filenames[index])
        image = self.augment(image)

        # Returns the image and label.
        return image, int(self.labels[index])

    def augment(self, image):
        """
        Method for augmenting a given input image into a tensor.
        :param image: A Pillow Image.
        :return: A augmented image Tensor.
        """

        # Mean and Standard Deviation for normalising the dataset.
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

        # Declares the list of standard transforms for the input image.
        augmentations = [transforms.Resize((self.arguments.image_x, self.arguments.image_y), Image.LANCZOS),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=mean, std=std)]

        # Adds additional transformations if selected.
        if self.arguments.augmentation and self.mode == "train":

            # Class for Random 90 degree rotations.
            class RandomRotation:
                def __init__(self, angles): self.angles = angles
                def __call__(self, x):
                    return transforms.functional.rotate(x, float(np.random.choice(self.angles)))

            # Adds the additional augmentations to the list of augmentations.
            augmentations = augmentations[:1] + [transforms.RandomVerticalFlip(),
                                                 transforms.RandomHorizontalFlip(),
                                                 RandomRotation([0, 90, 180, 270])] + augmentations[1:]

        # Applies the augmentations to the image.
        return transforms.Compose(augmentations)(image)


def get_datasets(arguments):
    """
    Gets the training, validation and testing Dataset objects.
    :param arguments: ArgumentParser Namespace object.
    :return: Dataset objects for training, validation and testing.
    """

    # Reads the training data csv file containing filenames and labels.
    csv_file = pd.read_csv(os.path.join(arguments.dataset_dir, "ISIC_2019_Training_GroundTruth.csv"))

    # Gets the filenames and labels from the csv file.
    filenames = csv_file["image"].tolist()
    labels = np.argmax(np.array(csv_file.drop(["image", "UNK"], axis=1)), axis=1)

    # Adds the file path to each filename.
    for i in range(len(filenames)):
        filenames[i] = f"{arguments.dataset_dir}/ISIC_2019_Training_Input/{filenames[i]}.jpg"

    # Splits the 8 labels into benign, malignant and unknown.
    for i in range(len(labels)):
        if labels[i] in [0, 2]:
            labels[i] = 0
        elif labels[i] in [1]:
            labels[i] = 1
        else:
            labels[i] = 2

    # Gets the indices for each unknown image.
    indices = [i for i, x in enumerate(labels) if x == 2]

    # Gets the filenames from the training array and then deletes them from the original array.
    test_filenames = np.take(filenames, indices)
    filenames = np.delete(filenames, indices)

    # Gets the labels from the training array and deletes them from the original array.
    test_labels = np.take(labels, indices)
    labels = np.delete(labels, indices)

    # Splits the dataset into training and temp(testing and validation).
    filenames, temp_filenames, labels, temp_labels = train_test_split(filenames, labels,
                                                                      test_size=0.2 + arguments.val_split,
                                                                      random_state=arguments.seed)

    # Splits the temp data into testing and validation
    temp_filenames, val_filenames, temp_labels, val_labels = train_test_split(temp_filenames, temp_labels,
                                                                              test_size=0.2 + arguments.val_split,
                                                                              random_state=arguments.seed)

    # Adds the additional testing images and labels to the testing dataset.
    test_filenames = np.append(test_filenames, temp_filenames)
    test_labels = np.append(test_labels, temp_labels)

    # Creates the training, validation and testing dataset objects.
    train_data = Dataset(arguments, "train", filenames, labels)
    val_data = Dataset(arguments, "validation", val_filenames, val_labels)
    test_data = Dataset(arguments, "test", test_filenames, test_labels)

    # Returns the dataset objects.
    return train_data, val_data, test_data
