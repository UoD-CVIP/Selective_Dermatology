# -*- coding: utf-8 -*-

"""
This file contains the following dataset handling functions and classes:
    Dataset - Class for the handling and dynamic loading and augmenting of images and labels.
"""


# Built-in/Generic Imports

# Library Imports
import numpy as np
from torch.utils import data
from PIL import Image, ImageFile
from torchvision import transforms


__author__ = ["Jacob Carse", "Stephen Hogg"]
__credits__ = ["Jacob Carse", "Stephen Hogg"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Jacob Carse"
__email__ = "j.carse@dundee.ac.uk"
__status__ = "Development"


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
