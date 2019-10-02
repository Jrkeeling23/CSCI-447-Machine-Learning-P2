import pandas as pd
import numpy as np
import csv

class Data:
    """
    Anything relating to data such as imputations should be done here.
    """

    def __init__(self):
        """
        Keep the data and functions handling data private (denoted with __ )!
        Initializes data dictionary, which holds the data frames alongside there names.
        """
        # label = ['Age', ,
        self.data_dict = {}
        self.load_data()  # load raw data frames

        self.test_dict = {}
        self.train_dict = {}
        self.split_data()  # split data into testing and training sets

        if self.pre_process_data() is False:
            # TODO: complete the data
            pass
        else:
            pass  # Keep this pass

    def load_data(self):
        """
        Loads data into a dictionary
        :return: None
        """
        # Classification/home/justin/Desktop/ml_p2
        self.data_dict["abalone"] = pd.read_csv(r'data/abalone.data', header=None)
        self.data_dict["car"] = pd.read_csv(r'data/car.data', header=None)
        # TODO Load segmentation data
        self.data_dict["segmentation"] = pd.read_csv(r'data/segmentation.data', header=1, skiprows=[0])
        # Regression
        self.data_dict["machine"] = pd.read_csv(r'data/machine.data', header=None)
        self.data_dict["forest_fires"] = pd.read_csv(r'data/forestfires.data', header=None)
        self.data_dict["wine"] = pd.read_csv(r'data/wine.data', header=None)

    def pre_process_data(self):
        """
        Check if data is complete. No missing values, etc...
        :return: Boolean TODO: Or just fix in here
        """
        # TODO: Ensure that the data is complete
        # TODO: REFORMAT S.T LABEL IS ALWAYS LAST
        pass

    def split_data(self):
        """
        Split data for testing and training
        :return:
        """
        for data_set_name, data_set in self.data_dict.items():  # iterate through
            # use numpys split with pandas sample to randomly split the data
            training_data_temp, test_data_temp = np.split(data_set.sample(frac=1), [int(.8 * len(data_set))])
            # add training/testing data into dictionary with corresponding data set name
            if training_data_temp is not None and test_data_temp is not None:
                self.train_dict[data_set_name] = training_data_temp
                self.test_dict[data_set_name] = test_data_temp


    def k_fold(self, k_val):
        """
        Use k-fold to split data
        TODO: 10 Fold or 5 if need be.
        :param k_val: k value to set size of folds.
        :return:
        """
        pass
