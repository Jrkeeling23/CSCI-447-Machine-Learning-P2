import numpy as np
import pandas as pd
import random


class Medoids:

    def __init__(self, data):
        self.df = data

    def perform_medoids(self, k, data_name):
        """
        Will carry out the all the medoid functions
        :param k: number of medoids
        :param data_name:
        :return:
        """
        medoid_points = self.select_random(k)
        self.assign_to_medoids()

    def select_random(self, k):
        """
        :param k: number of medoids
        :return: k random data points
        """
        return self.df.sample(n=k)

    def assign_to_medoids(self):
        # TODO: use euclidean distance to assign other examples to medoids
        pass
