import numpy as np
import pandas as pd
import random


class KMedoids:

    def __init__(self, data):
        self.df = data
        self.medoids = None
        self.medoids_list = None  # contains a list of the Medoid instances
        self.data_name = None

    def perform_medoids(self, k, data_name):
        """
        Will carry out the all the medoid functions
        :param k: number of medoids
        :param data_name:
        :return:
        """
        self.data_name = data_name
        self.medoids = self.select_random(k)
        self.create_medoid_instances(self.medoids)
        self.assign_to_medoids()

    def select_random(self, k):
        """
        randomly selects examples to represent the medoids
        :param k: number of medoids
        :return: k random data points
        """
        return self.df.sample(n=k)

    def assign_to_medoids(self):
        """

        :return:
        """
        for index, row in self.df.iterrows():
            if index in self.medoids.index:
                continue
            else:
                self.check_all_medoid_distances(row)  # checking the row to all medoids for closest distance

        return True

    def check_all_medoid_distances(self, query_point):
        """

        :return:
        """
        distance_dict = {}
        for index, row in self.medoids.iterrows():
            distance_dict[index] = self.euclidean_distance(query_point, row)  # add the distances to the dict with index

    def euclidean_distance(self, query_point, comparison_point):
        """
        With multi dimensions: sqrt((x2-x1)+(y2-y1)+(z2-z1)+...))
        :param query_point: an example from the data set
        :param comparison_point: a medoid
        :return: float distance
        """
        temp_add = 0  # (x2-x1)^2 + (y2 - y1)^2 ; addition part
        for feature_col in range(len(query_point)):
            if self.df.get_label_col(self.data_name) is feature_col:
                continue
            if type(query_point[feature_col]) is float or type(query_point[feature_col]) is int:
                temp_sub = (query_point[feature_col] - comparison_point[feature_col]) ** 2  # x2 -x1 and square
                temp_add += temp_sub  # continuously add until square root

        return temp_add ** (1 / 2)  # square root

    def sort_dict_by_value(self, sort_this):
        """
        determines the closest medoid to query point
        :param sort_this: dictionary of distances from query point to medoids
        :return: closest medoid
        """
        k_size = self.medoids.shape[0]
        first_iteration = True
        closest = None
        for k, v in sort_this.items():
            if first_iteration:
                closest = k  # grab specific medoid
                first_iteration = False  # assign first distance
                continue
            else:
                if closest > v:
                    closest = k  # change closest var to closer
        return closest

    def create_medoid_instances(self, medoids):
        """
        creates Medoid instances from the randomly selected medoids
        :param medoids:
        :return:
        """
        self.medoids_list = []
        for index, row in medoids.iterrows():
            self.medoids_list.append(Medoids(row, index))
        print(self.medoids_list)


class Medoids:
    def __init__(self, point, index):
        """
        Medoids instance. Initializes the medoids current point, what data is contained within that medoid and an index
        :param point: the point representing the medoid
        :param index: index of the point
        """
        self.medoid_point = point
        self.encompasses = {}
        self.index = index

    def assign_to_medoid(self, index, row):
        """
        Assigns data to the medoid
        :param index: index of data
        :param row:  row of data  ... may not be needed
        :return:
        """
        self.encompasses[index] = row
