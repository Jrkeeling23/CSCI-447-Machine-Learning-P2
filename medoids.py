import numpy as np
import pandas as pd
import random
from process_data import Data


def get_label_col(data_name):
    col_loc = {'abalone': 8, 'car': 5, 'segmentation': 0, 'machine': 0, 'forest_fires': 12, 'wine': 0}
    return col_loc[data_name]


def euclidean_distance(query_point, comparison_point, data_name):
    """
    Performs the Euclidean distance function
    :param query_point: a data point
    :param comparison_point: a comparison point
    :param df: used to get the label columns
    :return: a distance
    """
    temp_add = 0  # (x2-x1)^2 + (y2 - y1)^2 ; addition part
    for feature_col in range(len(query_point)):
        if get_label_col(data_name) is feature_col:
            continue
        if type(query_point[feature_col]) is float or type(query_point[feature_col]) is int:
            temp_sub = (query_point[feature_col] - comparison_point[feature_col]) ** 2  # x2 -x1 and square
            temp_add += temp_sub  # continuously add until square root

    return temp_add ** (1 / 2)  # square root ... return the specific distance


class KMedoids:

    def __init__(self, data):
        self.df = data
        self.medoids_list = None  # contains a list of the Medoid instances
        self.data_name = None

    def perform_medoids(self, k, data_name):
        """
        Will carry out the all the medoid functions
        :param k: number of medoids
        :param data_name:
        :return:
        """
        self.data_name = data_name  # assigns the current data set being used for label column purposes
        self.select_random(k)  # select random data points to represent the medoids
        self.assign_to_medoids()  # assign the remaining data points to its closest medoid
        for med in self.medoids_list:
            print(med.encompasses)

    def select_random(self, k):
        """
        randomly selects examples to represent the medoids
        :param k: number of medoids
        :return: k random data points
        """
        self.create_medoid_instances(self.df.sample(n=k))

    def assign_to_medoids(self):
        """
        assigns the remaining data points to its closest medoid
        :return:
        """
        for index, row in self.df.iterrows():
            if index in self.medoids_list:
                continue  # exclude to data points that are medoids
            elif index not in self.medoids_list and index is not None:
                chosen_medoid = self.check_all_medoid_distances(row)  # check the row to  medoids for closest distance
                chosen_medoid.assign_to_medoid(index)  # assign the index of the data point to the medoid

    def check_all_medoid_distances(self, query_point):
        """
        calculate the Euclidean distances from the query point to the medoid
        :param query_point:
        :return:  distance
        """
        distances = {}  # dict to hold the indexes and the distances
        for med in self.medoids_list:
            distances[med] = euclidean_distance(query_point, med.medoid_point, self.data_name)
        return self.sort_dict_by_value(distances)

    def sort_dict_by_value(self, sort_this):
        """
        determines the closest medoid to query point
        :param sort_this: dictionary of distances from query point to medoids
        :return: closest medoid
        """
        k_size = len(self.medoids_list)
        first_iteration = True
        closest_medoid = None
        saved_closest = None

        for k, v in sort_this.items():
            if first_iteration:
                saved_closest = v  # grab specific medoid
                first_iteration = False  # assign first distance
                closest_medoid = k
                continue
            else:
                if saved_closest > v:
                    saved_closest = v  # change closest var to closer
                    closest_medoid = k
        return closest_medoid  # closest medoid

    def create_medoid_instances(self, medoids):
        """
        creates Medoid instances from the randomly selected medoids
        :param medoids:
        :return:
        """
        self.medoids_list = []
        for index, row in medoids.iterrows():
            self.medoids_list.append(Medoids(row, index))

    # TODO: function that will be called after data points have been assigned to a medoid.... it will have a while
    #  loop that will continue iterating until medoid points do not change.


class Medoids:
    def __init__(self, point, index):
        """
        Medoids instance. Initializes the medoids current point, what data is contained within that medoid and an index
        :param point: the point representing the medoid
        :param index: index of the point
        """
        self.medoid_point = point
        self.encompasses = []
        self.index = index
        self.distance_list = {}

    def assign_to_medoid(self, index):
        """
        Assigns data to the medoid
        :param index: index of data
        :return:
        """
        self.encompasses.append(index)

    def change_medoid_point(self):
        # TODO: Once the data points have been assigned to a medoid, a better data_point may need to be used instead!
        pass

    def check_for_better_fit(self):
        # TODO: Function that will check the medoids encompassed data points for a better fit.
        pass
