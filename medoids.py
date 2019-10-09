import numpy as np
import pandas as pd
import random


def euclidean_distance(query_point, comparison_point, df):
    """
    Performs the Euclidean distance function
    :param query_point: a data point
    :param comparison_point: a comparison point
    :param df: used to get the label columns
    :return: a distance
    """
    temp_add = 0  # (x2-x1)^2 + (y2 - y1)^2 ; addition part
    for feature_col in range(len(query_point)):
        if df.get_label_col(df.data_name) is feature_col:
            continue
        if type(query_point[feature_col]) is float or type(query_point[feature_col]) is int:
            temp_sub = (query_point[feature_col] - comparison_point.row[feature_col]) ** 2  # x2 -x1 and square
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
            else:
                chosen_medoid = self.check_all_medoid_distances(row)  # check the row to  medoids for closest distance
                chosen_medoid.encompasses.assign_to_medoid(index)  # assign the index of the data point to the medoid

    def check_all_medoid_distances(self, query_point):
        """
        calculate the Euclidean distances from the query point to the medoid
        :param query_point:
        :return:  distance
        """
        distances = {}  # dict to hold the indexes and the distances
        for med in self.medoids_list:
            distances[med] = euclidean_distance(query_point, med.row, self.df)
        return self.sort_dict_by_value(distances)

    def sort_dict_by_value(self, sort_this):
        """
        determines the closest medoid to query point
        :param sort_this: dictionary of distances from query point to medoids
        :return: closest medoid
        """
        k_size = len(self.medoids_list)
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
        return closest  # closest medoid

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

    # TODO: add functions to change the medoids and pick a more appropriate point
