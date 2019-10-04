import pandas as pd
import numpy as np
from process_data import Data


class KNN:
    """
    Anything to do with k-nearest neighbor should be in here.
    """

    def perform_knn(self, query_point, train_data, k_val):
        """
        Function performs KNN to classify predicted class.
        :param k_val: number of neighbors
        :param query_point: all data to compare an example from test_data too.
        :param train_data:  all data to "query" and predict
        :return: Predicted class
        """
        print("\n-----------------Performing KNN-----------------")
        distance_list = []
        index_list = []
        for index, row in train_data.iterrows():  # iterate through all data and get distances
            if len(distance_list) is k_val + 1:  # keep list of size k
                # distance_list.sort(reverse=True)  # least to greatest
                distance = self.euclidean_distance(query_point, row)  # check first spot
                for i in range(k_val):
                    
                    distance_list[0] = distance  # swap value to closer neighbor
                    index_list.append()
            else:
                distance_list[index] = (self.euclidean_distance(query_point, row))  # all features of x to a euclidean.
        # distance_list.sort(reverse=True) # Sort least to greatest.
        # distance_list = distance_list[1:k_val + 1]  # get k closest neighbors
        # for key, value in sorted(distance_list.items(), key=lambda item: item[1]):
        #     # print("%s: %s" % (key, value))

        key_min = min(distance_list.keys(), key=(lambda k: distance_list[k]))
        print(key_min)
        # for val in key_min.items():


        print(str(k_val), "Nearest Neighbors to Query Point: ", query_point, ':', distance_list)


        return self.predict_by_distance(distance_list)

    def euclidean_distance(self, query_point, comparison_point):
        """
        With multi dimensions: sqrt((x2-x1)+(y2-y1)+(z2-z1)+...))
        :param query_point: Testing example.
        :param comparison_point: example in training data.
        :return: float distance
        """
        print("\n-----------------Getting Euclidean Distances-----------------")
        temp_add = 0  # (x2-x1)^2 + (y2 - y1)^2 ; addition part
        for feature_col in range(len(query_point)):
            if type(query_point[feature_col]) is float or type(query_point[feature_col]) is int:
                temp_sub = (query_point[feature_col] - comparison_point[feature_col]) ** 2  # x2 -x1 and square
                temp_add += temp_sub  # continuously add until square root

        return temp_add ** (1 / 2)  # square root

    def predict_by_distance(self, distance_list):
        """
        Determines the prediction of class by closest neighbors.
        :param distance_list:
        :return: Predicted class
        """
        print("\n-----------------Deciding Predicted Nearest Neighbor-----------------")
        loop_iterator_location = len(distance_list)  # Variable changes if nearest neighbor conflict.
        while True:
            nearest_neighbor = distance_list[0]  # Sets the current pick to the first value in the list
            predict_dictionary = {}  # Temp dictionary to keep track of counts
            for class_obj in distance_list[
                             :loop_iterator_location]:  # Loops through the input list to create a dictionary with values being count of classes
                if class_obj in predict_dictionary.keys():  # Increases count if key exists
                    predict_dictionary[class_obj] += 1
                    if predict_dictionary[nearest_neighbor] < predict_dictionary[class_obj]:
                        nearest_neighbor = class_obj  # Sets the nearest neighbor to the class that occurs most.
                else:
                    predict_dictionary[class_obj] = 1  # Create key and set count to 1
            check_duplicates = list(predict_dictionary.values())  # Create a list to use the count function
            if check_duplicates.count(predict_dictionary[
                                          nearest_neighbor]) == 1:  # Sets conflict to False if the count of the top class occurrences is the only class sharing that count
                break
            else:
                loop_iterator_location -= 1  # By reducing the loop iterator, we remove the furthest neighbor from our counts.
        print("Predicted Nearest Neighbor: ", nearest_neighbor)
        return nearest_neighbor

    def edit_data(self):
        """
        Edit values for edit_knn by classifying x_initial; if wrong, remove x_initial. (option1)
        OR... if correct remove (option 2)
        :return: Edited values back to KNN
        """
        # TODO: edit data according to pseudo code from class on 9/23
        pass

    def condense_data(self):
        """
        Condense the data set by instantiating a Z = None. Add x_initial to Z if initial class(x_initial) != class(x)
        where x is an example in Z.
        So: Eliminates redundant data.
        :return:
        """
        # TODO: edit data according to pseudo code from class on 9/23
        pass