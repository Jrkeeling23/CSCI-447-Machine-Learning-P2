import pandas as pd
import numpy as np
from process_data import Data


class KNN:
    """
    Anything to do with k-nearest neighbor should be in here.
    """

    def __init__(self):
        self.current_data_set = None
        self.data = None

    def perform_knn(self, query_point, train_data, k_val, name, in_data):
        """
        Function performs KNN to classify predicted class.
        :param in_data: the data instance from main to call process data functions
        :param name:  name of the data_set
        :param k_val: number of neighbors
        :param query_point: all data to compare an example from test_data too.
        :param train_data:  all data to "query" and predict
        :return: Predicted class
        """
        self.current_data_set = name
        self.data = in_data
        print("\n-----------------Performing KNN-----------------")
        distance_dict = {}  # place all indexes, which are unique, and distances in dictionary
        distance_list = []  # holds the k-number of distances
        label_list = []  # holds the k-number of labels associated with disances
        for index, row in train_data.iterrows():  # iterate through all data and get distances
            distance = (self.euclidean_distance(query_point, row))  # all features of x to a euclidean.
            distance_dict[index] = distance

        count = 0  # stops for loop
        for key, value in sorted(distance_dict.items(), key=lambda item: item[1]):
            # key is the index and value is the distance. Ordered least to greatest by sort().
            # if statement to grab the k number of distances and labels
            if count > k_val:
                break
            elif count is 0:
                count += 1  # first value is always 0.
                continue
            else:
                distance_list.append(value)  # add distance
                label_list.append(train_data.loc[key,self.data.get_label_col(self.current_data_set)])  # add label
                count += 1
        # TODO: get rid of prints, only needed to show you all the structure.
        print(distance_list)
        print(label_list)

        print(str(k_val), "Nearest Neighbors to Query Point: ", query_point, ':', distance_list)

        return self.predict_by_distance(distance_list, label_list)  # return the predicted values

    def euclidean_distance(self, query_point, comparison_point):
        """
        With multi dimensions: sqrt((x2-x1)+(y2-y1)+(z2-z1)+...))
        :param query_point: Testing example.
        :param comparison_point: example in training data.
        :return: float distance
        """
        # print("\n-----------------Getting Euclidean Distances-----------------")
        temp_add = 0  # (x2-x1)^2 + (y2 - y1)^2 ; addition part
        for feature_col in range(len(query_point)):
            if self.data.get_label_col(self.current_data_set) is feature_col:
                continue
            if type(query_point[feature_col]) is float or type(query_point[feature_col]) is int:
                temp_sub = (query_point[feature_col] - comparison_point[feature_col]) ** 2  # x2 -x1 and square
                temp_add += temp_sub  # continuously add until square root

        return temp_add ** (1 / 2)  # square root

    def predict_by_distance(self, distance_list, label_list):
        """
        Determines the prediction of class by closest neighbors.
        :param label_list: k-number of labels associated with distance list
        :param distance_list: k-number of closest distances
        :return: Predicted class
        """
        # TODO: finish predict using the labels. Predict the labels
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

    def edit_data(self, data_set, k_value, name, validation):
        """
        Edit values for edit_knn by classifying x_initial; if wrong, remove x_initial. (option1)
        OR... if correct remove (option 2)
        :param data_set: the training data that will be edited
        :param k_value: the number of neighbors being checked against
        :param name: name of the data_set
        :param validation: the test data, so that there is a measurement of performance to know when to stop
        :return: Edited data_set back to KNN
        """
        # TODO: edit data according to pseudo code from class on 9/23
        self.current_data_set = name
        # prev_set = data_set
        data_set_perform = 0  # for getting an initial measure on performance
        for index, row in data_set.iterrows:  # loops through the validation set and if it matches, then it adds one to the score
            knn = self.perform_knn(row, data_set, k_value, self.current_data_set, validation)
            if knn == row[self.data.get_label_col(self.current_data_set)]:
                data_set_perform+=1
        prev_set_perform = data_set_perform  # for allowing the loop to occur
        while data_set_perform > prev_set_perform:  # doesn't break until the performance drops below the previous set
            prev_set_perform = data_set_perform  # sets the previous set and previous set performance
            prev_set = data_set
            list_to_remove = []  # initializes the list of items that will be removed
            for index, row in data_set.iterrows():  # does knn on itself
                knn_value = self.perform_knn(row, data_set, k_value, self.current_data_set, data_set)
                actual_value = row[self.data.get_label_col(self.current_data_set)]
                if knn_value!=actual_value:  # comparing the knn done on itself to it's actual value.  If it doesn't match, it will be removed
                    list_to_remove.append(index)
            data_set.drop(list_to_remove)  # removes the data points that don't match
            data_set_perform = 0  # resets the performance measure
            for index, row in data_set.iterrows:  # gets the performance measure
                knn = self.perform_knn(row, data_set, k_value, self.current_data_set, validation)
                if knn == row[self.data.get_label_col(self.current_data_set)]:
                    data_set_perform += 1
        return prev_set  # returns the set with the best performance





    def condense_data(self, data_set, k_val, name, in_data):
        """
        Condense the data set by instantiating a Z = None. Add x_initial to Z if initial class(x_initial) != class(x)
        where x is an example in Z.
        So: Eliminates redundant data.
        :return:
        """
        # TODO: edit data according to pseudo code from class on 9/23
        pass
