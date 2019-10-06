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
        print("Distance List: ",distance_list)
        print('Label list', label_list)

        print(str(k_val), "Nearest Neighbors (Class) to Query Point: ", label_list)

        return self.predict_by_distance(label_list)  # return the predicted values

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

    def predict_by_distance(self, label_list):
        """
        Determines the prediction of class by closest neighbors.
        :param label_list: k-number of labels associated with distance list
        :param distance_list: k-number of closest distances
        :return: Predicted class
        """
        print("\n-----------------Deciding Predicted Nearest Neighbor-----------------")
        loop_iterator_location = len(label_list)  # Variable changes if nearest neighbor conflict.
        while True:
            nearest_neighbor = label_list[0]  # Sets the current pick to the first value in the list
            predict_dictionary = {}  # Temp dictionary to keep track of counts
            for class_obj in label_list[
                             :loop_iterator_location]:  # Loops through the input list of labels to create a dictionary with values being count of classes
                if class_obj in predict_dictionary.keys():  # Increases count if key exists
                    predict_dictionary[class_obj] += 1
                    if predict_dictionary[nearest_neighbor] < predict_dictionary[class_obj]:
                        nearest_neighbor = class_obj  # Sets the nearest neighbor to the class that occurs most.
                else:
                    predict_dictionary[class_obj] = 1  # Create key and set count to 1
            check_duplicates = list(predict_dictionary.values())  # Create a list to use the count function
            if check_duplicates.count(predict_dictionary[
                                          nearest_neighbor]) == 1:  # Breaks out of loop if the count of the top class occurrences is the only class sharing that count
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

    def condense_data(self, data_set, k_val, name, in_data):
        """
        Condense the data set by instantiating a Z = None. Add x_initial to Z if initial class(x_initial) != class(x)
        where x is an example in Z.
        So: Eliminates redundant data.
         :param dataSet: data we want to reduce.
         :param k_val: # of neighbors, used when performing knn
        :return: condensed data
        """
        # TODO: edit data according to pseudo code from class on 9/23
        print("\n-----------------Performing Condensed Dataset Reduction-----------------")
        self.data = in_data
        self.current_data_set = name
        # new dataset to hold condensed values
        # condensed_data = pd.DataFrame()
        firstElem = [] # use later to store values to remake dataset
        list_for_adding =[]
        list_for_adding.append(firstElem)
        for  val in data_set.iloc[0]:
            firstElem.append(val)
        col_list = list(data_set.columns)

        # finally got adding 1 row down
        condensed_data = pd.DataFrame([firstElem], columns= col_list)
        # condensed_data = condensed_data.append(firstElem)

        has_changed = True  # bool to break if condensedData no longer changes
        condensed_size = len(condensed_data.index)  # var to keep track of size of condensed data
        # add first found example to the data set (assuming [0][:] is valid here

        while has_changed is True:  # outside loop for CNN

            lowest_distance = 99999999 # holding distance here, settting to 999 just to make sure we get a smaller num
            minimum_index = -1  # index for that minimum element
            # go through every point in the data set, get point with lowest distance with class != to our example
            # TODO:  May want to make this random, not sure if needed
            for index, row in data_set.iterrows():
         # go through the condensed dataset and find point with the lowest distance to point from actual data (euclidian)
                for c_index, c_row in condensed_data.iterrows():  # should start with at least one
                  #  print(c_row)

                    e_dist = self.euclidean_distance(row,c_row)  # take distance
                    if e_dist < lowest_distance:  # compare current dist to last seen lowest
                        lowest_distance = e_dist  # store lowest distance
                        minimum_index = c_index  # store minimum index to check classification
                        # classify our 2 vals with KNN and compare class values
                # selecting value found an minimum index for condensed, and using the row we are iterating on
                condensed_value = self.perform_knn(condensed_data.iloc[minimum_index][:], data_set, k_val, self.current_data_set, self.data)
                data_set_value = self.perform_knn(row, data_set, k_val, self.current_data_set, self.data)
                # compare the classes of the two predicted values
                # this assumes we get examples back that we need to select class from KNN
                #  TODO:  change this as needed by KNN algo
                if condensed_value != data_set_value:
                    # create new data set with new values
                    print("\n-----------------Adding datapoint to condensed dataset-----------------")
                    # add new values to list and append that 2 list for condensed data
                    vals = []
                    for val in row:
                        vals.append(val)
                    list_for_adding.append(vals)

                    condensed_data = pd.DataFrame(list_for_adding)

                    # print(len(condensed_data.index))
                    # print(condensed_size)
                print(has_changed)
            # checking if the size of the condense dataset has changed, if so keep going, if not end loop
            if condensed_size is len(condensed_data.index) or len(condensed_data.index) > 100:
                has_changed = False  # if the length Has not changed, end loop
                break
            elif condensed_size > 10000:  # just in case break condition TODO: possibly remove
                print("in elif")
                has_changed = False
                break
            else:
                has_changed = True  # size has changed, keep going
                condensed_size = len(condensed_data.index) # update our length


            # another break
           # print(has_changed)
           # if has_changed is False:
            #    print("in final break 2")
            #    break

        print("\n-----------------Finished performing Condensed Dataset Reduction-----------------")
        return condensed_data
