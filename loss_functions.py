import process_data as data
import pandas as pd
from KNN import KNN


class LF:

    def zero_one_loss(self, data_set, k_val, name, in_data):
            """
                 Condense the data set by instantiating a Z = None. Add x_initial to Z if initial class(x_initial) != class(x)
                 where x is an example in Z.
                 So: Eliminates redundant data.
                 :param data_set: data we want to reduce.
                 :param k_val: # of neighbors, used when performing knn
                 :param name: name of data set we are working with (for knn)
                 :param in_data:  data object to perform operations in process_data class
                 :return: the zero one less (accuracy) of KNN on a given set of data
             """
            knn = KNN()
            zero_count = 0  # count of correctly classified examples
            one_count = 0  # count of incorrectly classified examples
            class_col = in_data.get_label_col(name)  # column that class values will be stored in for given dataset
            print("\n-----------------Performing Zero One Loss-----------------")

            # go through each datapoint (row) and check what class KNN thinks it is
            for index, row in data_set.iterrows():

                knn_class = knn.perform_knn(row, data_set, k_val,name, in_data)
                actual_class = row[class_col]
                # if they match increment the zero count
                print("KNN class: ", knn_class, " ACTUAL CLASS: ", actual_class)
                if int(knn_class) == int(actual_class):
                    zero_count += 1
                else:
                    one_count += 1   # else increment one count

            print("\n--- KNN Classified ", zero_count, "Examples Correctly and ", one_count, "Incorrectly---")
            print("\n--- With total Zero One Loss of: ", (zero_count/one_count), "---")
            return zero_count/one_count  # total accuracy
