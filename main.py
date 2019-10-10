"""
CSCI-447 Machine Learning
Justin Keeling, Alex Harry, Andrew Smith, John Lambrecht
11 Oct 2019
"""
from process_data import Data
from KNN import KNN


def main():
    """
    Calls function in other files until program is finished.
    :return: None
    """
    knn = KNN()
    data = Data()  # loads the data and checks if complete

    while True:
        data.split_data()  # split into both test and train
        predicted_class = {}  # holds data_set_name and a list of predicted classes

        for name, train_data_set in data.train_dict.items():  # iterate through data and get key(Data name) and data_set
            print("Current Data Set: ", name)
            predicted_class[name] = []  # create a list of for a data set of predicted values
            test_data_set = data.test_dict[name]  # TODO: Use same keys for all dictionaries; Access testing data by key.
            for _, query_point in train_data_set.iterrows():
                # give query example and its corresponding train_data_set, along with # of desired neighbors to consider
                predicted_class[name].append(knn.perform_knn(query_point, train_data_set, 5, name, data))


def test():
    knn = KNN()
    data = Data()  # loads the data and checks if complete
    data.split_data()
    k_neighbors = 5
    for names, data_set in data.train_dict.items():
        print(data.train_dict['segmentation'])
        # knn.perform_knn(k_neighbors)
testing = False
if __name__ == "__main__":
    if testing:
        test()
    else:
        main()
