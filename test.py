import unittest
from KNN import KNN
import pandas as pd
from process_data import Data
from medoids import KMedoids
import numpy as np
from pandas.util.testing import assert_frame_equal

# Source to understand how to test in python: https://pymbook.readthedocs.io/en/latest/testing.html and https://docs.python.org/2/library/unittest.html
class Test(unittest.TestCase):

    def test_predict_by_distance(self):
        knn = KNN()
        class_list = ['right_pick','wrong_pick', 'wrong', 'right_pick', 'wrong_again']
        self.assertEqual(knn.predict_by_distance(class_list), 'right_pick')

    def test_predict_by_distance_with_conflict(self):
        knn = KNN()
        class_list = [17, 9, 17, 13, 9]
        self.assertEqual(knn.predict_by_distance(class_list), 17)

    def test_eucldean_distance(self):
        '''
        test to see of creating correct distances
        :return:
        '''
        query_point = [0, 0]
        comparison_point = [2, 2]
        knn = KNN()
        self.assertEqual(knn.euclidean_distance(query_point, comparison_point), (8**0.5))

    def test_perform_knn(self):
        query_point = [0, 0]
        comparison_point = [2, 2]
        knn = KNN()
        label = 'true'

    def test_condense_data(self):
        # compare that the size of  output pandas data frame is less than input (that CNN reduced the data)
        # importing part of abalone data to test this as we need the 2D structure
        knn = KNN()
        data = Data()
        data_temp = pd.read_csv(r'data/abalone.data', header=None)
        data_set = data_temp.loc[:400][:]  # get first 100 rows of data_set
        k_val = 5
        name = 'abalone'  # used in KNN, needed here
        cond_data = knn.condense_data(data_set, k_val, name, data)

        self.assertGreater(len(data_set.index),len(cond_data.index))


    def test_medoids(self):
        data = Data()
        data_temp = pd.read_csv(r'data/abalone.data', header=None)
        training_data_temp, test_data_temp = np.split(data_temp.sample(n=300), [int(.8 * 300)])
        data.test_dict = test_data_temp
        data.train_dict = training_data_temp

        md = KMedoids(test_data_temp, training_data_temp)
        # md.data_name = 'abalone'

        md.perform_medoids(5, 'abalone')
        self.assertEqual(len(md.medoids_list), 5)


# Source to understand how to test in python: https://pymbook.readthedocs.io/en/latest/testing.html and https://docs.python.org/2/library/unittest.html
if __name__ == '__main__':
    unittest.main()
