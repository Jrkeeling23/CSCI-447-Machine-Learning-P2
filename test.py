import unittest
from KNN import KNN


# Source to understand how to test in python: https://pymbook.readthedocs.io/en/latest/testing.html and https://docs.python.org/2/library/unittest.html
class Test(unittest.TestCase):

    def test_predict_by_distance(self):
        knn = KNN()
        class_list = ['right_pick','wrong_pick', 'wrong', 'right_pick', 'wrong_again']
        self.assertEqual(knn.predict_by_distance(class_list), 'right_pick')

    def test_predict_by_distance_with_conflict(self):
        knn = KNN()
        class_list = ['wrong_pick','right_pick', 'wrong_pick', 'wrong', 'right_pick', 'wrong_again']
        self.assertEqual(knn.predict_by_distance(class_list), 'right_pick')



# Source to understand how to test in python: https://pymbook.readthedocs.io/en/latest/testing.html and https://docs.python.org/2/library/unittest.html
if __name__ == '__main__':
    unittest.main()
