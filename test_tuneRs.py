import unittest

class GridTest(unittest.TestCase):

    def setUp(self):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from tuneRs.tuneRs import GridSearchResample
        np.random.seed(0)
        self.x_train = np.random.rand(80, 2)
        self.x_test = np.random.rand(20, 2)
        self.y_train = np.random.randint(0, 2, 80)
        self.y_test = np.random.randint(0, 2, 20)
        self.test_resampled = GridSearchResample(LogisticRegression(random_state=0), {'C': [0.1, 1, 10]}, random_state=1)
        self.test_split = GridSearchResample(LogisticRegression(random_state=0), {'C': [0.1, 1, 10]},
                                        val_data=(self.x_test, self.y_test), random_state=1)

    def testResampled(self):
        print("Testing Grid Search with resampling...")
        self.test_resampled.fit(self.x_train, self.y_train)
        test_rs_score = int(self.test_resampled.best_score_ * 10)  # should be 5
        test_rs_params = int(self.test_resampled.best_params_['C'] * 10)  # should be 100
        self.AssertEqual(test_rs_score, 5)
        self.AssertEqual(test_rs_params, 100)

    def testSplit(self):
        print("Testing Grid Search without resampling...")
        self.test_split.fit(self.x_train, self.y_train)
        test_split_score = int(self.test_split.best_score_ * 10)  # should be 6
        test_split_params = int(self.test_split.best_params_['C'] * 10)  # should be 1
        self.AssertEqual(test_split_score, 6)
        self.AssertEqual(test_split_params, 1)

class RandomTest(unittest.TestCase):

    def setUp(self):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from tuneRs.tuneRs import RandomSearchResample
        from skopt.space import Real
        np.random.seed(0)
        self.x_train = np.random.rand(80, 2)
        self.x_test = np.random.rand(20, 2)
        self.y_train = np.random.randint(0, 2, 80)
        self.y_test = np.random.randint(0, 2, 20)
        self.test_resampled = RandomSearchResample(LogisticRegression(random_state=0), {'C': Real(0.1, 10)},
                                                   num_iter=10, random_state=1)
        self.test_split = RandomSearchResample(LogisticRegression(random_state=0), {'C': Real(0.1, 10)},
                                               num_iter=10, val_data=(self.x_test, self.y_test), random_state=1)

    def testResampled(self):
        print("Testing Random Search with resampling...")
        self.test_resampled.fit(self.x_train, self.y_train)
        test_rs_score = int(self.test_resampled.best_score_ * 10)  # should be 5
        test_rs_params = int(self.test_resampled.best_params_['C'] * 10)  # should be 73
        self.AssertEqual(test_rs_score, 5)
        self.AssertEqual(test_rs_params, 73)

    def testSplit(self):
        print("Testing Random Search without resampling...")
        self.test_split.fit(self.x_train, self.y_train)
        test_split_score = int(self.test_split.best_score_ * 10)  # should be 6
        test_split_params = int(self.test_split.best_params_['C'] * 10)  # should be 90
        self.AssertEqual(test_split_score, 6)
        self.AssertEqual(test_split_params, 90)

if __name__=='__main__':
    unittest.main()
