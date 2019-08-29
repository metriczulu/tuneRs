import unittest


class GridTest(unittest.TestCase):

    def setUp(self):
        import numpy as np
        from tuneRs.tuneRs import GridSearchResample
        from sklearn.linear_model import LogisticRegression
        np.random.seed(0)
        self.x_train = np.random.rand(80, 2)
        self.x_test = np.random.rand(20, 2)
        self.y_train = np.random.randint(0, 2, 80)
        self.y_test = np.random.randint(0, 2, 20)
        self.test_resampled = GridSearchResample(LogisticRegression(random_state=0), {'C': [0.1, 1, 10]}, random_state=1)
        self.test_split = GridSearchResample(LogisticRegression(random_state=0), {'C': [0.1, 1, 10]},
                                        val_data=(self.x_test, self.y_test), random_state=1)

    def test_resampled(self):
        self.test_resampled.fit(self.x_train, self.y_train)
        test_rs_score = int(self.test_resampled.best_score_ * 10)  # should be 5
        test_rs_params = int(self.test_resampled.best_params_['C'] * 10)  # should be 100
        self.assertEqual(test_rs_score, 5, "Incorrect grid resample score")
        self.assertEqual(test_rs_params, 100, "Incorrect grid resample param")

    def test_split(self):
        self.test_split.fit(self.x_train, self.y_train)
        test_split_score = int(self.test_split.best_score_ * 10)  # should be 6
        test_split_params = int(self.test_split.best_params_['C'] * 10)  # should be 1
        self.assertEqual(test_split_score, 6, "Incorrect grid split score")
        self.assertEqual(test_split_params, 1, "Incorrect grid split score")


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

    def test_resampled(self):
        self.test_resampled.fit(self.x_train, self.y_train)
        test_rs_score = int(self.test_resampled.best_score_ * 10)  # should be 5
        test_rs_params = int(self.test_resampled.best_params_['C'] * 10)  # should be 73
        self.assertEqual(test_rs_score, 5, "Incorrect random resample score")
        self.assertEqual(test_rs_params, 73, "Incorrect random resample param")

    def test_split(self):
        self.test_split.fit(self.x_train, self.y_train)
        test_split_score = int(self.test_split.best_score_ * 10)  # should be 6
        test_split_params = int(self.test_split.best_params_['C'] * 10)  # should be 90
        self.assertEqual(test_split_score, 6, "Incorrect random split score")
        self.assertEqual(test_split_params, 90, "Incorrect random split param")

if __name__=='__main__':
    unittest.main()
