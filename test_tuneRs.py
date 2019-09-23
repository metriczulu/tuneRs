import unittest

class TunerTest(unittest.TestCase):

    def setUp(self):
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from tuneRs.space import LogNormal
        self.lr = LogisticRegression(random_state=0)
        np.random.seed(0)
        self.x_train = np.random.rand(80, 2)
        self.x_test = np.random.rand(20, 2)
        self.y_train = np.random.randint(0, 2, 80)
        self.y_test = np.random.randint(0, 2, 20)
        self.grid_C = {'C': [0.1, 1, 10]}
        self.rand_C = {'C': LogNormal(0.1, 10)}

    def test_grid_resampled(self):
        from tuneRs.tuneRs import GridSearchResample
        test_resampled = GridSearchResample(self.lr, self.grid_C, random_state=1)
        test_resampled.fit(self.x_train, self.y_train)
        test_rs_score = int(test_resampled.best_score_ * 10)  # should be 5
        test_rs_params = int(test_resampled.best_params_['C'] * 10)  # should be 100
        print(f"random rs {test_rs_score}, {test_rs_params}")
        self.assertEqual(test_rs_score, 5, "Incorrect grid resample score")
        self.assertEqual(test_rs_params, 100, "Incorrect grid resample param")

    def test_random_resampled(self):
        from tuneRs.tuneRs import RandomSearchResample
        test = RandomSearchResample(self.lr, self.rand_C, n_iter=10, random_state=1)
        test.fit(self.x_train, self.y_train)
        test_score = int(test.best_score_ * 10)  # should be 5
        test_params = int(test.best_params_['C'] * 10)  # should be 36
        print(f"random rs {test_score}, {test_params}")
        self.assertEqual(test_score, 5, "Incorrect random resample score")
        self.assertEqual(test_params, 36, "Incorrect random resample param")

    def test_grid_cv(self):
        from tuneRs.tuneRs import GridSearchCrossval
        test_resampled = GridSearchCrossval(self.lr, self.grid_C, random_state=1)
        test_resampled.fit(self.x_train, self.y_train)
        test_rs_score = int(test_resampled.best_score_ * 10)  # should be 5
        test_rs_params = int(test_resampled.best_params_['C'] * 10)  # should be 1
        print(f"random rs {test_rs_score}, {test_rs_params}")
        self.assertEqual(test_rs_score, 5, "Incorrect grid cv score")
        self.assertEqual(test_rs_params, 1, "Incorrect grid cv param")

    def test_random_cv(self):
        from tuneRs.tuneRs import RandomSearchCrossval
        test = RandomSearchCrossval(self.lr, self.rand_C, n_iter=10, random_state=1)
        test.fit(self.x_train, self.y_train)
        test_score = int(test.best_score_ * 10)  # should be 5
        test_params = int(test.best_params_['C'] * 10)  # should be 2
        print(f"random rs {test_score}, {test_params}")
        self.assertEqual(test_score, 5, "Incorrect random cv score")
        self.assertEqual(test_params, 2, "Incorrect random cv param")

    def test_grid_simple(self):
        from tuneRs.tuneRs import GridSearchSimple
        test_resampled = GridSearchSimple(self.lr, self.grid_C, val_set=(self.x_test, self.y_test), random_state=1)
        test_resampled.fit(self.x_train, self.y_train)
        test_rs_score = int(test_resampled.best_score_ * 10)  # should be 6
        test_rs_params = int(test_resampled.best_params_['C'] * 10)  # should be 1
        print(f"random rs {test_rs_score}, {test_rs_params}")
        self.assertEqual(test_rs_score, 6, "Incorrect grid simple score")
        self.assertEqual(test_rs_params, 1, "Incorrect grid simple param")

    def test_random_simple(self):
        from tuneRs.tuneRs import RandomSearchSimple
        test = RandomSearchSimple(self.lr, self.rand_C, val_set=(self.x_test, self.y_test), n_iter=10, random_state=1)
        test.fit(self.x_train, self.y_train)
        test_score = int(test.best_score_ * 10)  # should be 6
        test_params = int(test.best_params_['C'] * 10)  # should be 2
        print(f"random rs {test_score}, {test_params}")
        self.assertEqual(test_score, 6, "Incorrect random simple score")
        self.assertEqual(test_params, 2, "Incorrect random simple param")


if __name__=='__main__':
    unittest.main()
