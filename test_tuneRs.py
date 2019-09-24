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

    def test_resampled(self):
        from tuneRs.tuners import ResampleSearch
        test_resampled = ResampleSearch(self.lr, grid_params=self.grid_C, random_state=1)
        test_resampled.fit(self.x_train, self.y_train)
        test_rs_score = int(test_resampled.best_score_ * 10)  # should be 5
        test_rs_params = int(test_resampled.best_params_['C'] * 10)  # should be 100
        print(f"random rs {test_rs_score}, {test_rs_params}")
        self.assertEqual(test_rs_score, 5, "Incorrect grid resample score")
        self.assertEqual(test_rs_params, 100, "Incorrect grid resample param")

        test2 = ResampleSearch(self.lr, random_params=self.rand_C, n_random=10, random_state=1)
        test2.fit(self.x_train, self.y_train)
        test2_score = int(test2.best_score_ * 10)  # should be 5
        test2_params = int(test2.best_params_['C'] * 10)  # should be 49
        print(f"random rs {test2_score}, {test2_params}")
        self.assertEqual(test2_score, 5, "Incorrect random resample score")
        self.assertEqual(test2_params, 49, "Incorrect random resample param")

    def test_cv(self):
        from tuneRs.tuners import CrossvalSearch
        test_resampled = CrossvalSearch(self.lr, grid_params=self.grid_C, random_state=1)
        test_resampled.fit(self.x_train, self.y_train)
        test_rs_score = int(test_resampled.best_score_ * 10)  # should be 5
        test_rs_params = int(test_resampled.best_params_['C'] * 10)  # should be 1
        print(f"random rs {test_rs_score}, {test_rs_params}")
        self.assertEqual(test_rs_score, 5, "Incorrect grid cv score")
        self.assertEqual(test_rs_params, 1, "Incorrect grid cv param")

        test2 = CrossvalSearch(self.lr, random_params=self.rand_C, n_random=10, random_state=1)
        test2.fit(self.x_train, self.y_train)
        test2_score = int(test2.best_score_ * 10)  # should be 5
        test2_params = int(test2.best_params_['C'] * 10)  # should be 1
        print(f"random rs {test2_score}, {test2_params}")
        self.assertEqual(test2_score, 5, "Incorrect random cv score")
        self.assertEqual(test2_params, 1, "Incorrect random cv param")

    def test_simple(self):
        from tuneRs.tuners import SimpleSearch
        test_resampled = SimpleSearch(self.lr, grid_params=self.grid_C, val_set=(self.x_test, self.y_test),
                                          random_state=1)
        test_resampled.fit(self.x_train, self.y_train)
        test_rs_score = int(test_resampled.best_score_ * 10)  # should be 6
        test_rs_params = int(test_resampled.best_params_['C'] * 10)  # should be 1
        print(f"random rs {test_rs_score}, {test_rs_params}")
        self.assertEqual(test_rs_score, 6, "Incorrect grid simple score")
        self.assertEqual(test_rs_params, 1, "Incorrect grid simple param")

        test2 = SimpleSearch(self.lr, random_params=self.rand_C, val_set=(self.x_test, self.y_test), n_random=10,
                                  random_state=1)
        test2.fit(self.x_train, self.y_train)
        test2_score = int(test2.best_score_ * 10)  # should be 6
        test2_params = int(test2.best_params_['C'] * 10)  # should be 23
        print(f"random rs {test2_score}, {test2_params}")
        self.assertEqual(test2_score, 6, "Incorrect random simple score")
        self.assertEqual(test2_params, 23, "Incorrect random simple param")

        test3_resampled = SimpleSearch(self.lr, grid_params=self.grid_C, val_set=0.3, random_state=1)
        test3_resampled.fit(self.x_train, self.y_train)
        test3_rs_score = int(test3_resampled.best_score_ * 10)  # should be 6
        test3_rs_params = int(test3_resampled.best_params_['C'] * 10)  # should be 100
        print(f"random rs {test3_rs_score}, {test3_rs_params}")
        self.assertEqual(test3_rs_score, 6, "Incorrect grid2 simple score")
        self.assertEqual(test3_rs_params, 100, "Incorrect grid2 simple param")

        test4 = SimpleSearch(self.lr, random_params=self.rand_C, val_set=0.3, n_random=10,
                                  random_state=1)
        test4.fit(self.x_train, self.y_train)
        test4_score = int(test4.best_score_ * 10)  # should be 6
        test4_params = int(test4.best_params_['C'] * 10)  # should be 31
        print(f"random rs {test4_score}, {test2_params}")
        self.assertEqual(test4_score, 6, "Incorrect random2 simple score")
        self.assertEqual(test4_params, 31, "Incorrect random2 simple param")

        test5_resampled = SimpleSearch(self.lr, grid_params=self.grid_C, random_state=1)
        test5_resampled.fit(self.x_train, self.y_train)
        test5_rs_score = int(test5_resampled.best_score_ * 10)  # should be 6
        test5_rs_params = int(test5_resampled.best_params_['C'] * 10)  # should be 1
        print(f"random rs {test5_rs_score}, {test5_rs_params}")
        self.assertEqual(test5_rs_score, 6, "Incorrect grid3 simple score")
        self.assertEqual(test5_rs_params, 1, "Incorrect grid3 simple param")

        test6 = SimpleSearch(self.lr, random_params=self.rand_C, n_random=10, random_state=1)
        test6.fit(self.x_train, self.y_train)
        test6_score = int(test6.best_score_ * 10)  # should be 6
        test6_params = int(test6.best_params_['C'] * 10)  # should be 4
        print(f"random rs {test6_score}, {test6_params}")
        self.assertEqual(test6_score, 6, "Incorrect random3 simple score")
        self.assertEqual(test6_params, 4, "Incorrect random3 simple param")

if __name__=='__main__':
    unittest.main()
