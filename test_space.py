import unittest

class SpaceTest(unittest.TestCase):

    def test_uniform(self):
        from tuneRs.space import Uniform
        test = Uniform(0, 20, dtype="int")
        truth_test = [(0<=i<=20) for i in test.rvs(10)]
        self.assertNotIn(False, truth_test, "Int distribution out of range.")
        test = Uniform(0, 20, dtype="float")
        truth_test = [(0 <= i <= 20) for i in test.rvs(10)]
        self.assertNotIn(False, truth_test, "Float distribution out of range.")
        test = Uniform(0, 20, dtype="float32")
        truth_test = [(0 <= i <= 20) for i in test.rvs(10)]
        self.assertNotIn(False, truth_test, "Float32 distribution out of range.")

    def test_normal(self):
        from tuneRs.space import Normal
        test = Normal(0, 5, min=-5, max=10, dtype="float")
        truth_test = [(-5<=i<=10) for i in test.rvs(10)]
        self.assertNotIn(False, truth_test, "Float distribution out of range.")
        test = Normal(0, 5, min=-5, max=10, dtype="int")
        truth_test = [(-5<=i<=10) for i in test.rvs(10)]
        self.assertNotIn(False, truth_test, "Int distribution out of range.")
        test = Normal(0, 5, min=-5, max=10, dtype="float32")
        truth_test = [(-5<=i<=10) for i in test.rvs(10)]
        self.assertNotIn(False, truth_test, "Float32 distribution out of range.")

    def test_log_normal(self):
        from tuneRs.space import LogNormal
        test = LogNormal(1, 1000, dtype="int")
        truth_test = [(1<=i<=1000) for i in test.rvs(10)]
        self.assertNotIn(False, truth_test, "Int distribution out of range.")
        test = LogNormal(1, 1000, dtype="float")
        truth_test = [(1<=i<=1000) for i in test.rvs(10)]
        self.assertNotIn(False, truth_test, "Float distribution out of range.")
        test = LogNormal(1, 1000, dtype="float32")
        truth_test = [(1<=i<=1000) for i in test.rvs(10)]
        self.assertNotIn(False, truth_test, "Float32 distribution out of range.")

if __name__=='__main__':
    unittest.main()
