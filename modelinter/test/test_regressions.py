'''
Created on 6 Sep 2017

@author: adrien.papaioannou
'''
import unittest

from modelinter.models.pgm import regression_loop
from modelinter.preprocessing.imports_load_data import read_csvs, extract_arrays


class TestEDA(unittest.TestCase):
    def setUp(self):
        raw_data = read_csvs()
        self.arrays = extract_arrays(raw_data)

    def assertListAlmostEqual(self, list1, list2, tol, msg):
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, tol, msg)

    def test_regression_stocks(self):
        beta, alpha, epsilon, r2 = regression_loop(self.arrays.sp_r, self.arrays.stocks_r)
        expected_beta = [1.34131033, 1.66456114, 0.85578133, 1.0069497, 0.78762373, 1.1167474, 1.09363311, 1.34175063,
                         1.26110543, 1.3077713, 0.98436069, 1.22558994, 1.88128737, 0.3686435, 0.39936583, 1.28921579,
                         1.08473058, 0.84379251, 1.00299131, 1.24853135, 0.70235921]
        expected_alpha = [1.66465813e-04, 3.81073743e-04, -5.14613760e-05, -2.89155417e-05, 1.49015547e-03,
                          1.00584811e-03, -8.35629666e-05, 1.97283590e-04, -6.65059687e-04, -3.95377675e-04,
                          -4.81840876e-04, 1.42685687e-03, 9.18006399e-05, -7.54827246e-04]
        expected_r2 = [5.48443470e-01, 3.50164387e-01, 1.86360830e-01, 3.18604680e-01, 1.16982677e-01, 3.50959646e-01,
                       4.52604627e-01, 5.02251472e-01, 4.87021200e-01, 3.43814531e-01, 5.75192759e-01, 1.74058721e-01,
                       4.84524526e-01, 7.00500174e-02]
        self.assertListAlmostEqual(beta[:len(expected_beta)], expected_beta, tol=5, msg='beta not matching')
        self.assertListAlmostEqual(alpha[:len(expected_alpha)], expected_alpha, tol=5, msg='beta not matching')
        self.assertListAlmostEqual(r2[:len(expected_r2)], expected_r2, tol=5, msg='beta not matching')


if __name__ == "__main__":
    unittest.main()
