'''
Created on 7 Sep 2017

@author: orazio.angelini
'''
import unittest

import numpy as np

from modelinter.models.calculations import DefaultSettings, eval_PGM
from modelinter.models.constants import Const
from modelinter.models.pgm import StocksPgmA, StocksPgmB, StocksPgmC, OptionsPgmA, OptionsPgmB, OptionsPgmC
from modelinter.preprocessing.imports_load_data import read_csvs, extract_arrays


class Test(unittest.TestCase):

    def setUp(self):
        #load data, eval pgm parameters
        raw_data = read_csvs()
        self.arrays = extract_arrays(raw_data)
        DefaultSettings.alpha = 0.1
        self.PGM = eval_PGM(self.arrays, DefaultSettings)

    def assertListAlmostEqual(self, list1, list2, tol, msg):
        for a, b in zip(list1, list2):
            self.assertAlmostEqual(a, b, tol, msg)

    def test_regression_stocks(self):
        #do calculations
        PGM_stock_A = StocksPgmA(self.PGM.PGM_stock_params)
        PGM_stock_B = StocksPgmB(self.PGM.PGM_stock_params)
        PGM_stock_C = StocksPgmC(self.PGM.PGM_stock_params)
        PGM_stock_prediction_A = PGM_stock_A.predict(.01)
        np.random.seed(1)
        PGM_stock_prediction_B = PGM_stock_B.predict(.01)
        np.random.seed(1)
        PGM_stock_prediction_C = PGM_stock_C.predict(.01)
        PGM_options_A = OptionsPgmA(self.PGM.PGM_options_params)
        PGM_options_B = OptionsPgmB(self.PGM.PGM_options_params)
        PGM_options_C = OptionsPgmC(self.PGM.PGM_options_params)
        PGM_options_prediction_A = PGM_options_A.predict(vix=np.mean(self.arrays.vix), r=Const.RISK_FREE_RATE.value,
                                                         strike_price=None, stocks_prices=self.arrays.stocks_p[-1],
                                                         tau_i=[1] * self.arrays.stocks_p.shape[-1],
                                                         flag_i=['p'] * self.arrays.stocks_p.shape[-1], moneyness_i=1.1)
        np.random.seed(1)
        PGM_options_prediction_B = PGM_options_B.predict(vix=np.mean(self.arrays.vix), r=Const.RISK_FREE_RATE.value,
                                                         strike_price=None, stocks_prices=self.arrays.stocks_p[-1],
                                                         tau_i=[1] * self.arrays.stocks_p.shape[-1],
                                                         flag_i=['p'] * self.arrays.stocks_p.shape[-1], moneyness_i=1.1)
        np.random.seed(1)
        PGM_options_prediction_C = PGM_options_C.predict(vix=np.mean(self.arrays.vix), r=Const.RISK_FREE_RATE.value,
                                                         strike_price=None, stocks_prices=self.arrays.stocks_p[-1],
                                                         tau_i=[1] * self.arrays.stocks_p.shape[-1],
                                                         flag_i=['p'] * self.arrays.stocks_p.shape[-1], moneyness_i=1.1)
        #test that predictions are ok
        PGM_stock_prediction_A_expected = [0.0134131, 0.01664561,0.00855781,0.0100695, 0.00787624,0.01116747,0.01093633,0.01341751,0.01261105,0.01307771]
        PGM_stock_prediction_B_expected = [0.02973448, 0.00519304, 0.00076067, -0.002975, 0.0233367, -0.01768876, 0.02826122, 0.00502333, 0.01602008, 0.00935819]
        PGM_stock_prediction_C_expected = [0.0115625,0.02432211,0.02311861,0.01326042,0.00925791,0.03545982,0.02849107,0.01079361,0.02340456,-0.00179359]
        PGM_options_prediction_A_expected = [2.25637023,4.21749353,9.15615605,5.25452619,4.56891578,1.84272017,4.57236283,5.06469782,3.39962771,2.96651475]
        PGM_options_prediction_B_expected = [3.41618907,3.02847777,6.37033964,2.62421919,8.33572172,0.05093847,8.17066613,3.18456136,3.9275443,2.69208378]
        PGM_options_prediction_C_expected = [1.92584076,8.50740935,8.1899018,7.00520731,6.74419429,2.93409453,4.18522997,1.88011665,0.87244535,3.10930139]
        self.assertListAlmostEqual(PGM_stock_prediction_A[:10].tolist(),
                                   PGM_stock_prediction_A_expected,
                                   tol=3,
                                   msg='stocks PGM A prediction not matching')
        self.assertListAlmostEqual(PGM_options_prediction_A[:10].tolist(),
                                   PGM_options_prediction_A_expected,
                                   tol=3,
                                   msg='options PGM A prediction not matching')
        self.assertListAlmostEqual(PGM_stock_prediction_B[:10].tolist(),
                                   PGM_stock_prediction_B_expected,
                                   tol=3,
                                   msg='stocks PGM B prediction not matching')
        self.assertListAlmostEqual(PGM_options_prediction_B[:10].tolist(),
                                   PGM_options_prediction_B_expected,
                                   tol=3,
                                   msg='options PGM B prediction not matching')
        self.assertListAlmostEqual(PGM_stock_prediction_C[:10].tolist(),
                                   PGM_stock_prediction_C_expected,
                                   tol=3,
                                   msg='stocks PGM C prediction not matching')
        self.assertListAlmostEqual(PGM_options_prediction_C[:10].tolist(),
                                   PGM_options_prediction_C_expected,
                                   tol=3,
                                   msg='options PGM C prediction not matching')
if __name__ == "__main__":
    unittest.main()













