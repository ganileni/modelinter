'''
Created on 27 Sep 2017

@author: orazio.angelini
'''
import numpy as np
from py_vollib.black_scholes import black_scholes
from sklearn.covariance import GraphLassoCV, GraphLasso

from modelinter.models.constants import Const, Paths
from modelinter.models.utils import lf, Pkl

tradingyear = 252
annualize = np.sqrt(tradingyear)
riskfree_rate = 0.0025  # assumed constant


def regression_loop(index_returns, security_returns):
    beta, alpha, epsilon, r2 = [], [], [], []
    for j in range(security_returns.shape[1]):
        regression = lf(index_returns, security_returns[:, j])
        beta.append(regression[lf.m])
        alpha.append(regression[lf.q])
        # rsquared of the regression
        r2.append(regression[lf.r2])
        epsilon.append(regression[lf.obj].resid)
    return np.array(beta), np.array(alpha), np.array(epsilon), np.array(r2)


class BasePgmParams():
    """base model for both stocks and options. It encodes
    save/load functionality, and the logic for calculating lasso."""

    def save(self, name=''):
        Pkl.save(self, Paths.SAVE_DIR.value + self.modelname + name + Paths.PKL_EXT.value)

    def load(self, name=''):
        selfload = Pkl.load(Paths.SAVE_DIR.value + self.modelname + name + Paths.PKL_EXT.value)
        for attr in [_ for _ in selfload.__dir__() if not _.startswith('__')]:
            setattr(self, attr, getattr(selfload, attr))

    def estimate_glasso(self, alpha='cv', **kwargs):
        if 'mode' not in kwargs:
            kwargs['mode'] = 'cd'
        # estimate covariance matrix with GLASSO
        self.alpha = alpha
        if alpha == 'cv':
            self.glasso = GraphLassoCV(**kwargs)
        else:
            self.glasso = GraphLasso(alpha=alpha, **kwargs)
        self.glasso.fit(self.residuals)
        self.Sigma_ij = self.glasso.covariance_


class StocksPgmParams(BasePgmParams):
    """This class evaluates and contains the parameters for the stocks PGM."""
    modelname = '_stock_PGM_'

    def __init__(self, dow_t, r_it, alpha='cv', glasso_args=dict()):
        """Evaluate all the parameters needed for stocks."""
        self.beta_i, drop_alpha, self.epsilon_i, drop_r2 = \
            regression_loop(dow_t, r_it)
        # cap the betas at 3? (more than that and you get
        # stocks that go to negative prices on a big shock)
        # self.beta_i[self.beta_i>3] = 3
        # calculate std of epsilon_i
        self.sigma_epsilon_i = np.array([np.std(_) for _ in self.epsilon_i])
        # save residuals in a numpy array for GLASSO
        self.epsilon_i = np.vstack(self.epsilon_i).T
        self.residuals = self.epsilon_i
        # estimate GLASSO with CV
        self.estimate_glasso(alpha=alpha, **glasso_args)
        self.Sigma_ij = self.glasso.covariance_


class StocksPgm:
    """This class takes as an input a StocksPgmParams object
    and is the base for all the subclasses that make predictions."""

    def __init__(self, stocks_PGM_params):
        self.params = stocks_PGM_params

    # feed the index value and get the stocks returns
    # note that the model is tuned on daily returns.
    # if you want the variance for retuns over more than a day
    # you need to scale the variance!
    # returns are invariant instead
    def predict(self, dow_t):
        return dow_t * self.params.beta_i


class StocksPgmA(StocksPgm):
    """This class makes predictions for stocks in PGM A"""

    # days, samples are needed in the signature to standardize interface
    def predict(self, dow_t, days=1, samples=1):
        return super().predict(dow_t)


class StocksPgmB(StocksPgm):
    """This class makes predictions for stocks in PGM B"""

    def predict(self, dow_t, days=1, samples=1):
        res = super().predict(dow_t) \
              + np.random.normal(scale=np.sqrt(days) * self.params.sigma_epsilon_i,
                                 size=(samples, self.params.sigma_epsilon_i.size))
        # flatten output if only one sample was requested.
        if samples == 1:
            return res.flatten()
        else:
            return res


class StocksPgmC(StocksPgm):
    """This class makes predictions for stocks in PGM C"""

    def predict(self, dow_t, days=1, samples=1):
        res = super().predict(dow_t) \
              + np.random.multivariate_normal(
            mean=np.zeros(self.params.Sigma_ij.shape[0]),
            cov=days * self.params.Sigma_ij,
            size=samples)
        # flatten output if only one sample was requested.
        if samples == 1:
            return res.flatten()
        else:
            return res


class options_PGM_params(BasePgmParams):
    """This class evaluates and contains the parameters for the options PGM."""
    modelname = '_option_PGM_'

    # the model will be evaluated by estimating
    # the relation between vix == dow_t and volatility.
    def __init__(self, dow_t, sigma_it, alpha='cv', glasso_args=dict()):
        """Evaluate all the parameters needed for options."""
        self.theta_i, self.eta_i, self.delta_i, drop_r2 = \
            regression_loop(dow_t, sigma_it)
        self.sigma_delta_i = np.array([np.std(_) for _ in self.delta_i])
        self.delta_i = np.vstack(self.delta_i).T
        self.residuals = self.delta_i
        # estimate GLASSO
        self.estimate_glasso(alpha=alpha, **glasso_args)
        self.Sigma_ij = self.glasso.covariance_


class OptionsPgm():
    """This class takes as an input a options_PGM_params object
    and is the base for all the subclasses that make predictions."""

    def __init__(self, options_PGM_params):
        self.params = options_PGM_params

    # feed the index value and get the option prices.
    def predict(self, vix, r, strike_price, stocks_prices, tau_i, flag_i, moneyness_i, samples=1):
        """provides the  base logic for calculating a PGM prediction.
        The only thing that changes between calculation of PGMS A,B and C
        is the way you predict sigma_i, and we put that logic into
        subclasses."""
        # eval daily volatility and ANNUALIZE
        sigma_i = self.sigma_i(vix=vix, samples=samples) * Const.ANNUALIZE.value
        # then predict all the options values:
        options_prices = []
        # optimization: If one wants multiple samples from models B or C
        # it's much faster (~100 times) to draw all of them at the same time from
        # numpy.random rather than calling the predict() method
        # multiple times. This implies a loop over rows of sigma_i though.
        flag_onesample = False
        # this code checks if only 1 sample was requested. in this case we need
        # to increase ndim of sigma_i, and then flatten at the end of the loop.
        if sigma_i.ndim == 1:
            sigma_i = np.array([sigma_i])
            flag_onesample = True
        elif sigma_i.shape[0] == 1:
            flag_onesample = True
        # multiple stocks samples can be fed to the function, but this
        # requires a loop over S_i and strike_price, and also checking its dimensionality
        # and increasing it if it's 1
        if stocks_prices.ndim == 1:
            stocks_prices = np.array([stocks_prices])
            # at this point S_i and sigma_i should have the same number
            # of rows, so that we have a sample of volatilities
            # for each sample of stocks prices
            if stocks_prices.shape[0] != sigma_i.shape[0]:
                raise AssertionError('S_i should have nsamples rows')

        # if K is not given, calculate it fom moneyness:
        if strike_price is None:
            strike_price = stocks_prices / moneyness_i
        # if instead strike_price is given
        else:
            # there should be one strike per stock,
            # so check dimensionality of strike_price as well
            if strike_price.ndim == 1:
                strike_price = np.array([strike_price])
            # check that strike_price and S_i have the same dimensionality
            if strike_price.shape != stocks_prices.shape:
                raise AssertionError('strike_price.shape != S_i.shape')
        # loop over rows of sigma_i
        for SIGMA, STOCKS, STRIKES in zip(sigma_i, stocks_prices, strike_price):
            options_prices_onesample = []
            # loop over each option in the portfolio
            for S, K, s, tau, f in zip(STOCKS, STRIKES, SIGMA, tau_i, flag_i):
                options_prices_onesample.append(
                    black_scholes(flag=f,
                                  S=S,
                                  K=K,
                                  t=tau,
                                  r=r,
                                  sigma=s)
                )
            options_prices.append(options_prices_onesample)
        # flatten output if only one sample was requested.
        if flag_onesample: options_prices = np.array(options_prices).flatten()
        return np.array(options_prices)

    def sigma_i(self, vix):
        return self.params.eta_i + vix * self.params.theta_i


class OptionsPgmA(OptionsPgm):
    """This class makes predictions for options in PGM A"""

    def sigma_i(self, vix, samples=1):
        # samples is needed in the signature, but unused here.
        return super().sigma_i(vix)


class OptionsPgmB(OptionsPgm):
    """This class makes predictions for options in PGM B"""

    def sigma_i(self, vix, samples=1):
        return super().sigma_i(vix) \
               + np.random.normal(scale=self.params.sigma_delta_i,
                                  loc=np.zeros(self.params.sigma_delta_i.size),
                                  size=(samples, self.params.sigma_delta_i.size))


class OptionsPgmC(OptionsPgm):
    """This class makes predictions for options in PGM C"""

    def sigma_i(self, vix, samples=1):
        return super().sigma_i(vix) \
               + np.random.multivariate_normal(
            mean=np.zeros(self.params.Sigma_ij.shape[0]),
            cov=self.params.Sigma_ij,
            size=samples)
