from types import SimpleNamespace

import numpy as np
import pandas as pd
from tqdm import tqdm

from modelinter.models.constants import Const, Paths
from modelinter.models.pgm import StocksPgmParams, StocksPgmA, StocksPgmB, StocksPgmC, options_PGM_params, \
    OptionsPgmA, OptionsPgmB, OptionsPgmC
from modelinter.models.utils import returns, inverse_returns, countdays, Pkl
from modelinter.preprocessing.imports_load_data import extract_arrays_variable_length

GlobalSettings = SimpleNamespace(recalculate_results = False)

DefaultSettings = SimpleNamespace( # default options for the scenario we evaluate in the paper
    flags=lambda N: N * ['c'],  # all the options are calls
    taus=lambda N: np.array([5] * N),  # five years options
    strike_price=None,  # won't specify directly strike price...
    moneyness=1.1,  # ...but indirectly through moneyness
    samples=int(1e5),  # number of samples to extract
    dow_t0=18308.15,  # value of dow jones at start of CCAR scenario Q4 2016
    vix_t0=13.29,  # value of VIX at start of CCAR scenario Q4 2016
    t0=pd.to_datetime('01-10-2016', dayfirst=True),  # first day of scenario
    alpha='cv',  # penalty for GLASSO
    recalculate_results=GlobalSettings.recalculate_results,
    save_results=True
)

DefaultSettings_testdays = SimpleNamespace(
    # default options for the test where we change
    # the number of days used to train the model.
    flags=lambda N: N * ['c'],  # all the options are calls
    taus=lambda N: np.array([5] * N),  # five years options
    strike_price=None,  # won't specify directly strike price...
    moneyness=1.1,  # ...but indirectly through moneyness
    samples=int(1e4),  # number of samples to extract
    dow_t0=18308.15,  # value of dow jones at start of CCAR scenario Q4 2016
    vix_t0=13.29,  # value of VIX at start of CCAR scenario Q4 2016
    t0=pd.to_datetime('01-10-2016', dayfirst=True),  # first day of scenario
    alpha=0.1,  # penalty for GLASSO
    recalculate_results=GlobalSettings.recalculate_results,
    save_results=True
)


def eval_PGM(arrays, DefaultSettings):
    """evalutes parameters for all PGM models from data"""
    PGM_stock_params = StocksPgmParams(arrays.sp_r,
                                       arrays.stocks_r,
                                       alpha=DefaultSettings.alpha)
    PGM_stock_A = StocksPgmA(PGM_stock_params)
    PGM_stock_B = StocksPgmB(PGM_stock_params)
    PGM_stock_C = StocksPgmC(PGM_stock_params)
    PGM_options_params = options_PGM_params(arrays.vix,
                                            arrays.sigma,
                                            alpha=DefaultSettings.alpha)
    PGM_options_A = OptionsPgmA(PGM_options_params)
    PGM_options_B = OptionsPgmB(PGM_options_params)
    PGM_options_C = OptionsPgmC(PGM_options_params)
    return SimpleNamespace(
        PGM_stock_params=PGM_stock_params,
        PGM_stock_A=PGM_stock_A,
        PGM_stock_B=PGM_stock_B,
        PGM_stock_C=PGM_stock_C,
        PGM_options_params=PGM_options_params,
        PGM_options_A=PGM_options_A,
        PGM_options_B=PGM_options_B,
        PGM_options_C=PGM_options_C,
    )


def extract_scenario(arrays, raw_data, options_model, settings):
    """calculate/extract values of indices for CCAR scenario"""
    N = arrays.stocks_r.shape[1]
    # get indices values
    vix_t = raw_data.fed_sev_adverse['market_volatility_index'].values
    dow_t = raw_data.fed_sev_adverse['dow_jones_total_stock_market_index'].values
    t = raw_data.fed_sev_adverse.index.values
    # add starting values
    vix_t = np.insert(vix_t, 0, settings.vix_t0)
    dow_t = np.insert(dow_t, 0, settings.dow_t0)
    t = np.insert(t, 0, settings.t0)
    # SP500 returns
    dow_t = returns(settings.dow_t0, dow_t)
    # we'll also need the prices for stock at time t=0
    stock_prices_t0 = arrays.stocks_p[-1, :]
    # we'll need the prices of options at time t=0
    options_prices_t0 = options_pricing(vix_t=vix_t[0], stock_prices_t=stock_prices_t0, model=options_model, days=0,
                                        samples=1, settings=DefaultSettings)
    return SimpleNamespace(N=N,
                           t=t,
                           dow_t=dow_t,
                           vix_t=vix_t,
                           stock_prices_t0=stock_prices_t0,
                           options_prices_t0=options_prices_t0)


def options_pricing(vix_t, stock_prices_t, model, days, samples, settings):
    """function that hard-codes the values and settings
    to use in the paper's calculation"""
    # we'll use standard values for all the options.
    # Let's hard-code them in a function.
    # note that tau -time to expiration- decreases over time!
    # calculate the remaining time at given date
    yearsfraction = settings.taus(model.params.eta_i.size) \
                    - days / Const.TRADING_YEAR.value
    return model.predict(vix=vix_t, r=Const.RISK_FREE_RATE.value, strike_price=settings.strike_price,
                         stocks_prices=stock_prices_t, tau_i=yearsfraction,
                         flag_i=settings.flags(model.params.eta_i.size), moneyness_i=settings.moneyness,
                         samples=samples)


def stocks_pricing(dow_t, model, days, samples, scenario):
    """prices the stock portfolio"""
    # the stocks model outputs returns. Let's convert them to prices.
    r_it = model.predict(dow_t=dow_t, days=days, samples=samples)
    # to prices
    results = inverse_returns(scenario.stock_prices_t0, r_it)
    # if any of the stocks went negative, set its price to 1 cent
    # PROBLEM: this skews the exp val of the distribution
    # of stock portfolio prices in models B & C!
    results[results <= 0] = 0.01
    return results


def price_portfolio(dow_t, vix_t, model_stock, model_options, date, weights, scenario, samples=1):
    """prices the whole combined portfolio of stocks and options"""
    # let's make a function that gives us the value of the portfolio
    # at a certain date.
    days = countdays(scenario.t[0], date)
    # predict stocks
    stocks = stocks_pricing(dow_t, model_stock, days, samples=samples, scenario=scenario)
    # predict and weight options
    options = weights.options_weights * options_pricing(vix_t=vix_t, stock_prices_t=stocks, model=model_options, days=days,
                                              samples=samples, settings=DefaultSettings)
    # weight stocks
    stocks = weights.stock_weights * stocks
    if samples == 1:
        return stocks, options
    else:
        return [_ for _ in stocks], [_ for _ in options]


def calc_weights(stock_prices_t0, options_prices_t0):
    # portfolio weighting
    # for each option/stock pair, the weight will be inversely proportional
    # to the price, and normalized so that they add up to 100
    stock_weights = 100 / ((stock_prices_t0 + options_prices_t0) * stock_prices_t0.size)
    options_weights = stock_weights
    return SimpleNamespace(stock_weights=stock_weights,
                           options_weights=options_weights)


def sample_results(scenario, weights, models, sampling, progressbar=True):
    # L^s according to models A, B, and C
    portfolio_stocks_pgm_a, portfolio_stocks_pgm_b, portfolio_stocks_pgm_c = [], [], []
    # L^o according to models A, B, and C
    portfolio_options_pgm_a, portfolio_options_pgm_b, portfolio_options_pgm_c = [], [], []
    # one option-stock pair
    one_s, one_o = [], []
    # loop over quarters
    if progressbar:
        wrapper = tqdm
        kwargs = {'desc':'all', 'total':len(scenario.t)}
    else:
        wrapper = lambda x: x
        kwargs = {}
    for time, I, J in wrapper(zip(scenario.t, scenario.dow_t, scenario.vix_t),
                           **kwargs):
        # calculate value of the portfolio for model a
        a, b = price_portfolio(I, J, model_stock=models.PGM_stock_A, model_options=models.PGM_options_A, date=time,
                               weights=weights, scenario=scenario, samples=1)
        portfolio_stocks_pgm_a.append(np.sum(a))
        portfolio_options_pgm_a.append(np.sum(b))
        # sample multiple values for the portfolios from models B and C
        a, b = price_portfolio(I, J, model_stock=models.PGM_stock_B, model_options=models.PGM_options_B, date=time,
                               weights=weights, scenario=scenario, samples=sampling)
        portfolio_stocks_pgm_b.append(np.array([np.sum(_) for _ in a]))
        portfolio_options_pgm_b.append(np.array([np.sum(_) for _ in b]))
        a, b = price_portfolio(I, J, model_stock=models.PGM_stock_C, model_options=models.PGM_options_C, date=time,
                               weights=weights, scenario=scenario, samples=sampling)
        portfolio_stocks_pgm_c.append(np.array([np.sum(_) for _ in a]))
        portfolio_options_pgm_c.append(np.array([np.sum(_) for _ in b]))
        # look at 1000 samples each for first 50 option-stock pair
        one_s.append([_[:50] for _ in a[:1000]])
        one_o.append([_[:50] for _ in b[:1000]])

    return SimpleNamespace(portfolio_stocks_pgm_a=portfolio_stocks_pgm_a,
                           portfolio_stocks_pgm_b=portfolio_stocks_pgm_b,
                           portfolio_stocks_pgm_c=portfolio_stocks_pgm_c,
                           portfolio_options_pgm_a=portfolio_options_pgm_a,
                           portfolio_options_pgm_b=portfolio_options_pgm_b,
                           portfolio_options_pgm_c=portfolio_options_pgm_c,
                           one_s=one_s,
                           one_o=one_o)


def sample_day(dc, raw_data, DefaultSettings):
    """samples the results of the model when estimating it using
    1year + dc days"""
    arrays = extract_arrays_variable_length(raw_data, dc=dc)
    PGM = eval_PGM(arrays, DefaultSettings)
    scenario = extract_scenario(arrays=arrays,
                                raw_data=raw_data,
                                options_model=PGM.PGM_options_A,
                                settings=DefaultSettings)
    weights = calc_weights(scenario.stock_prices_t0, scenario.options_prices_t0)
    results = sample_results(scenario, weights, PGM,
                             DefaultSettings.samples,
                             progressbar=False)
    out = SimpleNamespace(
        stock_meanA=np.array([np.mean(_) for _ in results.portfolio_stocks_pgm_a]),
        stock_meanB=np.array([np.mean(_) for _ in results.portfolio_stocks_pgm_b]),
        stock_stdB=3 * np.array([np.std(_) for _ in results.portfolio_stocks_pgm_b]),
        stock_meanC=np.array([np.mean(_) for _ in results.portfolio_stocks_pgm_c]),
        stock_stdC=3 * np.array([np.std(_) for _ in results.portfolio_stocks_pgm_c]),

        options_meanA=np.array([np.mean(_) for _ in results.portfolio_options_pgm_a]),
        options_meanB=np.array([np.mean(_) for _ in results.portfolio_options_pgm_b]),
        options_stdB=3 * np.array([np.std(_) for _ in results.portfolio_options_pgm_b]),
        options_meanC=np.array([np.mean(_) for _ in results.portfolio_options_pgm_c]),
        options_stdC=3 * np.array([np.std(_) for _ in results.portfolio_options_pgm_c]),

        both_meanA=np.array([np.mean(_) + np.mean(__) for _, __ in zip(results.portfolio_options_pgm_a, results.portfolio_stocks_pgm_a)]),
        both_meanB=np.array([np.mean(_) + np.mean(__) for _, __ in zip(results.portfolio_options_pgm_b, results.portfolio_stocks_pgm_b)]),
        both_stdB=3 * np.array([np.std(_) + np.std(__) for _, __ in zip(results.portfolio_options_pgm_b, results.portfolio_stocks_pgm_b)]),
        both_meanC=np.array([np.mean(_) + np.mean(__) for _, __ in zip(results.portfolio_options_pgm_c, results.portfolio_stocks_pgm_c)]),
        both_stdC=3 * np.array([np.std(_) + np.std(__) for _, __ in zip(results.portfolio_options_pgm_c, results.portfolio_stocks_pgm_c)])
    )
    return out


def sample_day_loop(raw_data, DefaultSettings, days_change, progressbar=True):
    allresults = []
    n = 0
    if progressbar:
        wrapper = tqdm
        kwargs = {'desc':'all'}
    else:
        wrapper = lambda x: x
        kwargs = {}
    for dc in wrapper(days_change, **kwargs):
        allresults.append(sample_day(dc, raw_data, DefaultSettings))
        n += 1
        if n % 2 == 0:
            Pkl.save(allresults, Paths.SAVE_DIR.value + Paths.PKL_EXT.value)
    return allresults