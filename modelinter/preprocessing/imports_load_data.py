#### imports ####

import matplotlib
import pandas as pd

from modelinter.models.constants import Const, ConstE, TimeseriesVariablesE, Paths, Slices

matplotlib.use('nbagg')
from types import SimpleNamespace


def process_timeseries(load_path):
    timeseries = pd.read_csv(load_path)
    timeseries.timestamp = pd.to_datetime(timeseries.timestamp, format=ConstE.DATE_FORMAT.value)
    timeseries = timeseries.set_index(ConstE.TIMESTAMP.value)
    # move sp500 and vix to first columns, then all stocks sorted alphabetically
    timeseries = timeseries[[TimeseriesVariablesE.SP500.value, TimeseriesVariablesE.VIX.value]
                            + list(sorted([_ for _ in timeseries.columns.tolist() if (
        _ != TimeseriesVariablesE.SP500.value and _ != TimeseriesVariablesE.VIX.value)]))]
    return timeseries


def process_timeseries_returns(load_path):
    timeseries_returns = pd.read_csv(load_path)
    timeseries_returns.timestamp = pd.to_datetime(timeseries_returns.timestamp, format=ConstE.DATE_FORMAT.value)
    timeseries_returns = timeseries_returns.set_index(ConstE.TIMESTAMP.value)
    # move sp500 and vix to first columns, then all stocks sorted alphabetically
    timeseries_returns = timeseries_returns[[TimeseriesVariablesE.SP500.value, TimeseriesVariablesE.VIX.value]
                                            + list(sorted([_ for _ in timeseries_returns.columns.tolist() if (
        _ != TimeseriesVariablesE.SP500.value and _ != TimeseriesVariablesE.VIX.value)]))]
    return timeseries_returns


def process_fed_sev_adverse(load_path):
    fed_sev_adverse = pd.read_csv(load_path)
    # convert dates and set dataframe indices
    fed_sev_adverse.columns = [_.lower().replace(' ', '_') for _ in fed_sev_adverse.columns]
    fed_sev_adverse.date = pd.to_datetime(fed_sev_adverse.date, format=ConstE.ALT_DATE_FORMAT.value)
    fed_sev_adverse = fed_sev_adverse.set_index(ConstE.DATE.value)
    # make mapping between tickers and stocks names
    return fed_sev_adverse


def process_longnames(load_path):
    # mapping from ticker symbol to stock long name
    longname = pd.read_csv(load_path)
    longname = {_[0]: _[1] for _ in longname.values}
    return longname


def mapping_tickers(load_path):
    # make mapping between tickers and stocks names
    longname = pd.read_csv(load_path)
    longname = {_[0]: _[1] for _ in longname.values}
    return longname


def read_csvs():
    timeseries = process_timeseries(Paths.FREE_DATA_DIR.value + 'timeseries.csv')
    timeseries_returns = process_timeseries_returns(Paths.FREE_DATA_DIR.value + 'timeseries_returns.csv')
    fed_sev_adverse = process_fed_sev_adverse(Paths.DATA_DIR.value + 'FedSeverelyAdverse.csv')
    longname = process_longnames(Paths.DATA_DIR.value + 'sp500_constituents.csv')
    return SimpleNamespace(
        timeseries=timeseries,
        timeseries_returns=timeseries_returns,
        fed_sev_adverse=fed_sev_adverse,
        longname=longname
    )


def extract_arrays(raw_data):
    # extract numpy arrays
    # slice for last year before CCAR + SIGMA_WINDOW
    firstday = (raw_data.fed_sev_adverse.index[0]
                - pd.Timedelta(Const.WHOLE_YEAR.value, 'd')
                - pd.Timedelta(Const.SIGMA_WINDOW.value, 'd'))
    all_time = slice(firstday,
                     raw_data.fed_sev_adverse.index[0])
    # slice foractual time - only last year and not the first month we need for the volatility
    time = slice(-Const.TRADING_YEAR.value, None)
    # stocks returns
    stocks_r = raw_data.timeseries_returns.loc[all_time, :].iloc[:, Slices.stocks_subset].values
    # stocks prices
    stocks_p = raw_data.timeseries.loc[all_time, :].iloc[:, Slices.stocks_subset].values
    # S&P500 returns
    sp_r = raw_data.timeseries_returns.loc[all_time, :].iloc[:, Slices.indices_subset][TimeseriesVariablesE.SP500.value].values
    # VIX, note that we scale it by 10 business days ( == 2 weeks)
    # according to preliminary analysis findings
    vix = raw_data.timeseries.loc[all_time, :].iloc[:, Slices.indices_subset][TimeseriesVariablesE.VIX.value].shift(10).values
    # rolling window daily standard deviation
    sigma = (raw_data.timeseries_returns
             .loc[all_time, :]
             .iloc[:, Slices.stocks_subset]
             .rolling(center=False, window=Const.SIGMA_WINDOW.value)
             .std()
             .iloc[time, :]
             .values)
    # cut off the first SIGMA_WINDOW days we don't need
    stocks_r = stocks_r[time, :]
    stocks_p = stocks_p[time, :]
    sp_r = sp_r[time]
    vix = vix[time]
    return SimpleNamespace(
        time=time, #to slice dataframes in raw_data
        stocks_r=stocks_r, #stocks returns
        stocks_p=stocks_p, #stocks prices
        sp_r=sp_r, #s&p500 returns
        vix=vix, #vix levels
        sigma=sigma) #30-days rolling standard deviation shifted by 10 days

def extract_arrays_variable_length(raw_data, dc=0):
    """extracts array neded for calculation from the object produced by read_csvs().
    dc == "days (number) change"
    by default the function extracts one year before the start of CCAR scenario,
    you can add or remove days by making dc respectively positive or negative"""
    # slice for last N days before CCAR + SIGMA_WINDOW
    #this is the first day we need in the array
    firstday = (raw_data.fed_sev_adverse.index[0]  #when CCAR starts
                - pd.Timedelta(Const.WHOLE_YEAR.value, 'd')  #minus a year
                - pd.Timedelta(Const.SIGMA_WINDOW.value, 'd')  #minus a month to estimate sigma
                - pd.Timedelta(dc, 'd')) #minus dc arbitrary days

    # slice for last year before CCAR + SIGMA_WINDOW
    all_time = slice(firstday,
                     raw_data.fed_sev_adverse.index[0])

    # convert to business days
    busdays = raw_data.timeseries_returns.loc[all_time, :].shape[0]
    # extract numpy arrays
    # slice for actual time - only last year and not the first month we need for the volatility
    time = slice(-busdays, None)
    # stocks returns
    stocks_r = raw_data.timeseries_returns.loc[all_time, :].iloc[:, Slices.stocks_subset].values
    # stocks prices
    stocks_p = raw_data.timeseries.loc[all_time, :].iloc[:, Slices.stocks_subset].values
    # S&P500 returns
    sp_r = raw_data.timeseries_returns.loc[all_time, :].iloc[:, Slices.indices_subset][TimeseriesVariablesE.SP500.value].values
    # VIX, note that we scale it by 10 business days ( == 2 weeks)
    # according to preliminary analysis findings
    vix = raw_data.timeseries.loc[all_time, :].iloc[:, Slices.indices_subset][TimeseriesVariablesE.VIX.value].shift(Const.SIGMA_SHIFT.value).values
    # rolling window daily standard deviation
    sigma = (raw_data.timeseries_returns
             .loc[all_time, :]
             .iloc[:, Slices.stocks_subset]
             .rolling(center=False, window=Const.SIGMA_WINDOW.value)
             .std()
             .iloc[time, :]
             .values)
    # cut off the first SIGMA_WINDOW days we don't need
    stocks_r = stocks_r[time, :]
    stocks_p = stocks_p[time, :]
    sp_r = sp_r[time]
    vix = vix[time]
    return SimpleNamespace(
        stocks_r=stocks_r,
        stocks_p=stocks_p,
        sp_r=sp_r,
        vix=vix,
        sigma=sigma)


if __name__ == '__main__':
    print('import_load_data')
