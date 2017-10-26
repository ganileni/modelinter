from enum import Enum
from types import SimpleNamespace
import numpy as np
from os.path import dirname, normpath
import modelinter


class Const(Enum):
    TRADING_YEAR = 252  # length of a trading year
    WHOLE_YEAR = 365 #length of an actual year
    ANNUALIZE = np.sqrt(TRADING_YEAR)  # to ANNUALIZE daily volatility
    N_YEARS_KEEP = 5.5  # previous years of data to keep
    RISK_FREE_RATE = 0.0025  # assumed constant
    SIGMA_WINDOW = 20  # time window for volatility estimation
    # the following parameter is justified by notebook 0-preliminary_analysis.ipynb
    SIGMA_SHIFT = 10 # by how much to delay the rolling volatility estimation


class ConstE(Enum):
    TIMESTAMP = 'timestamp'
    DATE = 'date'
    DATE_FORMAT = '%Y-%m-%d'
    ALT_DATE_FORMAT = '%d-%m-%Y'


class TimeseriesVariablesE(Enum):
    SP500 = 'SP500'
    VIX = 'VIX'

# ugly hack, but it should work on all OSs
absolute = dirname(modelinter.__file__)
class Paths(Enum):
    SAVE_DIR = normpath(absolute + '/resources/data/interim/') + '/'
    FIGURES_DIR = normpath(absolute + '/resources/plots/') + '/'
    DATA_DIR = normpath(absolute + '/resources/data/raw/') + '/'
    FREE_DATA_DIR = normpath(absolute + '/resources/data/processed/free_subset/') + '/'
    PKL_EXT = '.pkl'


Slices = SimpleNamespace(
    # to slice the pandas dataframe for stock data
    stocks_subset=slice(2, None),
    indices_subset=slice(0, 2)
)