import pickle
from inspect import getsourcelines, getfile

import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt

from modelinter.models.constants import Paths


def returns(a_0, a_1):
    """linear returns formula"""
    return (a_1 - a_0) / a_0


def inverse_returns(C, r):
    "invert the returns formula"
    return C * (1 + r)


def lf(X, Y, missing='drop', zeroc=True, finiteonly=False, **kwargs):
    """lf stands for "linear fit".
    returns a dictionary with results for the regression X ~ m*Y + q
    if zeroc=False, q is set to 0."""

    X = np.array(X)
    Y = np.array(Y)
    if finiteonly:
        keep = np.isfinite(X) & np.isfinite(Y)
        X = X[keep]
        Y = Y[keep]
    Xc = sm.add_constant(X) if zeroc else X
    mod = sm.OLS(endog=Y, exog=Xc, missing=missing, **kwargs)
    res = mod.fit()
    result = {lf.r2: res.rsquared,
              lf.m: res.params[1],
              lf.q: res.params[0],
              lf.stderr_m: res.bse[1],
              lf.pval_m: res.pvalues[1],
              lf.stderr_q: res.bse[0],
              lf.pval_q: res.pvalues[0],
              lf.obj: res,
              lf.df: len(X),
              lf.stderr_y: res.mse_resid
              } if zeroc else {
        lf.r2: res.rsquared,
        lf.q: np.nan,
        lf.m: res.params[0],
        lf.stderr_m: res.bse[0],
        lf.pval_m: res.pvalues[0],
        lf.stderr_q: np.nan,
        lf.pval_q: np.nan,
        lf.obj: res,
        lf.df: len(X),
        lf.stderr_y: res.mse_resid
    }
    return result


# add these strings in the function scope
lf.r2 = 'R^2'
lf.m = 'm'
lf.q = 'q'
lf.stderr_m = 'stderr m'
lf.pval_m = 'pval m'
lf.stderr_q = 'stderr q'
lf.pval_q = 'pval q'
lf.obj = 'obj'
lf.df = 'df'
lf.stderr_y = 'stderr y'


def linfitplot(ax, X, Y, zeroc=True, finiteonly=False, **kwargs):
    reg = lf(X, Y, zeroc=zeroc, finiteonly=finiteonly)
    lineprops = {'color': 'black'}
    for key in kwargs:
        lineprops[key] = kwargs[key]
    sp = np.linspace(np.nanmin(X), np.nanmax(X), 3)
    if zeroc:
        ax.plot(sp, reg['m'] * sp + reg['q'], **lineprops)
    else:
        ax.plot(sp, reg['m'] * sp, **lineprops)
    return reg


class Pkl():
    def save(obj, filename):
        with open(filename, 'wb') as file:
            pickle.dump(obj, file)

    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)


def countdays(t0, t1):
    """count business days between 2 dates"""
    return np.busday_count(t0.astype('<M8[D]'), t1.astype('<M8[D]'))

QUARTER_MAP = {1: 'Q1', 4: 'Q2', 7: 'Q3', 10: 'Q4'}
def dateToQuarter(string):
    """takes a string of the type YYYY-MM and converts it to YYY-QN
    basically from month to quarter"""
    y,m = string.split('-')
    return y + '-' + QUARTER_MAP[int(m)]


def show_figure(fig, nb=False):
    # pycharm workaround:
    fig.show()
    # problems with tkinter are solved only if
    # you call this method on fig
    if not nb: fig.canvas._master.wait_window()


def draw_plot(plot_function, function_args, nb=False):
    """for drawing plots in IDEs and generally
    where you're popping out a window.
    If you're drawing a plot in a notebook,
    set nb=True"""
    fig, ax = plt.subplots()
    plot_function(ax=ax, **function_args)
    fig.tight_layout()
    show_figure(fig, nb=nb)



def save_figure(fig, name):
    """saves a figure in 3 different formats
    low-res PNG, hi-res PNG and PDF."""
    fig.savefig(Paths.FIGURES_DIR.value + name + '.png', dpi=150)
    fig.savefig(Paths.FIGURES_DIR.value + name + '_hr.png', dpi=600)
    fig.savefig(Paths.FIGURES_DIR.value + name + '.pdf', dpi=600)

def view_code(obj):
    """pretty print source code of an object, if any"""
    print("In file: " + getfile(obj) + "\n\n")
    print(''.join(getsourcelines(obj)[0]))
