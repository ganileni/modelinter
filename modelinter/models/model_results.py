'''
Created on 8 Sep 2017

@author: orazio.angelini
'''

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, skew

from modelinter.models.utils import inverse_returns, dateToQuarter


def plot_stocks_portfolio_distribution(ax, results, quarter, bins=500):
    sec_name = 'stocks '
    B = results.portfolio_stocks_pgm_b[quarter]
    C = results.portfolio_stocks_pgm_c[quarter]
    ax.hist(C, label=sec_name + 'model C distribution', alpha=.5, bins=bins)
    ax.hist(B, label=sec_name + 'model B distribution', alpha=.5, bins=bins)
    ax.axvline(results.portfolio_stocks_pgm_a[quarter], label=sec_name + 'model A expected', color='red')
    ax.axvline(np.mean(B), label=sec_name + 'model B expected', color='blue')
    ax.axvline(np.mean(C), label=sec_name + 'model C expected', color='green')
    ax.legend(loc='best')


def plot_correlation(ax, results, quarter=1):
    ax.scatter(results.portfolio_stocks_pgm_c[quarter], results.portfolio_options_pgm_c[quarter],
               s=3, color='blue', alpha=.2, marker='v',
               label='C models, 100k samples')
    ax.scatter(results.portfolio_stocks_pgm_b[quarter], results.portfolio_options_pgm_b[quarter],
               s=3, color='red', alpha=.2, marker='^',
               label='B models, 100k samples')
    ax.legend(loc='best')
    ax.set_xlabel('sampled stock portfolio value $L_t^s$')
    ax.set_ylabel('sampled options portfolio value $L_t^o$')


def check_discontinuity(ax, results, stock_number=1):
    ax.scatter([_[stock_number] for _ in results.one_s],
               [_[stock_number] for _ in results.one_o],
               s=4, color='blue', alpha=.5, marker='v', label='C model')
    ax.legend(loc='best')
    ax.set_xlabel('sampled individual stock value')
    ax.set_ylabel('sampled individual option value')


def plot_options_portfolio_distribution(ax, results, quarter, bins=500):
    sec_name = 'options '
    B = results.portfolio_options_pgm_b[quarter]
    C = results.portfolio_options_pgm_c[quarter]
    n, bins, patches = ax.hist(C, label=sec_name + 'model C distribution', color='blue', alpha=.5, bins=bins)
    ax.hist(B, label=sec_name + 'model B distribution', color='orange', alpha=.5, bins=bins)
    xcoord = np.linspace(np.min(C), np.max(C), num=500)
    ycoord = norm.pdf(xcoord, loc=np.mean(C), scale=np.std(C))
    ycoord = np.max(n) * ycoord / np.max(ycoord)
    ax.plot(xcoord, ycoord, color='blue', alpha=.5, label=' comparison normal distribution')
    ax.axvline(results.portfolio_options_pgm_a[quarter], label=sec_name + 'model A expected', color='red')
    ax.axvline(np.mean(B), label=sec_name + 'model B expected', color='blue')
    ax.axvline(np.mean(C), label=sec_name + 'model C expected', color='green')
    ax.legend(loc='best')
    ax.set_title('options portfolio samples')


def print_skew(results, quarter=1):
    B = results.portfolio_options_pgm_b[quarter]
    C = results.portfolio_options_pgm_c[quarter]
    normal_skews = [skew(np.random.normal(size=int(len(B)))) for _ in range(20)]
    skew(np.random.normal(size=int(1e5))), skew(B), skew(C)
    print('skewness and 2*std of a similarly-sized normal distribution', '\n',
          np.mean(normal_skews), np.std(normal_skews),
          '\n', 'skewness of PGM B options portfolio distribution', '\n',
          skew(B),
          '\n', 'skewness of PGM C options portfolio distribution', '\n',
          skew(C)
          )


def print_skew_stocks(results, quarter=1):
    B = results.portfolio_stocks_pgm_b[quarter]
    C = results.portfolio_stocks_pgm_c[quarter]
    normal_skews = [skew(np.random.normal(size=int(len(B)))) for _ in range(20)]
    skew(np.random.normal(size=int(1e5))), skew(B), skew(C)
    print('skewness and 2*std of a similarly-sized normal distribution', '\n',
          np.mean(normal_skews), np.std(normal_skews),
          '\n', 'skewness of PGM B stock portfolio distribution', '\n',
          skew(B),
          '\n', 'skewness of PGM C stock portfolio distribution', '\n',
          skew(C)
          )


def plot_scenario(ax, scenario, settings):
    ax.plot(scenario.t,
            (scenario.vix_t + settings.vix_t0),
            color='red', linestyle='solid', label='VIX')
    ax2 = ax.twinx()
    ax2.plot(scenario.t,
             inverse_returns(settings.dow_t0, scenario.dow_t),
             color='blue', linestyle='dashed', label='Dow Jones')
    ax.plot(scenario.t[0],
            (scenario.vix_t + settings.vix_t0)[0],
            color='blue', linestyle='dashed', label='Dow Jones')
    ax.legend(loc='upper center', fontsize=8)
    ax.set_title('CCAR scenario')
    ax.set_xticks(scenario.t)
    ax.set_xticklabels(labels=[dateToQuarter(repr(_)[18:25]) for _ in scenario.t],
                       rotation='45')


def plot_stocks_portfolio(ax, results, scenario):
    meanA = np.array([np.mean(_) for _ in results.portfolio_stocks_pgm_a])  # * rescale_s
    meanB = np.array([np.mean(_) for _ in results.portfolio_stocks_pgm_b])  # * rescale_s
    stdB = 3 * np.array([np.std(_) for _ in results.portfolio_stocks_pgm_b])  # * rescale_s
    meanC = np.array([np.mean(_) for _ in results.portfolio_stocks_pgm_c])  # * rescale_s
    stdC = 3 * np.array([np.std(_) for _ in results.portfolio_stocks_pgm_c])  # * rescale_s
    sec_name = 'stocks '
    ax.plot(scenario.t, meanA, color='red', linestyle='solid',
            label='PGM A $E(L^s_t)$')
    ax.plot(scenario.t, meanB, color='green', linestyle='dashed',
            label='PGM B $E(L^s_t)$')
    ax.plot(scenario.t, meanC, color='blue', linestyle='dotted',
            label='PGM C $E(L^s_t)$')
    ax.axhline(meanA[0], linewidth=.5, linestyle='dashed',
               color='black', label='baseline', alpha=.5)
    ax.fill_between(scenario.t,
                    meanB - stdB,
                    meanB + stdB,
                    alpha=.2, color='green',
                    label='PGM B $3 \sigma(L^s_t)$')
    ax.fill_between(scenario.t,
                    meanC - stdC,
                    meanC + stdC,
                    alpha=.2, color='blue',
                    label='PGM C $3 \sigma(L^s_t)$')
    ax.legend(loc='best', fontsize=8)
    ax.set_title('stock portfolio')
    ax.set_xticks(scenario.t)
    ax.set_xticklabels(labels=[dateToQuarter(repr(_)[18:25]) for _ in scenario.t],
                       rotation='45')


def plot_options_portfolio(ax, results, scenario):
    dashes = [10, 2, 1, 2]
    meanA = np.array([np.mean(_) for _ in results.portfolio_options_pgm_a])  # * rescale_o
    meanB = np.array([np.mean(_) for _ in results.portfolio_options_pgm_b])  # * rescale_o
    stdB = 3 * np.array([np.std(_) for _ in results.portfolio_options_pgm_b])  # * rescale_o
    meanC = np.array([np.mean(_) for _ in results.portfolio_options_pgm_c])  # * rescale_o
    stdC = 3 * np.array([np.std(_) for _ in results.portfolio_options_pgm_c])  # * rescale_o
    sec_name = 'options '
    ax.plot(scenario.t, meanA, color='red', linestyle='solid', label='PGM A $E(L^s_t)$')
    ax.plot(scenario.t, meanB, color='green', linestyle='dashed', label='PGM B $E(L^s_t)$')
    ax.plot(scenario.t, meanC, color='blue', linestyle='dotted', label='PGM C $E(L^s_t)$')
    ax.axhline(meanA[0], linewidth=.5, linestyle='dashed', color='black', label='baseline', alpha=.5)
    # line1 = ax.axhline(0, linewidth = .5, linestyle = 'dashed', color = 'black', alpha = .5)
    # line1.set_dashes(dashes)
    ax.fill_between(scenario.t, meanB - stdB, meanB + stdB, alpha=.2, color='green', label='PGM B $3 \sigma (L^s_t)$')
    ax.fill_between(scenario.t, meanC - stdC, meanC + stdC, alpha=.2, color='blue', label='PGM C $3 \sigma (L^s_t)$')
    ax.legend(loc='best', fontsize=8)
    ax.set_title('options portfolio')
    ax.set_xticks(scenario.t)
    ax.set_xticklabels(labels=[dateToQuarter(repr(_)[18:25]) for _ in scenario.t], rotation='45')


def plot_combined_portfolios(ax, results, scenario):
    dashes = [10, 2, 1, 2]
    meanA = np.array([np.mean(_ + __) for _, __ in zip(results.portfolio_options_pgm_a, results.portfolio_stocks_pgm_a)])
    meanB = np.array([np.mean(_ + __) for _, __ in zip(results.portfolio_options_pgm_b, results.portfolio_stocks_pgm_b)])
    stdB = 3 * np.array([np.std(_ + __) for _, __ in zip(results.portfolio_options_pgm_b, results.portfolio_stocks_pgm_b)])
    meanC = np.array([np.mean(_ + __) for _, __ in zip(results.portfolio_options_pgm_c, results.portfolio_stocks_pgm_c)])
    stdC = 3 * np.array([np.std(_ + __) for _, __ in zip(results.portfolio_options_pgm_c, results.portfolio_stocks_pgm_c)])
    sec_name = 'combined portfolios '
    ax.plot(scenario.t, meanA, color='red', linestyle='solid', label='PGM A $E(L_t)$')
    ax.plot(scenario.t, meanB, color='green', linestyle='dashed', label='PGM B $E(L_t)$')
    ax.plot(scenario.t, meanC, color='blue', linestyle='dotted', label='PGM C $E(L_t)$')
    ax.axhline(meanA[0], linewidth=.5, linestyle='dashed', color='black', label='baseline', alpha=.5)
    # line1 = ax.axhline(0, linewidth = .5, linestyle = 'dashed', color = 'black', alpha = .5)
    # line1.set_dashes(dashes)
    ax.fill_between(scenario.t, meanB - stdB, meanB + stdB, alpha=.2, color='green', label='PGM B $3 \sigma (L_t)$')
    ax.fill_between(scenario.t, meanC - stdC, meanC + stdC, alpha=.2, color='blue', label='PGM C $3 \sigma (L_t)$')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title('combined portfolios')
    ax.set_xticks(scenario.t)
    ax.set_xticklabels(labels=[dateToQuarter(repr(_)[18:25]) for _ in scenario.t], rotation='45')


def plot_models_interaction(ax, results, scenario):
    # mean and std for compined portfolios
    both_meanC = np.array([np.mean(_) + np.mean(__) for _, __ in zip(results.portfolio_options_pgm_c, results.portfolio_stocks_pgm_c)])
    both_stdC = 3 * np.array([np.std(_ + __) for _, __ in zip(results.portfolio_options_pgm_c, results.portfolio_stocks_pgm_c)])
    # shuffle the portfolio samples to destroy correlations
    portfolio_stocks_pgm_c_scramble = deepcopy(results.portfolio_stocks_pgm_c)
    portfolio_options_pgm_c_scramble = deepcopy(results.portfolio_options_pgm_c)
    [np.random.shuffle(_) for _ in portfolio_stocks_pgm_c_scramble]
    [np.random.shuffle(_) for _ in portfolio_options_pgm_c_scramble]
    uncorr_stdC = 3 * np.array([np.std(_ + __) for _, __ in zip(portfolio_options_pgm_c_scramble, portfolio_stocks_pgm_c_scramble)])

    ax.plot(scenario.t, both_stdC / (3 * both_meanC),
            color='black', linestyle='solid',
            label='PGM C, combined portfolios')
    ax.plot(scenario.t, uncorr_stdC / (3 * both_meanC),
            color='black', linestyle='dashed',
            label='PGM C, combined portfolios - no correlation')

    ax.set_ylabel('$\sigma [L_t] \quad / \quad E[L_t]$')

    ax.legend(loc='best', fontsize=8)
    ax.set_xticks(scenario.t)
    ax.set_xticklabels(labels=[dateToQuarter(repr(_)[18:25]) for _ in scenario.t], rotation='45')


def plot_paper_correlations(results, scenario, quarter=7):
    fig, axes = plt.subplots(1, 2, **{'figsize': (10, 5)})
    plot_correlation(axes[0], results=results, quarter=quarter)
    plot_models_interaction(axes[1], results=results, scenario=scenario)
    fig.tight_layout()
    return fig


def plot_paper_scenarios(results, scenario, DefaultSettings):
    # sharex gives weird x ticklabels
    fig, axes = plt.subplots(2, 2, **{'figsize': (10, 7)})  # sharex = True
    plot_stocks_portfolio(ax=axes[0, 0], results=results, scenario=scenario)
    # turn off xticklabels
    labels = [item.get_text() for item in axes[0, 0].get_xticklabels()]
    empty_string_labels = [''] * len(labels)
    axes[0, 0].set_xticklabels(empty_string_labels)

    plot_options_portfolio(ax=axes[0, 1], results=results, scenario=scenario)
    # turn off xticklabels
    labels = [item.get_text() for item in axes[0, 1].get_xticklabels()]
    empty_string_labels = [''] * len(labels)
    axes[0, 1].set_xticklabels(empty_string_labels)

    plot_scenario(ax=axes[1, 0], scenario=scenario, settings=DefaultSettings)
    plot_combined_portfolios(ax=axes[1, 1], results=results, scenario=scenario)
    fig.tight_layout()
    return (fig)
