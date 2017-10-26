# This script will make all the calculations that we executed in the paper.
# It will save the results in .pkl files, so that you can explore them
# in the notebook without having to calculate everything on the spot.
# mind that the calculations could take several hours on a typical laptop.
from os.path import isfile

import numpy as np

from modelinter.models.calculations import eval_PGM, DefaultSettings, extract_scenario, calc_weights, sample_results, \
    DefaultSettings_testdays, sample_day_loop
from modelinter.models.constants import Paths, Const
from modelinter.models.model_results import plot_paper_scenarios, plot_paper_correlations
from modelinter.models.model_tuning import plot_days_3d
from modelinter.models.utils import Pkl, save_figure
from modelinter.preprocessing.imports_load_data import read_csvs, extract_arrays

if __name__ == '__main__':

    #############################
    # calculations for notebook 2
    #############################

    np.random.seed(1)
    raw_data = read_csvs()
    arrays = extract_arrays(raw_data)
    PGM = eval_PGM(arrays, DefaultSettings)
    scenario = extract_scenario(arrays=arrays,
                                raw_data=raw_data,
                                options_model=PGM.PGM_options_A,
                                settings=DefaultSettings)
    # weights for stocks and options respectively
    weights = calc_weights(scenario.stock_prices_t0, scenario.options_prices_t0)

    # if results have already been calculated and saved:
    if isfile(Paths.SAVE_DIR.value + 'results' + Paths.PKL_EXT.value) and not DefaultSettings.recalculate_results:
        # load pickled results
        results = Pkl.load(Paths.SAVE_DIR.value + 'results' + Paths.PKL_EXT.value)
    else:
        # sample the results
        # code: L_[Stock or Option][PGM A, B or C]
        # one_s, one_o = 1000 samples each for first 50 option-stock pair
        results = sample_results(scenario, weights, PGM,
                                 DefaultSettings.samples)
        if DefaultSettings.save_results:
            # pickle results
            Pkl.save(results,
                     Paths.SAVE_DIR.value + 'results' + Paths.PKL_EXT.value)
    # plot the paper's figures and save them
    fig = plot_paper_scenarios(results, scenario, DefaultSettings)
    save_figure(fig, name='results')
    fig = plot_paper_correlations(results, scenario, quarter=7)
    save_figure(fig, name='interaction')

    #############################
    # calculations for notebook 3
    #############################

    # let's take a sample every 30 days for 4 years
    days_change = list(range(-int(3 * Const.WHOLE_YEAR.value / 4),
                             5 * Const.WHOLE_YEAR.value,
                             30))
    savefile = 'days_change' + Paths.PKL_EXT.value

    # if results have already been calculated and saved:
    if isfile(Paths.SAVE_DIR.value + savefile) and not DefaultSettings_testdays.recalculate_results:
        # load pickled results
        results = Pkl.load(Paths.SAVE_DIR.value + savefile)
    else:
        # sample the results



        results = sample_day_loop(raw_data, DefaultSettings_testdays, days_change)
        if DefaultSettings_testdays.save_results:
            # pickle results
            Pkl.save(results,
                     Paths.SAVE_DIR.value + savefile)

    # we need the timestamps for the CCAR scenario, let's extract them
    arrays = extract_arrays(raw_data)
    PGM = eval_PGM(arrays, DefaultSettings_testdays)
    scenario = extract_scenario(arrays=arrays,
                                raw_data=raw_data,
                                options_model=PGM.PGM_options_A,
                                settings=DefaultSettings_testdays)
    t = scenario.t
    fig = plot_days_3d(t, days_change, results)
    save_figure(fig, name='days_change')
