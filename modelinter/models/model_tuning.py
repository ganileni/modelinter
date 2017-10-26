import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # this is actually needed for the 3d plot

from modelinter.models.utils import dateToQuarter


def plot_days_3d(t, days_change, allresults):
    fig = plt.figure()
    # the following line will never be executed
    # but it is needed to make IDEs recognize that
    # Axes3D is required as a dependency for this plot
    if False: Axes3D
    ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(elev = 45, azim=30) # syntax for rotating plot
    for j, dc in enumerate(days_change):  # remove [:n]
        ax.plot(xs=np.array(t, dtype=float),
                ys=[dc] * len(allresults[j].both_meanA),
                zs=allresults[j].both_meanA,
                color='royalblue', alpha=.5)
    return fig


def plot_days(ax, t, allresults, days_change):
    dashes = [[1, 1, 1, 1],
              [5, 5, 5, 5],
              [10, 1, 1, 1],
              []]
    for k, d in zip(list(sorted([1, 11, 4, 13, 7])), dashes):
        lyne = ax.plot(np.array(days_change) + 365,
                       [_.both_meanA[k] for _ in allresults],
                       label=dateToQuarter(str(t[k])[:7]), linewidth=1, color='k')
        lyne[0].set_dashes(d)
    # , s=3, alpha = .8)
    ax.legend(loc='center right')
    ax.set_xlabel('business days used to train the model')
    ax.set_ylabel('PGM A $E[L_t]$')
