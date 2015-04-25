"""Test script for plotting functions """

import itertools

import matplotlib.pyplot as plt
import seaborn as sb

from ..core import Cdf, Hist, Pdf, Pmf
from ..core.shared import _underride_dict


def mulitplot(objects, xlab=None, ylab=None, plt_kwds=None, fig_kwds=None):
    """Plot multiple stats objects on a single plot. 

    Plots all objects on a single axes. Legend will reflect the 
    label of the individual objects. The color sequence is taken from current 
    seaborn pallet. 

    Args:
      objects (list) : Either Hist, Pmf or Pdf objects.
      plt_kwds (dict) : Dictionary for keyword arguments passed to object.plot()
      fig_kwds (dict) : Dictionary for keyword arguments passed to matplotlib.pyplot.figure()

    Returns:
      fig : the matplotlib figure container for the plot. 
    """
    # Initalise dictionaries for extra param handelig 
    if plt_kwds is None:
        plt_kwds = dict()
    if fig_kwds is None:
        fig_kwds = dict()

    fig = plt.figure(**fig_kwds)
    ax = fig.add_subplot(111)
    palette = itertools.cycle(sb.color_palette())

    
    for obj in objects:
        # Set some useful defaults if Hist is detected
        if isinstance(obj, Hist):
            _underride_dict(plt_kwds, alpha=0.65)

        obj.plot(color=next(palette), axes=ax, **plt_kwds)

    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)

    return fig



cols = sb.color_palette()

h1 = Hist([1, 3, 4, 3, 2, 1, 2, 3, 4, 5, 3, 4, 2, 2, 4, 5, 5, 6, 6, 4, 3, 2], label='Hist 1')
h2 = Hist([5, 3, 4, 2, 2, 4, 5, 5, 6, 6, 4, 3, 2, 5, 3, 4, 2, 2, 4, 5, 5, 6, 6, 4, 3, 2], label='Hist 2')


mulitplot([h1,h2], plt_kwds={'alpha': 0.6})

# p = Pmf(h1)
# c = Cdf(h1)

# plt.figure()
# plt.hold
# ax = h1.plot(color=cols[0], alpha=0.6)
# ax = h2.plot(color=cols[1], alpha=0.6)

# plt.figure()
# p.plot()

# plt.figure()
# c.plot()

# plt.show(block=False)



