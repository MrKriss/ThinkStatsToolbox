"""Thei module holds utility functions to generate a variety of statistical plots."""

import itertools

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb

from .shared import _underride_dict
from .hist import Hist
from .tools import fit_line
from ..utils.data_generators import render_normal_probability


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
            plt_kwds = _underride_dict(plt_kwds, alpha=0.65)

        obj.plot(color=next(palette), axes=ax, **plt_kwds)

    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)

    return fig


def normal_probability_plot(sample, fit_color='0.8', **options):
    """Makes a normal probability plot with a fitted line.

    sample: sequence of numbers
    fit_color: color string for the fitted line
    options: passed along to Plot
    """
    xs, ys = render_normal_probability(sample)

    arr = np.asarray(sample)
    mean = arr.mean()
    std = arr.std()

    fit = fit_line(xs, mean, std)

    fig, ax = plt.subplots()

    options = _underride_dict(options, linewidth=3, alpha=0.8)
    ax.plot(*fit, color=fit_color, label='model', **options)

    # Draw a second sample and plot alongside fitted model
    xs, ys = render_normal_probability(sample)
    ax.plot(xs, ys, **options)

    return ax 


