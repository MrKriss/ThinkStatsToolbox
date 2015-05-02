"""Thei module holds utility functions to generate a variety of statistical plots."""

import itertools
from collections import Iterable

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb

from .shared import _underride_dict, config_current_plot
from .hist import Hist
from .cdf import Cdf
from .tools import fit_line
from ..utils.data_generators import render_normal_probability
from ..config import SEABORN_CONFIG


def mulitplot(objects, plt_kwds=None, fig_kwds=None, palette=None, **options):
    """Plot multiple stats objects on a single plot.

    Plots all objects on a single axes. Legend will reflect the 
    label of the individual objects. The color sequence is taken from current 
    seaborn pallet. 

    Args:
      objects (list) : Either Hist, Pmf or Pdf objects.
      plt_kwds (dict) : Dictionary for keyword arguments passed to object.plot()
      fig_kwds (dict) : Dictionary for keyword arguments passed to matplotlib.pyplot.figure()
      palette (list) : A list of colors to iterate over in all the line plots, if given.
      **options : Keyword arguments used for configuring axis, labels, titles.  

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

    if palette is None:
        palette = itertools.cycle(sb.color_palette())
    else:
        palette = itertools.cycle(palette)

    for obj in objects:
        # Set some useful defaults if Hist is detected
        if isinstance(obj, Hist):
            plt_kwds = _underride_dict(plt_kwds, alpha=0.65)

        obj.plot(color=next(palette), axes=ax, **plt_kwds)

    options = _underride_dict(options, legend=True)
    config_current_plot(**options)

    return fig


def normal_probability_plot(samples, labels=None, fit_color='0.7', palette=None, **options):
    """Makes a normal probability plot with a fitted line.

    Args:
        samples: sequence of numbers or a cdf object, or a list of sequences or cdf objects. 
            If a list, all sequences / objects are plotted, but the fit line is only plotted 
            for the first one in the list.
        labels: sequence of labels for the samples parsed in.  
        fit_color: color string for the fitted line
        pallet: sequence of colurs to iterate over for the plots
        options: passed along to Plot

    Returns:
      fig : the matplotlib figure container for the plot. 
    """
    import seaborn as sb
    sb.set_context(**SEABORN_CONFIG['context'])
    sb.set_palette(SEABORN_CONFIG['pallet'])
    sb.set_style(SEABORN_CONFIG['style'])

    # Handle extra parameters parsed to config plot
    plot_configs = dict()
    for n in ['title', 'xlabel', 'ylabel', 'xscale', 'yscale',
              'xticks', 'yticks', 'axis', 'xlim', 'ylim', 'legend']:
        if n in options:
            plot_configs[n] = options.pop(n)

    # Determine whether samples is a sequence or list of sequences.
    if isinstance(samples, str):
        raise TypeError(
            'samples argument must be a sequence of numbers or a Cdf object, or a list thereof. Not a string.')
    if not isinstance(samples, Iterable) and not isinstance(samples, Cdf):
        raise TypeError(
            'samples argument must be a sequence of numbers or a Cdf object, or a list thereof')

    iters_not_str = [(isinstance(s, Iterable) or isinstance(s, Cdf)) and not isinstance(s, str) for s in samples]

    if not all(iters_not_str):
        # samples is a single sequence
        samples = [samples]

    # Set colour palette
    if palette is None:
        palette = itertools.cycle(sb.color_palette())
    else: 
        palette = itertools.cycle(palette) 

    # initialise plot
    fig, ax = plt.subplots()

    for idx, obj in enumerate(samples):

        if isinstance(obj, Cdf):
            values = obj.xs
        else:
            values = obj

        if idx == 0:
            xs, ys = render_normal_probability(values)

            arr = np.asarray(values)
            mean = arr.mean()
            std = arr.std()

            fit = fit_line(xs, mean, std)
            options = _underride_dict(options, linewidth=1.5)
            ax.plot(*fit, color=fit_color, label='model', **options)

        # Draw a sample and plot alongside fitted model
        if labels is None:
            if hasattr(obj, 'label'):
                label = obj.label
            else:
                label = 'Sample %s' % idx

        options = _underride_dict(options, linewidth=1.5, alpha=0.8)
        xs, ys = render_normal_probability(values)
        ax.plot(xs, ys, color=next(palette), label=label, **options)

    # Add labels and title
    plot_configs = _underride_dict(plot_configs, xlabel='Sample values',
                                   ylabel='Standard Normal Sample', 
                                   title='Normal Probability Plot',
                                   legend=True)

    config_current_plot(**plot_configs)

    return fig
