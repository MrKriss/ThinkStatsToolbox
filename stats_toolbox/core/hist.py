"""Classes and function to do with the Histogramt data structure. """

import sys
import warnings

import math
import numpy as np
import matplotlib.pyplot as plt

from .shared import _DictWrapper, _underride_dict, config_current_plot

from ..config import SEABORN_CONFIG


class Hist(_DictWrapper):

    """Represents a histogram, which is a map from values to frequencies.

    Values can be any hashable type; frequencies are integer counters.
    """
    
    def freq(self, x):
        """Gets the frequency associated with the value x.

        Args:
            x: number value or sequnce of values. 

        Returns:
            int: frequency
        """
        # Check if a list of values of a single value
        if hasattr(x, '__iter__') and not isinstance(x, str):
            return [self.d.get(y, 0) for y in x]
        else:
            return self.d.get(x, 0)

    def most_freq(self, n=10):
        """Return the top n most frequently occuring items. """
        vals = sorted(self.items(), key=lambda x: x[1], reverse=True)
        return vals[:n]

    def least_freq(self, n=10):
        """Return the top n most frequently occuring items. """
        vals = sorted(self.items(), key=lambda x: x[1])
        return vals[:n]

    def mean(self):
        """Calculate the mean of the Histogram."""
        mean = 0.0
        for x, f in self.d.items():
            mean += f * x
        return mean / self.total()

    def var(self, mu=None):
        """Computes the variance of the Histogram.

        mu: the point around which the variance is computed;
                if omitted, computes the mean

        returns: float variance
        """
        if mu is None:
            mu = self.mean()

        var = 0.0
        for x, f in self.d.items():
            var += f * (x - mu) ** 2
        return var / self.total()

    def std(self, mu=None):
        """Computes the standard deviation of a PMF.

        mu: the point around which the variance is computed;
                if omitted, computes the mean

        returns: float standard deviation
        """
        var = self.var(mu)
        return math.sqrt(var)

    def is_subset(self, other):
        """Checks whether values in this histogram are a subset of values in the given histogram. """
        for val, freq in self.items():
            if freq > other.sreq(val):
                return False
        return True

    def subtract(self, other):
        """Subtracts the values in the given histogram from this histogram."""
        for val, freq in other.items():
            self.incr(val, -freq)

    def __sub__(self, other):
        """Subtracts the values in the given histogram from this histogram."""
        hist = self.copy()
        hist.subtract(other)
        return hist

    def add(self, other):
        """Add the values in the given histogram to this histogram."""
        for val, freq in other.items():
            self.incr(val, +freq)

    def __add__(self, other):
        """Add the values in the given histogram to this histogram."""
        hist = self.copy()
        hist.add(other)
        return hist

    def to_pmf(self, label=None):
        """Convert to a Probability Mass Function Object."""
        from ..core.pmf import Pmf

        label = label if label is not None else self.label
        return Pmf(self, label=label)

    def plot(self, mpl_hist=False, **options):
        """Plots the Hist with a bar plot.

        The default width of the bars is based on the minimum difference
        between values in the Hist.  If that's too small, you can override
        it by providing a width keyword argument, in the same units
        as the values.

        Args:
          mpl_hist: Whether to use the matplotlib hist function to plot with instead of a bar plot. 
          options: keyword args passed to pyplot.bar / pyplot.hist

        Returns:
          ax : The matplotlib axies object
        """        
        # Initialise seaborn with config file if it is yet to be imported
        if 'seaborn' not in sys.modules:
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

        # Set axes instance if given
        if 'axes' in options:  
            axes = options.pop('axes')
            plt.sca(axes)

        # find the minimum distance between adjacent values
        xs, ys = self.render()

        if not mpl_hist:

            if 'width' not in options:
                try:
                    options['width'] = 0.9 * np.diff(xs).min()
                except TypeError:
                    warnings.warn("Hist: Can't compute bar width automatically."
                                  "Check for non-numeric types in Hist."
                                  "Or try providing width option.")

            options = _underride_dict(options, label=self.label)
            options = _underride_dict(options, align='center')
            options = _underride_dict(options, linewidth=0, alpha=0.6)

            if options['align'] == 'left':
                options['align'] = 'edge'
            elif options['align'] == 'right':
                options['align'] = 'edge'
                options['width'] *= -1

            plt.bar(xs, ys, **options)
            # Perform any additional plot configurations 
            config_current_plot(**plot_configs)

        else:
            # Must generate the full data before passing to pyplot.hist
            data = []
            for k, v in self.items():
                data.extend([k] * v) 

            options = _underride_dict(options, label=self.label, linewidth=1.2, alpha=0.7, 
                                      histtype='stepfilled')

            plt.hist(data, **options)
            # Perform any additional plot configurations 
            config_current_plot(**plot_configs)

        return plt.gca()
