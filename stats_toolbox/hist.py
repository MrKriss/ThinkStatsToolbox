"""Classes and function to do with the Histogramt data structure. """

import warnings

import numpy as np
import matplotlib.pyplot as plt

from .common import _DictWrapper, _underride_dict

from .config import SEABORN_CONFIG


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

    def plot(self, **options):
        """Plots a Pmf or Hist with a bar plot.

        The default width of the bars is based on the minimum difference
        between values in the Hist.  If that's too small, you can override
        it by providing a width keyword argument, in the same units
        as the values.

        Args:
          hist: Hist or Pmf object
          options: keyword args passed to pyplot.bar

        Returns:
          ax : The matplotlib axies object
        """
        import seaborn as sb
        sb.set_context(SEABORN_CONFIG['context'])
        sb.set_palette(SEABORN_CONFIG['pallet'])
        sb.set_style(SEABORN_CONFIG['style'])

        # find the minimum distance between adjacent values
        xs, ys = self.render()

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
        plt.legend()

        return plt.gca()
