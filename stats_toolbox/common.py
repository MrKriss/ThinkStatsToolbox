"""Base dictionary class inherited by the Hist and Pmf classes. 

Also holds common utility functions to handle plotting parameters """

import copy
import math

import logging
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class _DictWrapper(object):

    """An object that contains a dictionary."""

    def __init__(self, obj=None, label=None):
        """Initialize the distribution.

        Args:
            obj: Hist, Pmf, Cdf, Pdf, dict, pandas Series, list of pairs
            label: string label
        """
        # Import other object to refernce here to avoid circular imports 
        from .pdf import Pdf
        from .cdf import Cdf

        # Common state variables 
        self.label = label if label is not None else '_nolegend_'
        self.d = {}
        # flag whether the distribution is under a log transform
        self.log = False

        if obj is None:
            return

        if isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            self.label = label if label is not None else obj.label

        if isinstance(obj, (dict, _DictWrapper, Cdf, Pdf)):
            self.d.update(obj.items())
        elif isinstance(obj, pd.Series):
            self.d.update(obj.value_counts().iteritems())
        else:
            # finally, treat it like a list
            self.d.update(Counter(obj))

    def __hash__(self):
        return id(self)

    def __str__(self):
        cls = self.__class__.__name__
        return '%s(%s)' % (cls, str(self.d))

    __repr__ = __str__

    def __eq__(self, other):
        return self.d == other.d

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def iterkeys(self):
        """Return an iterator over keys."""
        return iter(self.d)

    def __contains__(self, value):
        return value in self.d

    def __getitem__(self, value):
        return self.d.get(value, 0)

    def __setitem__(self, value, prob):
        self.d[value] = prob

    def __delitem__(self, value):
        del self.d[value]

    def copy(self, label=None):
        """Return a copy.

        Make a shallow copy of d.  If you want a deep copy of d,
        use copy.deepcopy on the whole object.

        Args:
            label: string label for the new Hist

        Returns: 
            new _DictWrapper with the same type
        """
        new = copy.copy(self)
        new.d = copy.copy(self.d)
        new.label = label if label is not None else self.label
        return new

    def scale(self, factor):
        """Multiply the values by a factor.

        Args:
            factor: what to multiply by

        Returns: 
            new object
        """
        new = self.copy()
        new.d.clear()

        for val, prob in self.items():
            new.set(val * factor, prob)
        return new

    def log(self, m=None):
        """Log transforms the probabilities.
        
        Removes values with probability 0.

        Normalizes so that the largest logprob is 0.
        """
        if self.log:
            raise ValueError("Pmf/Hist already under a log transform")
        self.log = True

        if m is None:
            m = self.max_like()

        for x, p in self.d.items():
            if p:
                self.set(x, math.log(p / m))
            else:
                self.remove(x)

    def exp(self, m=None):
        """Exponentiate the probabilities.

        m: how much to shift the ps before exponentiating

        If `m` is None, normalizes so that the largest prob is 1.
        """
        if not self.log:
            raise ValueError("Pmf/Hist not under a log transform")
        self.log = False

        if m is None:
            m = self.max_like()

        for x, p in self.d.items():
            self.set(x, math.exp(p - m))

    def get_dict(self):
        """Get the dictionary."""
        return self.d

    def set_dict(self, d):
        """Set the dictionary."""
        self.d = d

    def values(self):
        """Get an unsorted sequence of values.

        Note: one source of confusion is that the keys of this
        dictionary are the values of the Hist/Pmf, and the
        values of the dictionary are frequencies/probabilities.
        """
        return self.d.keys()

    def items(self):
        """Get an unsorted sequence of (value, freq/prob) pairs."""
        return self.d.items()

    def render(self, **options):
        """Generate a sequence of points suitable for plotting.

        Note: options are ignored

        Returns:
            tuple of (sorted value sequence, freq/prob sequence)
        """
        if min(self.d.keys()) is np.nan:
            logging.warning('Hist: contains NaN, may not render correctly.')

        return zip(*sorted(self.items()))

    def to_cdf(self, label=None):
        """Make a Cdf from this object."""
        from .cdf import Cdf

        label = label if label is not None else self.label
        return Cdf(self, label=label)

    def pprint(self):
        """Print the values and freqs/probs in ascending order."""
        for val, prob in sorted(self.d.items()):
            print(val, prob)

    def set(self, x, y=0):
        """Sets the freq/prob associated with the value x.

        Args:
            x: number value
            y: number freq or prob
        """
        self.d[x] = y

    def incr(self, x, term=1):
        """Increment the freq/prob associated with the value x.

        Args:
            x: number value
            term: how much to increment by
        """
        self.d[x] = self.d.get(x, 0) + term

    def mult(self, x, factor):
        """Scale the freq/prob associated with the value x.

        Args:
            x: number value
            factor: how much to multiply by
        """
        self.d[x] = self.d.get(x, 0) * factor

    def remove(self, x):
        """Remove a value.

        Throws an exception if the value is not there.

        Args:
            x: value to remove
        """
        del self.d[x]

    def total(self):
        """Return the total of the frequencies/probabilities in the map."""
        total = sum(self.d.values())
        return total

    def max_like(self):
        """Return the largest frequency/probability in the map."""
        return max(self.d.values())

    def largest(self, n=10):
        """Return the largest n values, with frequency/probability.

        n: number of items to return
        """
        return sorted(self.d.items(), reverse=True)[:n]

    def smallest(self, n=10):
        """Return the smallest n values, with frequency/probability.

        n: number of items to return
        """
        return sorted(self.d.items(), reverse=False)[:n]


class UnimplementedMethodException(Exception):

    """Exception if someone calls a method that should be overridden."""


# Functions

def _underride_dict(d, **options):
    """Add key-value pairs to d only if key is not in d.

    If d is None, create a new dictionary.

    d: dictionary
    options: keyword args to add to d
    """
    if d is None:
        d = {}

    for key, val in options.items():
        d.setdefault(key, val)

    return d


# Plotting functions 
LEGEND = True
LOC = 1


def config_current_plot(**options):
    """Configures the current plot with the options dictionay.

    Pulls options out of the option dictionary and passes them to
    the corresponding pyplot functions.
    """
    names = ['title', 'xlabel', 'ylabel', 'xscale', 'yscale',
             'xticks', 'yticks', 'axis', 'xlim', 'ylim']

    for name in names:
        if name in options:
            # Call appropriate function to set property
            getattr(plt, name)(options[name])

    global LEGEND
    LEGEND = options.get('legend', LEGEND)

    if LEGEND:
        global LOC
        LOC = options.get('loc', LOC)
        plt.legend(loc=LOC)
