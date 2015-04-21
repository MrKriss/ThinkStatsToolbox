"""This file contains classes and utilites for empirical PMFs/PDFs and CDFs.

The code contained is a slightly modified version of the "thinkstats2.py" 
module written as part of the "Think Stats" book by Allen B. Downey, 
available from greenteapress.com

The modifications I intend to make are:
1. Incorperate plotting methods which use seaborn. 
2. Refactoring to snake case for function names.
3. Make Python 3 compatible.
4. Incorperate to / from conversion functins into the classes.

Classes
-------
Cdf: represents a discrete cumulative distribution function

Functions
---------


License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import copy
import bisect
import random
import logging
import math

import numpy as np
import matplotlib.pyplot as plt

from .common import UnimplementedMethodException, _underride_dict, config_current_plot
from .config import SEABORN_CONFIG


class Cdf(object):

    """Represents a cumulative distribution function.

    Attributes:
        xs: sequence of values
        ps: sequence of probabilities
        label: string used as a graph label.
    """
    
    def __init__(self, obj=None, ps=None, label=None):
        """Initializes.
        
        If ps is provided, obj must be the corresponding list of values.

        obj: Hist, Pmf, Cdf, Pdf, dict, pandas Series, list of pairs
        ps: list of cumulative probabilities
        label: string label
        """
        # Dependent imports done here to handle cyclic import 
        from .pdf import Pdf
        from .hist import Hist
        from .common import _DictWrapper

        self.label = label if label is not None else '_nolegend_'

        if isinstance(obj, (_DictWrapper, Cdf, Pdf)):
            if not label:
                self.label = label if label is not None else obj.label

        if obj is None:
            # caller does not provide obj, make an empty Cdf
            self.xs = np.asarray([])
            self.ps = np.asarray([])
            if ps is not None:
                logging.warning("Cdf: can't pass ps without also passing xs.")
            return
        else:
            # if the caller provides xs and ps, just store them          
            if ps is not None:
                if isinstance(ps, str):
                    logging.warning("Cdf: ps can't be a string")

                self.xs = np.asarray(obj)
                self.ps = np.asarray(ps)
                return

        # caller has provided just obj, not ps
        if isinstance(obj, Cdf):
            self.xs = copy.copy(obj.xs)
            self.ps = copy.copy(obj.ps)
            return

        if isinstance(obj, _DictWrapper):
            dw = obj
        else:
            dw = Hist(obj)

        if len(dw) == 0:
            self.xs = np.asarray([])
            self.ps = np.asarray([])
            return

        xs, freqs = zip(*sorted(dw.items()))
        self.xs = np.asarray(xs)
        self.ps = np.cumsum(freqs, dtype=np.float)
        self.ps /= self.ps[-1]

    def __str__(self):
        return 'Cdf(%s, %s)' % (str(self.xs), str(self.ps))

    __repr__ = __str__

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, x):
        return self.prob(x)

    def __setitem__(self):
        raise UnimplementedMethodException()

    def __delitem__(self):
        raise UnimplementedMethodException()

    def __eq__(self, other):
        return np.all(self.xs == other.xs) and np.all(self.ps == other.ps)

    def copy(self, label=None):
        """Returns a copy of this Cdf.

        label: string label for the new Cdf
        """
        if label is None:
            label = self.label
        return Cdf(list(self.xs), list(self.ps), label=label)

    def to_pmf(self, label=None):
        """Makes a Pmf."""
        from .pmf import Pmf

        if label is None:
            label = self.label
        return Pmf(self, label=label)

    def values(self):
        """Returns a sorted list of values. """
        return self.xs

    def items(self):
        """Returns a sorted sequence of (value, probability) pairs. """
        a = self.ps
        b = np.roll(a, 1)
        b[0] = 0
        return list(zip(self.xs, a-b))

    def shift(self, term):
        """Adds a term to the xs.

        term: how much to add
        """
        new = self.copy()
        # don't use +=, or else an int array + float yields int array
        new.xs = new.xs + term
        return new

    def scale(self, factor):
        """Multiplies the xs by a factor.

        factor: what to multiply by
        """
        new = self.copy()
        # don't use *=, or else an int array * float yields int array
        new.xs = new.xs * factor
        return new

    def prob(self, x):
        """Returns CDF(x), the probability that corresponds to value x.

        Args:
            x: number or sequence of numbers 

        Returns:
            float probability for each numeber
        """
        # Check if a list of values of a single value
        if hasattr(x, '__iter__') and not isinstance(x, str):
            # List of values
            xs = np.asarray(x)
            index = np.searchsorted(self.xs, xs, side='right')
            ps = self.ps[index-1]
            ps[xs < self.xs[0]] = 0.0
            return ps
        else:
            # Single value
            if x < self.xs[0]:
                return 0.0
            index = bisect.bisect(self.xs, x)
            p = self.ps[index-1]
            return p

    def value(self, p):
        """Returns InverseCDF(p), the value that corresponds to probability p.

        Args:
            p: number, or sequence of numbers in the range [0, 1]

        Returns:
            number value, or array of values
        """
        # Check if a list of values of a single value
        if hasattr(p, '__iter__') and not isinstance(p, str):
            # List of values
            ps = np.asarray(p)
            if np.any(ps < 0) or np.any(ps > 1):
                raise ValueError('Probability p must be in range [0, 1]')

            index = np.searchsorted(self.ps, ps, side='left')
            return self.xs[index]
        else:
            if p < 0 or p > 1:
                raise ValueError('Probability p must be in range [0, 1]')
            index = bisect.bisect_left(self.ps, p)
            return self.xs[index]

    def percentile(self, p):
        """Returns the value that corresponds to percentile p.

        Args:
            p: number in the range [0, 100]

        Returns:
            number value
        """
        return self.value(p / 100.0)

    def percentile_rank(self, x):
        """Returns the percentile rank of the value x.

        x: potential value in the CDF

        returns: percentile rank in the range 0 to 100
        """
        return self.prob(x) * 100.0

    def random(self):
        """Chooses a random value from this distribution."""
        return self.value(random.random())

    def sample(self, n):
        """Generates a random sample from this distribution.
        
        n: int length of the sample
        returns: NumPy array
        """
        ps = np.random.random(n)
        return self.value(ps)

    def mean(self):
        """Computes the mean of a CDF.

        Returns:
            float mean
        """
        old_p = 0
        total = 0.0
        for x, new_p in zip(self.xs, self.ps):
            p = new_p - old_p
            total += p * x
            old_p = new_p
        return total

    def credible_interval(self, percentage=90):
        """Computes the central credible interval.

        If percentage=90, computes the 90% CI.

        Args:
            percentage: float between 0 and 100

        Returns:
            sequence of two floats, low and high
        """
        prob = (1 - percentage / 100.0) / 2
        interval = self.value(prob), self.value(1 - prob)
        return interval

    confidence_interval = credible_interval

    def _round(self, multiplier=1000.0):
        """ 
        An entry is added to the cdf only if the percentile differs
        from the previous value in a significant digit, where the number
        of significant digits is determined by multiplier.  The
        default is 1000, which keeps log10(1000) = 3 significant digits.
        """
        # TODO(write this method)
        raise UnimplementedMethodException()

    def render(self, **options):
        """Generates a sequence of points suitable for plotting.

        An empirical CDF is a step function; linear interpolation
        can be misleading.

        Note: options are ignored

        Returns:
            tuple of (xs, ps)
        """
        def interleave(a, b):
            c = np.empty(a.shape[0] + b.shape[0])
            c[::2] = a
            c[1::2] = b
            return c

        a = np.array(self.xs)
        xs = interleave(a, a)
        shift_ps = np.roll(self.ps, 1)
        shift_ps[0] = 0
        ps = interleave(shift_ps, self.ps)
        return xs, ps

    def max(self, k):
        """Computes the CDF of the maximum of k selections from this dist.

        k: int

        returns: new Cdf
        """
        cdf = self.copy()
        cdf.ps **= k
        return cdf

    def plot(self, complement=False, transform=None, **options):
        """Plots a CDF as a line.

        Args:
          cdf: Cdf object
          complement: boolean, whether to plot the complementary CDF
          transform: string, one of 'exponential', 'pareto', 'weibull', 'gumbel'
          options: keyword args passed to pyplot.plot

        Returns:
          ax : matplotlib axes object
          scale : dictionary with the scale options that should be passed to
              Config, Show or Save.
        """
        # Initialise seaborn with config file 
        import seaborn as sb
        sb.set_context(SEABORN_CONFIG['context'])
        sb.set_palette(SEABORN_CONFIG['pallet'])
        sb.set_style(SEABORN_CONFIG['style'])

        xs, ps = self.render()
        xs = np.asarray(xs)
        ps = np.asarray(ps)

        # Strip out scale and plotting config kwrds before calling plot
        scale = dict(xscale='linear', yscale='linear')
        for s in ['xscale', 'yscale']: 
            if s in options:
                scale[s] = options.pop(s)

        plot_configs = dict()
        for n in ['title', 'xlabel', 'ylabel', 'xscale', 'yscale',
                  'xticks', 'yticks', 'axis', 'xlim', 'ylim']:
            if n in options:  
                plot_configs[n] = options.pop(n)

        if transform == 'exponential':
            complement = True
            scale['yscale'] = 'log'

        if transform == 'pareto':
            complement = True
            scale['yscale'] = 'log'
            scale['xscale'] = 'log'

        if complement:
            ps = [1.0-p for p in ps]

        if transform == 'weibull':
            xs = np.delete(xs, -1)
            ps = np.delete(ps, -1)
            ps = [-math.log(1.0-p) for p in ps]
            scale['xscale'] = 'log'
            scale['yscale'] = 'log'

        if transform == 'gumbel':
            xs = np.delete(xs, 0)
            ps = np.delete(ps, 0)
            ps = [-math.log(p) for p in ps]
            scale['yscale'] = 'log'

        options = _underride_dict(options, label=self.label, linewidth=3)

        # Plot and configure axes and legend
        plt.plot(xs, ps, **options)
        options.update(scale)
        options.update(plot_configs) 
        config_current_plot(**options)

        return plt.gca(), scale, {'compliment' : complement}, 


# ---------------- #
# Module Functions #
# ---------------- #

import scipy

def eval_normal_cdf(x, mu=0, sigma=1):
    """Wrapper for evaluating the CDF of a normal distribution"""
    return scipy.stats.norm.cdf(x, loc=mu, scale=sigma)


