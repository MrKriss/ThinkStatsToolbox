"""Probability mass function related Classes and Metods. """

import math
import random 
import warnings

import numpy as np
import matplotlib.pyplot as plt

from .shared import _DictWrapper, _underride_dict
from ..config import SEABORN_CONFIG


class Pmf(_DictWrapper):

    """Represents a probability mass function.
    
    Values can be any hashable type; probabilities are floating-point.
    The resulting Pmfs are not necessarily normalized.
    """

    def __init__(self, obj=None, label=None):
        """Initialise with base dictionary init then normalise."""
        super().__init__(obj, label)

        if len(self) > 0:
            self.normalize()

    def prob(self, x, default=0):
        """Gets the probability associated with the value x.

        Args:
            x: number value or sequence of values
            default: value to return if the key is not there

        Returns:
            float probability
        """
        # Check if a list of values of a single value
        if hasattr(x, '__iter__') and not isinstance(x, str):
            return [self.d.get(y, default) for y in x]
        else:
            return self.d.get(x, default)

    def percentile(self, percentage):
        """Computes a percentile of a given Pmf.

        Note: this is not super efficient.  If you are planning
        to compute more than a few percentiles, compute the Cdf.

        percentage: float 0-100

        returns: value from the Pmf
        """
        p = percentage / 100.0
        total = 0
        for val, prob in sorted(self.items()):
            total += prob
            if total >= p:
                return val

    def prob_greater(self, x):
        """Probability that a sample from this Pmf exceeds x.

        If x is a second Pmf: returns probability that a value from this pmf is 
        greater than a value from this second pmf.

        x: number or second Pmf object

        returns: float probability
        """
        if isinstance(x, _DictWrapper):
            total = 0.0
            for v1, p1 in self.items():
                for v2, p2 in x.items():
                    if v1 > v2:
                        total += p1 * p2
            return total
        else:
            t = [prob for (val, prob) in self.d.items() if val > x]
            return sum(t)

    def prob_less(self, x):
        """Probability that a sample from this Pmf is less than x.

        If x is a second Pmf: returns probability that a value from this pmf is 
        less than a value from this second pmf.

        x: number or second Pmf object

        returns: float probability
        """
        if isinstance(x, _DictWrapper):
            total = 0.0
            for v1, p1 in self.items():
                for v2, p2 in x.items():
                    if v1 < v2:
                        total += p1 * p2
            return total
        else:
            t = [prob for (val, prob) in self.d.items() if val < x]
            return sum(t)

    def __lt__(self, obj):
        """Less than.

        obj: number or _DictWrapper

        returns: float probability
        """
        return self.prob_less(obj)

    def __gt__(self, obj):
        """Greater than.

        obj: number or _DictWrapper

        returns: float probability
        """
        return self.prob_greater(obj)

    def __ge__(self, obj):
        """Greater than or equal.

        obj: number or _DictWrapper

        returns: float probability
        """
        return 1 - (self < obj)

    def __le__(self, obj):
        """Less than or equal.

        obj: number or _DictWrapper

        returns: float probability
        """
        return 1 - (self > obj)

    def normalize(self, fraction=1.0):
        """Normalizes this PMF so the sum of all probs is fraction.

        Args:
            fraction: what the total should be after normalization

        Returns: the total probability before normalizing
        """
        if self.log:
            raise ValueError("Normalize: Pmf is under a log transform")

        total = self.total()
        if total == 0.0:
            raise ValueError('Normalize: total probability is zero.')

        factor = fraction / total
        for x in self.d:
            self.d[x] *= factor

        return total

    def random(self):
        """Chooses a random element from this PMF.

        Note: this is not very efficient.  If you plan to call
        this more than a few times, consider converting to a CDF.

        Returns:
            float value from the Pmf
        """
        target = random.random()
        total = 0.0
        for x, p in self.d.items():
            total += p
            if total >= target:
                return x

        # we shouldn't get here
        raise ValueError('Random: Pmf might not be normalized.')

    def mean(self):
        """Computes the mean of a PMF.

        Returns:
            float mean
        """
        mean = 0.0
        for x, p in self.d.items():
            mean += p * x
        return mean

    def var(self, mu=None):
        """Computes the variance of a PMF.

        mu: the point around which the variance is computed;
                if omitted, computes the mean

        returns: float variance
        """
        if mu is None:
            mu = self.mean()

        var = 0.0
        for x, p in self.d.items():
            var += p * (x - mu) ** 2
        return var

    def std(self, mu=None):
        """Computes the standard deviation of a PMF.

        mu: the point around which the variance is computed;
                if omitted, computes the mean

        returns: float standard deviation
        """
        var = self.var(mu)
        return math.sqrt(var)

    def maximum_likelihood(self):
        """Returns the value with the highest probability.

        Returns: float probability
        """
        _, val = max((prob, val) for val, prob in self.items())
        return val

    def credible_interval(self, percentage=90):
        """Computes the central credible interval.

        If percentage=90, computes the 90% CI.

        Args:
            percentage: float between 0 and 100

        Returns:
            sequence of two floats, low and high
        """
        cdf = self.to_cdf()
        return cdf.credible_interval(percentage)

    def __add__(self, other):
        """Computes the Pmf of the sum of values drawn from self and other.

        other: another Pmf or a scalar

        returns: new Pmf
        """
        try:
            return self.add_pmf(other)
        except AttributeError:
            return self.add_constant(other)

    def add_pmf(self, other):
        """Computes the Pmf of the sum of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.items():
            for v2, p2 in other.items():
                pmf.incr(v1 + v2, p1 * p2)
        return pmf

    def add_constant(self, other):
        """Computes the Pmf of the sum a constant and values from self.

        other: a number

        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.items():
            pmf.set(v1 + other, p1)
        return pmf

    def __sub__(self, other):
        """Computes the Pmf of the diff of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        try:
            return self.sub_pmf(other)
        except AttributeError:
            return self.add_constant(-other)

    def sub_pmf(self, other):
        """Computes the Pmf of the diff of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.items():
            for v2, p2 in other.items():
                pmf.incr(v1 - v2, p1 * p2)
        return pmf

    def __mul__(self, other):
        """Computes the Pmf of the product of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        try:
            return self.mul_pmf(other)
        except AttributeError:
            return self.mul_constant(other)

    def mul_pmf(self, other):
        """Computes the Pmf of the diff of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.items():
            for v2, p2 in other.items():
                pmf.incr(v1 * v2, p1 * p2)
        return pmf

    def mul_constant(self, other):
        """Computes the Pmf of the product of a constant and values from self.

        other: a number

        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.items():
            pmf.set(v1 * other, p1)
        return pmf

    def __div__(self, other):
        """Computes the Pmf of the ratio of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        try:
            return self.div_pmf(other)
        except AttributeError:
            return self.mul_constant(1 / other)

    __truediv__ = __div__

    def div_pmf(self, other):
        """Computes the Pmf of the ratio of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        pmf = Pmf()
        for v1, p1 in self.items():
            for v2, p2 in other.items():
                pmf.incr(v1 / v2, p1 * p2)
        return pmf

    def max(self, k):
        """Computes the CDF of the maximum of k selections from this dist.

        k: int

        returns: new Cdf
        """
        cdf = self.to_cdf()
        return cdf.max(k)

    def plot(self, style='', **options):
        """Plots a Pmf or Hist as a line.

        Args:
          pmf: Hist or Pmf object
          style: style string passed along to pyplot.plot
          options: keyword args passed to pyplot.plot

        Returns:
            ax : Matplotlib axese object
        """
        # Initialise seaborn with config file 
        import seaborn as sb
        sb.set_context(SEABORN_CONFIG['context'])
        sb.set_palette(SEABORN_CONFIG['pallet'])
        sb.set_style(SEABORN_CONFIG['style'])
        
        xs, ys = self.render()

        width = options.pop('width', None)
        if width is None:
            try:
                width = np.diff(xs).min()
            except TypeError:
                warnings.warn("Pmf: Can't compute bar width automatically. "
                              "Check for non-numeric types in Pmf. "
                              "Or try providing width option.")
                raise TypeError
        
        points = []

        lastx = np.nan
        lasty = 0
        for x, y in zip(xs, ys):
            if (x - lastx) > 1e-5:
                points.append((lastx, 0))
                points.append((x, 0))

            points.append((x, lasty))
            points.append((x, y))
            points.append((x+width, y))

            lastx = x + width
            lasty = y
        points.append((lastx, 0))
        pxs, pys = zip(*points)

        align = options.pop('align', 'center')
        if align == 'center':
            pxs = np.array(pxs) - width/2.0
        if align == 'right':
            pxs = np.array(pxs) - width

        options = _underride_dict(options, label=self.label)
        options = _underride_dict(options, linewidth=3, alpha=0.8)

        ax = plt.plot(pxs, pys, style, **options)

        return ax







