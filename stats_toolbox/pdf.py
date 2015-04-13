"""Probabiity density function related classes and methods. """

import numpy as np

from .common import UnimplementedMethodException


class Pdf(object):

    """Represents a probability density function (PDF)."""

    def density(self, x):
        """Evaluates this Pdf at x.

        Returns: float or NumPy array of probability density
        """
        raise UnimplementedMethodException()

    def get_linspace(self):
        """Get a linspace for plotting.

        Not all subclasses of Pdf implement this.

        Returns: numpy array
        """
        raise UnimplementedMethodException()

    def to_pmf(self, **options):
        """Convert object to a discrete version of this Pdf.

        Args: options can include
            label: string
            low: low end of range
            high: high end of range
            n: number of places to evaluate

        Returns: new Pmf
        """
        from .pmf import Pmf

        label = options.pop('label', '')
        xs, ds = self.render(**options)
        return Pmf(dict(zip(xs, ds)), label=label)

    def render(self, **options):
        """Generates a sequence of points suitable for plotting.

        If options includes low and high, it must also include n;
        in that case the density is evaluated an n locations between
        low and high, including both.

        If options includes xs, the density is evaluate at those location.

        Otherwise, self.GetLinspace is invoked to provide the locations.

        Returns:
            tuple of (xs, densities)
        """
        low, high = options.pop('low', None), options.pop('high', None)
        if low is not None and high is not None:
            n = options.pop('n', 101)
            xs = np.linspace(low, high, n)
        else:
            xs = options.pop('xs', None)
            if xs is None:
                xs = self.get_linspace()
            
        ds = self.density(xs)
        return xs, ds

    def items(self):
        """Generates a sequence of (value, probability) pairs. """
        return zip(*self.render())
