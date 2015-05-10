"""Probabiity density function related classes and methods. """

import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from .shared import UnimplementedMethodException, _underride_dict, config_current_plot
from ..config import SEABORN_CONFIG


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


    def plot(self, **options):
        """Plots the Pdf as a line.

        Args:
          pdf: Pdf, Pmf, or Hist object
          options: keyword args passed to pyplot.plot
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

        low, high = options.pop('low', None), options.pop('high', None)
        n = options.pop('n', 101)

        xs, ps = self.render(low=low, high=high, n=n)
        options = _underride_dict(options, label=self.label)

        ax = plt.plot(xs, ps, **options)
        config_current_plot(**plot_configs)

        return ax


class NormalPdf(Pdf):

    """Represents the PDF of a Normal distribution."""

    def __init__(self, mu=0, sigma=1, label=None):
        """Constructs a Normal Pdf with given mu and sigma.

        mu: mean
        sigma: standard deviation
        label: string
        """
        self.mu = mu
        self.sigma = sigma
        self.label = label if label is not None else '_nolegend_'

    def __str__(self):
        """ The default String representation of an object. """
        return 'NormalPdf(%f, %f)' % (self.mu, self.sigma)

    def get_linspace(self):
        """Get a linspace for plotting.

        Returns: numpy array
        """
        low, high = self.mu-3*self.sigma, self.mu+3*self.sigma
        return np.linspace(low, high, 101)

    def density(self, xs):
        """Evaluates this Pdf at xs.

        xs: scalar or sequence of floats

        returns: float or NumPy array of probability density
        """
        return stats.norm.pdf(xs, self.mu, self.sigma)


class ExponentialPdf(Pdf):

    """Represents the PDF of an exponential distribution."""

    def __init__(self, lam=1, label=None):
        """Constructs an exponential Pdf with given parameter.

        lam: rate parameter
        label: string
        """
        self.lam = lam
        self.label = label if label is not None else '_nolegend_'

    def __str__(self):
        """ The default String representation of an object. """
        return 'ExponentialPdf(%f)' % (self.lam)

    def get_linspace(self):
        """Get a linspace for plotting.

        Returns: numpy array
        """
        low, high = 0, 5.0/self.lam
        return np.linspace(low, high, 101)

    def density(self, xs):
        """Evaluates this Pdf at xs.

        xs: scalar or sequence of floats

        returns: float or NumPy array of probability density
        """
        return stats.expon.pdf(xs, scale=1.0/self.lam)


class EstimatedPdf(Pdf):

    """Represents a PDF estimated by KDE."""

    def __init__(self, sample, label=None):
        """Estimates the density function based on a sample.

        sample: sequence of data
        label: string
        """
        self.label = label if label is not None else '_nolegend_'
        self.kde = stats.gaussian_kde(sample)
        low = min(sample)
        high = max(sample)
        self.linspace = np.linspace(low, high, 101)

    def __str__(self):
        """ The default String representation of an object. """
        return 'EstimatedPdf(label=%s)' % str(self.label)

    def get_linspace(self):
        """Get a linspace for plotting.

        Returns: numpy array
        """
        return self.linspace

    def density(self, xs):
        """Evaluates this Pdf at xs.

        returns: float or NumPy array of probability density
        """
        return self.kde.evaluate(xs)
