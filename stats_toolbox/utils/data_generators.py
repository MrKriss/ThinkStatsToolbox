"""Data generators for characteristic distributions. 
"""

import numpy as np
import scipy 

from ..core.tools import jitter
from ..core.cdf import Cdf

# Point Evaluaters

def eval_normal_cdf(x, mu=0, sigma=1):
    """Wrapper to evaluate a the cdf of a Normal distribution."""
    return scipy.stats.norm.cdf(x, loc=mu, scale=sigma)


# Sequence evaluaters

def render_normal_probability(ys, jitter=0.0):
    """Generates data for a normal probability plot.

    ys: sequence of values
    jitter (float) : magnitude of jitter added to the ys 

    returns: 
        xs, ys : numpy arrays 
    """
    n = len(ys)
    xs = np.random.normal(0, 1, n)
    xs.sort()
    
    if jitter:
        ys = jitter(ys, jitter)
    else:
        ys = np.array(ys)
    ys.sort()

    return xs, ys


def render_expo_cdf(lam, low, high, n=101):
    """Generates sequences of xs and ps for an exponential CDF.

    lam: parameter
    low: float
    high: float
    n: number of points to render

    returns: numpy arrays (xs, ps)
    """
    xs = np.linspace(low, high, n)
    ps = 1 - np.exp(-lam * xs)
    #ps = stats.expon.cdf(xs, scale=1.0/lam)
    return xs, ps


def render_normal_cdf(mu, sigma, low, high, n=101):
    """Generates sequences of xs and ps for a Normal CDF.

    mu: parameter
    sigma: parameter
    low: float
    high: float
    n: number of points to render

    returns: numpy arrays (xs, ps)
    """
    xs = np.linspace(low, high, n)
    ps = stats.norm.cdf(xs, mu, sigma)
    return xs, ps


def render_pareto_cdf(xmin, alpha, low, high, n=50):
    """Generates sequences of xs and ps for a Pareto CDF.

    xmin: parameter
    alpha: parameter
    low: float
    high: float
    n: number of points to render

    returns: numpy arrays (xs, ps)
    """
    if low < xmin:
        low = xmin
    xs = np.linspace(low, high, n)
    ps = 1 - (xs / xmin) ** -alpha
    #ps = stats.pareto.cdf(xs, scale=xmin, b=alpha)
    return xs, ps