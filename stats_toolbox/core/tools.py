"""Code for a variety of useful stats functions. """

import numpy as np


def cohen_effect_size(group1, group2):
    """Return Cohen's d statisitc for effect size between two samples.

    Args:
        group1, group2 (pandas Series or array like) : The two samples to compare.

    Returns:
        float : The value for Cohen's d statistic.
    """
    diff = group1.mean() - group2.mean()
    var1 = group1.var()
    var2 = group2.var()
    n1, n2 = len(group1), len(group2)
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / np.sqrt(pooled_var)
    return d


def fit_line(xs, inter, slope):
    """Fits a line to the given data.

    xs: sequence of x

    returns: tuple of numpy arrays (sorted xs, fit ys)
    """
    fit_xs = np.sort(xs)
    fit_ys = inter + slope * fit_xs
    return fit_xs, fit_ys


def jitter(values, jitter=0.5):
    """Jitters the values by adding a uniform variate in (-jitter, jitter).

    values: sequence
    jitter: scalar magnitude of jitter
    
    returns: new numpy array
    """
    n = len(values)
    return np.random.uniform(-jitter, +jitter, n) + values


