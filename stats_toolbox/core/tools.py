"""Code for a variety of useful stats functions. """

import math
import numpy as np
import pandas as pd

from .cdf import Cdf


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


def cov(xs, ys, meanx=None, meany=None):
    """Computes covariance Cov(X, Y).

    Args:
        xs: sequence of values
        ys: sequence of values
        meanx: optional float mean of xs
        meany: optional float mean of ys

    Returns:
        cov(X, Y)
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    if meanx is None:
        meanx = np.mean(xs)
    if meany is None:
        meany = np.mean(ys)

    cov = np.dot(xs-meanx, ys-meany) / len(xs)
    return cov


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


def pearson_corr(xs, ys):
    """Computes Pearsons Correlation Coefficient between two sequences.

    Args:
        xs: sequence of values
        ys: sequence of values

    Returns:
        Corr(X, Y)
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    meanx = xs.mean()
    varx = xs.var()

    meany = ys.mean()
    vary = ys.var()

    corr = cov(xs, ys, meanx, meany) / math.sqrt(varx * vary)

    return corr


def pearson_median_skewness(xs):
    """ Computes Pearson’s median skewness of a distribution.

    Pearson’s median skewness coefficient is a measure of skewness based on the difference 
    between the sample mean and median:
                        
                        gp = 3 * (mean−median) / std
    """
    xs = np.asarray(xs)
    median = xs.median()
    mean = xs.mean()
    var = xs.var()
    std = math.sqrt(var)
    gp = 3 * (mean - median) / std
    return gp


def resample(xs, n=None, weights=None):
    """Draw a new sample from xs with the same length as xs with replacement.

    xs: sequence of values
    n: sample size (default: len(xs))
    weights: a sequence of values. If given, resampling probabilities are proportional 
        to the given weights. Must be the same length as xs. 

    returns: NumPy array
    """
    if n is None:
        n = len(xs)

    if weights is not None:
        assert len(xs) == len(weights), 'Length of xs and weights must be the same.'
        cdf = Cdf(dict(enumerate(weights)))
        # Randomly generate new indicies with replacement.
        indices = cdf.sample(n)
        xs = np.asarray(xs)
        sample = xs[indices]
        return sample
    else:
        return np.random.choice(xs, n, replace=True)


def resample_rows(df, n=None, weights_col=None):
    """Resamples rows from a DataFrame with replacement.

    df: DataFrame
    n: number of rows to sample, defaults to sample length as df
    weights_col: if given, resampling probabilities are proportional to the 
        given columnname of the DataFrame. 

    returns: DataFrame
    """
    if n is None:
        n = len(df)

    if weights_col is not None:
        weights = df[weights_col]
        # Calculate cdf of the row indicies based on the weights column 
        cdf = Cdf(dict(weights))
        # Randomly generate new indicies with replacement.
        indices = cdf.sample(n)
        sample = df.loc[indices]
        return sample
    else:
        # simply resample rows with replacement
        return sample_rows(df, n, replace=True)


def sample_rows(df, nrows, replace=False):
    """Return a sample of rows from a dataframe.

    Args:
        df : a pandas DataFrame object
        nrows : number of rows to return
        replace : whether to sample with or withour replacement
    """
    indices = np.random.choice(df.index, nrows, replace=replace)
    sample = df.loc[indices]
    return sample


def serial_corr(series, lag=1):
    """Computes the serial correlation of a series.

    series: Series
    lag: integer number of intervals to shift

    returns: float correlation
    """
    xs = series[lag:]
    ys = series.shift(lag)[lag:]
    corr = pearson_corr(xs, ys)
    return corr


def spearman_corr(xs, ys):
    """Computes Spearman's rank correlation coefficient.

    Spearman’s rank correlation is an alternative to pearsons_corr that
    mitigates the effect of outliers and skewed distributions.

    Args:
        xs: sequence of values
        ys: sequence of values

    Returns:
        float Spearman's correlation
    """
    xranks = pd.Series(xs).rank()
    yranks = pd.Series(ys).rank()
    return pearson_corr(xranks, yranks)


def trimmed_mean(t, p=0.01):
    """Computes the trimmed mean of a sequence of numbers.

    Args:
        t: sequence of numbers
        p: fraction of values to trim off each end

    Returns:
        float
    """
    n = int(p * len(t))
    t = sorted(t)[n:-n]
    xs = np.asarray(t)
    return xs.mean()


def trimmed_var(t, p=0.01):
    """Computes the trimmed mean and variance of a sequence of numbers.

    Side effect: sorts the list.

    Args:
        t: sequence of numbers
        p: fraction of values to trim off each end

    Returns:
        float
    """
    n = int(p * len(t))
    t = sorted(t)[n:-n]
    xs = np.asarray(t)
    return xs.var()




