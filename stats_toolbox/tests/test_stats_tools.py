"""This file contains tests for the tools and utilities of the stats_toolbox module

The code to use the nose module for unit testing.

License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from nose.tools import assert_equal, assert_almost_equal, assert_true, assert_list_equal

from collections import Counter
import numpy as np
import pandas as pd

from ..core.hist import Hist
from ..core.pmf import Pmf
from ..core.cdf import Cdf

from ..core.tools import resample, resample_rows


def test_resample():
    """Test weighted and non weighted resampling of elements in a list."""
    arr = np.arange(30)
    
    # Without weights
    sample_arr = resample(arr)
    assert_equal(len(arr), len(sample_arr))

    sample_arr = resample(arr, n=50)
    assert_equal(50, len(sample_arr))

    # With weights
    weights = arr.copy()
    sample_arr = resample(arr, n=100, weights=None)
    weighted_sample_arr = resample(arr, n=100, weights=weights)
    assert_true(sample_arr.mean() < weighted_sample_arr.mean())
    

def test_resample_rows():
    """Test weighted and non weighted resampling of rows in a dataframe."""
    d = {'val': np.arange(30), 
         'weights': np.arange(30)}
    df = pd.DataFrame(d)
    
    # Without weights
    sample_df = resample_rows(df)
    assert_equal(len(df), len(sample_df))

    sample_df = resample_rows(df, n=50)
    assert_equal(50, len(sample_df))

    # With weights
    sample_df = resample_rows(df, n=100, weights_col=None)
    weighted_sample_df = resample_rows(df, n=100, weights_col='weights')
    assert_true(df.val.mean() < weighted_sample_df.val.mean())

