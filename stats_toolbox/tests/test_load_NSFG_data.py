"Tests for loading in example data"

import pandas as pd

from nose.tools import assert_equal, assert_almost_equal, assert_true, assert_list_equal

from ..utils.data_loaders import load_fem_preg_2002


def test_load_fem_preg_2002():

    df = load_fem_preg_2002()

    sorted_birth_wieght_frequencies = df.birthwgt_lb.value_counts().sort_index() 

    idx = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.]
    values = [8, 40, 53, 98, 229, 697, 2223, 3049, 1889, 623, 132, 26, 10, 3, 3, 1]
    target = pd.Series(data=values, index=idx)

    # test index and values
    assert_true(all(sorted_birth_wieght_frequencies.index == target.index))
    assert_true(all(sorted_birth_wieght_frequencies.values == target.values))

