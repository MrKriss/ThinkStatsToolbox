"""This file contains tests for the core classes of the stats_toolbox module

Most of these tests are taken from Allen B. Downey test suite 
for the thinkstats2.py

My modifications changed the code to use the nose module for unit testing.

License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import random

from nose.tools import assert_equal, assert_almost_equal, assert_true, assert_list_equal

from collections import Counter
import numpy as np

from ..core.hist import Hist
from ..core.pmf import Pmf
from ..core.pdf import Pdf, NormalPdf, ExponentialPdf, EstimatedPdf
from ..core.cdf import Cdf


# Difference allowed in float comparisons
TOLERANCE = 0.5 * 10**-7


class TestHist():

    "Tests for the Histogram Object"

    def test_hist(self):
        """ Test Histogram object construction."""
        hist = Hist('Christopher Musselle')
        assert len(str(hist)) == 110

        assert len(hist) == 13
        assert hist.freq('l') == 2
        assert hist.freq('s') == 3

        hist = Hist(Counter('Christopher Musselle'))
        assert len(hist) == 13
        assert hist.freq('l') == 2

        hist2 = Hist('ellessuM rehpotsirhC')
        assert hist == hist2 

        pmf = hist.to_pmf()
        assert abs(pmf.prob('e') - 0.15) < TOLERANCE

    def test_hist_summary_stats(self):
        """ Test Summary statistics for Histogram Object """
        hist1 = Hist(np.repeat(np.arange(0, 20, 2), np.arange(20, 0, -2)))
        target = np.repeat(np.arange(0, 20, 2), np.arange(20, 0, -2))

        assert abs(hist1.mean() - target.mean()) < TOLERANCE
        assert abs(hist1.std() - target.std()) < TOLERANCE
        assert abs(hist1.var() - target.var()) < TOLERANCE


class TestPmf:

    """ Tests for the PMF object """

    def test_pmf(self):
        """ Test pmf construction """
        pmf = Pmf('allen')
        # this one might not be a robust test
        assert len(str(pmf)) == 45

        assert len(pmf) == 4
        assert pmf.prob('l') == 0.4
        assert pmf['l'] == 0.4
        assert pmf.percentile(50) == 'l'

        pmf = Pmf(Counter('allen'))
        assert len(pmf) == 4
        assert pmf.prob('l') == 0.4

        pmf = Pmf(pmf)
        assert len(pmf) == 4
        assert pmf.prob('l') == 0.4

        pmf2 = pmf.copy()
        assert pmf == pmf2

        xs, ys = pmf.render()
        assert tuple(xs) == tuple(sorted(pmf.values()))
        
    def test_pmf_add_sub(self):
        """ Test add and sub methods """
        pmf = Pmf([1, 2, 3, 4, 5, 6])

        pmf1 = pmf + 1
        assert abs(pmf1.mean() - 4.5) < TOLERANCE

        pmf2 = pmf + pmf
        assert abs(pmf2.mean() - 7.0) < TOLERANCE

        pmf3 = pmf - 1
        assert abs(pmf3.mean() - 2.5) < TOLERANCE

        pmf4 = pmf - pmf
        assert abs(pmf4.mean() - 0) < TOLERANCE

    def test_pmf_mul_div(self):
        """ Test mult and div methods """
        pmf = Pmf([1, 2, 3, 4, 5, 6])

        pmf1 = pmf * 2
        assert abs(pmf1.mean() - 7) < TOLERANCE

        pmf2 = pmf * pmf
        assert abs(pmf2.mean() - 12.25) < TOLERANCE

        pmf3 = pmf / 2
        assert abs(pmf3.mean() - 1.75) < TOLERANCE

        pmf4 = pmf / pmf
        assert abs(pmf4.mean() - 1.4291667) < TOLERANCE

    def test_pmf_prob_less(self):
        """ Test prob less method """
        d6 = Pmf(range(1, 7))
        assert d6.prob_less(4) == 0.5
        assert d6.prob_greater(3) == 0.5
        two = d6 + d6
        three = two + d6
        assert abs((two > three) - 0.15200617284) < TOLERANCE
        assert abs((two < three) - 0.778549382716049) < TOLERANCE
        assert abs(two.prob_greater(three) - 0.15200617284) < TOLERANCE
        assert abs(two.prob_less(three) - 0.778549382716049) < TOLERANCE

    def test_pmf_max(self):
        """ Test max method """
        d6 = Pmf(range(1, 7))
        two = d6 + d6
        three = two + d6
        cdf = three.max(6)
        assert abs(cdf[14] - 0.558230962626) < TOLERANCE


class TestCdf():

    """ Testing methods for the CDF object. """

    def test_cdf(self):
        """ Test construction methods """
        t = [1, 2, 2, 3, 5]
        pmf = Pmf(t)
        hist = Hist(t)

        cdf = Cdf(pmf)
        assert len(str(cdf)) == 37

        assert cdf[0] == 0
        assert abs(cdf[1] - 0.2) < TOLERANCE
        assert abs(cdf[2] - 0.6) < TOLERANCE
        assert abs(cdf[3] - 0.8) < TOLERANCE
        assert abs(cdf[4] - 0.8) < TOLERANCE
        assert abs(cdf[5] - 1) < TOLERANCE
        assert abs(cdf[6] - 1) < TOLERANCE

        xs = range(7)
        ps = cdf.prob(xs)
        for p1, p2 in zip(ps, [0, 0.2, 0.6, 0.8, 0.8, 1, 1]):
            assert abs(p1 - p2) < TOLERANCE

        assert cdf.value(0) == 1
        assert cdf.value(0.1) == 1
        assert cdf.value(0.2) == 1
        assert cdf.value(0.3) == 2
        assert cdf.value(0.4) == 2
        assert cdf.value(0.5) == 2
        assert cdf.value(0.6) == 2
        assert cdf.value(0.7) == 3
        assert cdf.value(0.8) == 3
        assert cdf.value(0.9) == 5
        assert cdf.value(1) == 5

        ps = np.linspace(0, 1, 11)
        xs = cdf.value(ps)
        assert all(xs == [1, 1, 1, 2, 2, 2, 2, 3, 3, 5, 5])

        np.random.seed(17)
        xs = cdf.sample(7)
        assert_list_equal(xs.tolist(), [2, 2, 1, 1, 3, 3, 3])

        # when you make a Cdf from a Pdf, you might get some floating
        # point representation error
        assert len(cdf) == 4
        assert abs(cdf.prob(2) - 0.6) < TOLERANCE
        assert abs(cdf[2] - 0.6) < TOLERANCE
        assert cdf.value(0.6) == 2

        cdf = pmf.to_cdf()
        assert len(cdf) == 4
        assert abs(cdf.prob(2) - 0.6) < TOLERANCE
        assert cdf.value(0.6) == 2

        cdf = Cdf(pmf.d)
        assert len(cdf) == 4
        assert abs(cdf.prob(2) - 0.6) < TOLERANCE
        assert cdf.value(0.6) == 2

        cdf = Cdf(hist)
        assert len(cdf) == 4
        assert cdf.prob(2) == 0.6
        assert cdf.value(0.6) == 2

        cdf = Cdf(t)
        assert len(cdf) == 4
        assert cdf.prob(2) == 0.6
        assert cdf.value(0.6) == 2

        cdf = Cdf(Counter(t))
        assert len(cdf) == 4
        assert cdf.prob(2) == 0.6
        assert cdf.value(0.6) == 2

        cdf2 = cdf.copy()
        assert cdf2.prob(2) == 0.6
        assert cdf2.value(0.6) == 2
        
    def test_shift(self):
        """ Test shift method """
        t = [1, 2, 2, 3, 5]
        cdf = Cdf(t)
        cdf2 = cdf.shift(1)
        assert_equal(cdf[1], cdf2[2])

    def test_scale(self):
        """ Test scale method """
        t = [1, 2, 2, 3, 5]
        cdf = Cdf(t)
        cdf2 = cdf.scale(2)
        assert_equal(cdf[2], cdf2[4])

    def test_cdf_render(self):
        """ Test render method"""
        t = [1, 2, 2, 3, 5]
        cdf = Cdf(t)
        xs, ps = cdf.render()
        assert_equal(xs[0], 1)
        assert_equal(ps[2], 0.2)
        assert_equal(sum(xs), 22)
        assert_equal(sum(ps), 4.2)
        
    def test_pmf_from_cdf(self):
        """ Test conversion to pmf """
        t = [1, 2, 2, 3, 5]
        pmf = Pmf(t)
        cdf = Cdf(pmf)
        pmf2 = Pmf(cdf)
        for x in pmf.values():
            assert_almost_equal(pmf[x], pmf2[x])

        pmf3 = cdf.to_pmf()
        for x in pmf.values():
            assert_almost_equal(pmf[x], pmf3[x])


class TestPdf:

    """" Test methods for the Pdf object """

    def test_NormalPdf(self):
        """ Test construction methods for normal Pdf """
        pdf = NormalPdf(mu=1, sigma=2)
        assert_equal(len(str(pdf)), 29)
        assert_almost_equal(pdf.density(3), 0.12098536226)

        pmf = pdf.to_pmf()
        assert_almost_equal(pmf[1.0], 0.0239951295619)
        xs, ps = pdf.render()
        assert_equal(xs[0], -5.0)
        assert_almost_equal(ps[0], 0.0022159242059690038)

        pmf = Pmf(pdf)
        assert_almost_equal(pmf[1.0], 0.0239951295619)
        xs, ps = pmf.render()
        assert_equal(xs[0], -5.0)
        assert_almost_equal(ps[0], 0.00026656181123)
        
        cdf = Cdf(pdf)
        assert_almost_equal(cdf[1.0], 0.51199756478094904)
        xs, ps = cdf.render()
        assert_equal(xs[0], -5.0)
        assert_almost_equal(ps[0], 0.0)

    def test_ExponentialPdf(self):
        """Test construction of exponential pdf """
        pdf = ExponentialPdf(lam=0.5)
        assert_equal(len(str(pdf)), 24)
        assert_almost_equal(pdf.density(3), 0.11156508007421491)
        pmf = pdf.to_pmf()
        assert_almost_equal(pmf[1.0], 0.02977166586593202)
        xs, ps = pdf.render()
        assert_equal(xs[0], 0.0)
        assert_almost_equal(ps[0], 0.5)
        
    def test_EstimatedPdf(self):
        """Test constuction of Estimated pdf based on KDE """
        pdf = EstimatedPdf([1, 2, 2, 3, 5])
        assert_equal(len(str(pdf)), 30)
        assert_almost_equal(pdf.density(3)[0], 0.19629968)
        pmf = pdf.to_pmf()
        assert abs(pmf[1.0] - 0.010172282816895044) < 0.5 * 10**-7
        pmf = pdf.to_pmf(low=0, high=6)
        assert abs(pmf[0.0] - 0.0050742294053582942) < 0.5 * 10**-7



        
#     def testEvalNormalCdf(self):
#         p = thinkstats2.EvalNormalCdf(0)
#         assert_almost_equal(p, 0.5)

#         p = thinkstats2.EvalNormalCdf(2, 2, 3)
#         assert_almost_equal(p, 0.5)

#         p = thinkstats2.EvalNormalCdf(1000, 0, 1)
#         assert_almost_equal(p, 1.0)

#         p = thinkstats2.EvalNormalCdf(-1000, 0, 1)
#         assert_almost_equal(p, 0.0)

#         x = thinkstats2.EvalNormalCdfInverse(0.95, 0, 1)
#         assert_almost_equal(x, 1.64485362695)
#         x = thinkstats2.EvalNormalCdfInverse(0.05, 0, 1)
#         assert_almost_equal(x, -1.64485362695)


# class Test(unittest.TestCase):

#     def testOdds(self):
#         p = 0.75
#         o = thinkstats2.Odds(p)
#         assert_equal(o, 3)

#         p = thinkstats2.Probability(o)
#         assert_equal(p, 0.75)
        
#         p = thinkstats2.Probability2(3, 1)
#         assert_equal(p, 0.75)
        

#     def testBinomialCoef(self):
#         res = thinkstats2.BinomialCoef(10, 3)
#         assert_equal(round(res), 120)

#         res = thinkstats2.BinomialCoef(100, 4)
#         assert_equal(round(res), 3921225)

#     def testInterpolator(self):
#         xs = [1, 2, 3]
#         ys = [4, 5, 6]
#         interp = thinkstats2.Interpolator(xs, ys)

#         y = interp.Lookup(1)
#         assert_almost_equal(y, 4)

#         y = interp.Lookup(2)
#         assert_almost_equal(y, 5)

#         y = interp.Lookup(3)
#         assert_almost_equal(y, 6)

#         y = interp.Lookup(1.5)
#         assert_almost_equal(y, 4.5)

#         y = interp.Lookup(2.75)
#         assert_almost_equal(y, 5.75)

#         x = interp.Reverse(4)
#         assert_almost_equal(x, 1)

#         x = interp.Reverse(6)
#         assert_almost_equal(x, 3)

#         x = interp.Reverse(4.5)
#         assert_almost_equal(x, 1.5)

#         x = interp.Reverse(5.75)
#         assert_almost_equal(x, 2.75)







        


#     def testEvalPoissonPmf(self):
#         p = thinkstats2.EvalPoissonPmf(2, 1)
#         assert_almost_equal(p, 0.1839397205)

#     def testCov(self):
#         t = [0, 4, 7, 3, 8, 1, 6, 2, 9, 5]
#         a = np.array(t)
#         t2 = [5, 4, 3, 0, 8, 9, 7, 6, 2, 1]

#         assert_almost_equal(thinkstats2.Cov(t, a), 8.25)
#         assert_almost_equal(thinkstats2.Cov(t, -a), -8.25)

#         assert_almost_equal(thinkstats2.Corr(t, a), 1)
#         assert_almost_equal(thinkstats2.Corr(t, -a), -1)
#         assert_almost_equal(thinkstats2.Corr(t, t2), -0.1878787878)
        
#         assert_almost_equal(thinkstats2.SpearmanCorr(t, -a), -1)
#         assert_almost_equal(thinkstats2.SpearmanCorr(t, t2), -0.1878787878)
        
#     def testReadStataDct(self):
#         dct = thinkstats2.ReadStataDct('2002FemPreg.dct')
#         assert_equal(len(dct.variables), 243)
#         assert_equal(len(dct.colspecs), 243)
#         assert_equal(len(dct.names), 243)
#         assert_equal(dct.colspecs[-1][1], -1)

