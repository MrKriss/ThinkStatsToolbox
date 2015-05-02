"""This file contains tests for the core classes of the stats_toolbox module

Most of these tests are taken from Allen B. Downey test suite 
for the thinkstats2.py

My modifications changed the code to use the nose module for unit testing.

License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from nose.tools import assert_equal, assert_almost_equal, assert_true, assert_list_equal

from collections import Counter
import numpy as np

from ..core.hist import Hist
from ..core.pmf import Pmf
from ..core.cdf import Cdf


def test_hist():
    hist = Hist('Christopher Musselle')
    assert_equal(len(str(hist)), 110)

    assert_equal(len(hist), 13)
    assert_equal(hist.freq('l'), 2)
    assert_equal(hist.freq('s'), 3)

    hist = Hist(Counter('Christopher Musselle'))
    assert_equal(len(hist), 13)
    assert_equal(hist.freq('l'), 2)

    hist2 = Hist('ellessuM rehpotsirhC')
    assert_equal(hist, hist2)

    pmf = hist.to_pmf()
    assert_almost_equal(pmf.prob('e'), 0.15)

def test_hist_summary_stats():
    hist1 = Hist(np.repeat(np.arange(0,20,2), np.arange(20,0,-2)))
    target = np.repeat(np.arange(0,20,2), np.arange(20,0,-2))

    assert_almost_equal(hist1.mean(), target.mean())
    assert_almost_equal(hist1.std(), target.std())
    assert_almost_equal(hist1.var(), target.var())

def test_pmf():
    pmf = Pmf('allen')
    # this one might not be a robust test
    assert_equal(len(str(pmf)), 45)

    assert_equal(len(pmf), 4)
    assert_equal(pmf.prob('l'), 0.4)
    assert_equal(pmf['l'], 0.4)
    assert_equal(pmf.percentile(50), 'l')

    pmf = Pmf(Counter('allen'))
    assert_equal(len(pmf), 4)
    assert_equal(pmf.prob('l'), 0.4)

    pmf = Pmf(pmf)
    assert_equal(len(pmf), 4)
    assert_equal(pmf.prob('l'), 0.4)

    pmf2 = pmf.copy()
    assert_equal(pmf, pmf2)

    xs, ys = pmf.render()
    assert_equal(tuple(xs), tuple(sorted(pmf.values())))        
    
def test_pmf_add_sub():
    pmf = Pmf([1, 2, 3, 4, 5, 6])

    pmf1 = pmf + 1
    assert_almost_equal(pmf1.mean(), 4.5)

    pmf2 = pmf + pmf
    assert_almost_equal(pmf2.mean(), 7.0)

    pmf3 = pmf - 1
    assert_almost_equal(pmf3.mean(), 2.5)

    pmf4 = pmf - pmf
    assert_almost_equal(pmf4.mean(), 0)

def test_pmf_mul_div():
    pmf = Pmf([1, 2, 3, 4, 5, 6])

    pmf1 = pmf * 2
    assert_almost_equal(pmf1.mean(), 7)

    pmf2 = pmf * pmf
    assert_almost_equal(pmf2.mean(), 12.25)

    pmf3 = pmf / 2
    assert_almost_equal(pmf3.mean(), 1.75)

    pmf4 = pmf / pmf
    assert_almost_equal(pmf4.mean(), 1.4291667)

def test_pmf_prob_less():
    d6 = Pmf(range(1, 7))
    assert_equal(d6.prob_less(4), 0.5)
    assert_equal(d6.prob_greater(3), 0.5)
    two = d6 + d6
    three = two + d6
    assert_almost_equal(two > three, 0.15200617284)
    assert_almost_equal(two < three, 0.778549382716049)
    assert_almost_equal(two.prob_greater(three), 0.15200617284)
    assert_almost_equal(two.prob_less(three), 0.778549382716049)

def test_pmf_max():
    d6 = Pmf(range(1, 7))
    two = d6 + d6
    three = two + d6
    cdf = three.max(6)
    assert_almost_equal(cdf[14], 0.558230962626)

def test_cdf():
    t = [1, 2, 2, 3, 5]
    pmf = Pmf(t)
    hist = Hist(t)

    cdf = Cdf(pmf)
    assert_equal(len(str(cdf)), 37)

    assert_equal(cdf[0], 0)
    assert_almost_equal(cdf[1], 0.2)
    assert_almost_equal(cdf[2], 0.6)
    assert_almost_equal(cdf[3], 0.8)
    assert_almost_equal(cdf[4], 0.8)
    assert_almost_equal(cdf[5], 1)
    assert_almost_equal(cdf[6], 1)

    xs = range(7)
    ps = cdf.prob(xs)
    for p1, p2 in zip(ps, [0, 0.2, 0.6, 0.8, 0.8, 1, 1]):
        assert_almost_equal(p1, p2)

    assert_equal(cdf.value(0), 1)
    assert_equal(cdf.value(0.1), 1)
    assert_equal(cdf.value(0.2), 1)
    assert_equal(cdf.value(0.3), 2)
    assert_equal(cdf.value(0.4), 2)
    assert_equal(cdf.value(0.5), 2)
    assert_equal(cdf.value(0.6), 2)
    assert_equal(cdf.value(0.7), 3)
    assert_equal(cdf.value(0.8), 3)
    assert_equal(cdf.value(0.9), 5)
    assert_equal(cdf.value(1), 5)

    ps = np.linspace(0, 1, 11)
    xs = cdf.value(ps)
    assert_true((xs == [1, 1, 1, 2, 2, 2, 2, 3, 3, 5, 5]).all())

    np.random.seed(17)
    xs = cdf.sample(7)
    assert_list_equal(xs.tolist(), [2, 2, 1, 1, 3, 3, 3])

    # when you make a Cdf from a Pdf, you might get some floating
    # point representation error
    assert_equal(len(cdf), 4)
    assert_almost_equal(cdf.prob(2), 0.6)
    assert_almost_equal(cdf[2], 0.6)
    assert_equal(cdf.value(0.6), 2)

    cdf = pmf.to_cdf()
    assert_equal(len(cdf), 4)
    assert_almost_equal(cdf.prob(2), 0.6)
    assert_equal(cdf.value(0.6), 2)

    cdf = Cdf(pmf.d)
    assert_equal(len(cdf), 4)
    assert_almost_equal(cdf.prob(2), 0.6)
    assert_equal(cdf.value(0.6), 2)

    cdf = Cdf(hist)
    assert_equal(len(cdf), 4)
    assert_equal(cdf.prob(2), 0.6)
    assert_equal(cdf.value(0.6), 2)

    cdf = Cdf(t)
    assert_equal(len(cdf), 4)
    assert_equal(cdf.prob(2), 0.6)
    assert_equal(cdf.value(0.6), 2)

    cdf = Cdf(Counter(t))
    assert_equal(len(cdf), 4)
    assert_equal(cdf.prob(2), 0.6)
    assert_equal(cdf.value(0.6), 2)

    cdf2 = cdf.copy()
    assert_equal(cdf2.prob(2), 0.6)
    assert_equal(cdf2.value(0.6), 2)
    
def test_shift():
    t = [1, 2, 2, 3, 5]
    cdf = Cdf(t)
    cdf2 = cdf.shift(1)
    assert_equal(cdf[1], cdf2[2])

def test_scale():
    t = [1, 2, 2, 3, 5]
    cdf = Cdf(t)
    cdf2 = cdf.scale(2)
    assert_equal(cdf[2], cdf2[4])

def test_cdf_render():
    t = [1, 2, 2, 3, 5]
    cdf = Cdf(t)
    xs, ps = cdf.render()
    assert_equal(xs[0], 1)
    assert_equal(ps[2], 0.2)
    assert_equal(sum(xs), 22)
    assert_equal(sum(ps), 4.2)
    
def test_pmf_from_cdf():
    t = [1, 2, 2, 3, 5]
    pmf = Pmf(t)
    cdf = Cdf(pmf)
    pmf2 = Pmf(cdf)
    for x in pmf.values():
        assert_almost_equal(pmf[x], pmf2[x])

    pmf3 = cdf.to_pmf()
    for x in pmf.values():
        assert_almost_equal(pmf[x], pmf3[x])

# def test_Normal_Pdf():
#     pdf = thinkstats2.NormalPdf(mu=1, sigma=2)
#     assert_equal(len(str(pdf)), 29)
#     assert_almost_equal(pdf.Density(3), 0.12098536226)

#     pmf = pdf.MakePmf()
#     assert_almost_equal(pmf[1.0], 0.0239951295619)
#     xs, ps = pdf.Render()
#     assert_equal(xs[0], -5.0)
#     assert_almost_equal(ps[0], 0.0022159242059690038)

#     pmf = thinkstats2.Pmf(pdf)
#     assert_almost_equal(pmf[1.0], 0.0239951295619)
#     xs, ps = pmf.Render()
#     assert_equal(xs[0], -5.0)
#     assert_almost_equal(ps[0], 0.00026656181123)
    
#     cdf = thinkstats2.Cdf(pdf)
#     assert_almost_equal(cdf[1.0], 0.51199756478094904)
#     xs, ps = cdf.Render()
#     assert_equal(xs[0], -5.0)
#     assert_almost_equal(ps[0], 0.0)






# class Test(unittest.TestCase):

#     def testOdds(self):
#         p = 0.75
#         o = thinkstats2.Odds(p)
#         assert_equal(o, 3)

#         p = thinkstats2.Probability(o)
#         assert_equal(p, 0.75)
        
#         p = thinkstats2.Probability2(3, 1)
#         assert_equal(p, 0.75)
        
#     def testMean(self):
#         t = [1, 1, 1, 3, 3, 591]
#         mean = thinkstats2.Mean(t)
#         assert_equal(mean, 100)

#     def testVar(self):
#         t = [1, 1, 1, 3, 3, 591]
#         mean = thinkstats2.Mean(t)
#         var1 = thinkstats2.Var(t)
#         var2 = thinkstats2.Var(t, mean)
        
#         assert_almost_equal(mean, 100.0)
#         assert_almost_equal(var1, 48217.0)
#         assert_almost_equal(var2, 48217.0)

#     def testMeanVar(self):
#         t = [1, 1, 1, 3, 3, 591]
#         mean, var = thinkstats2.MeanVar(t)
        
#         assert_almost_equal(mean, 100.0)
#         assert_almost_equal(var, 48217.0)

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

#     def testTrim(self):
#         t = list(range(100))
#         random.shuffle(t)
#         trimmed = thinkstats2.Trim(t, p=0.05)
#         n = len(trimmed)
#         assert_equal(n, 90)





        
#     def testExponentialPdf(self):
#         pdf = thinkstats2.ExponentialPdf(lam=0.5)
#         assert_equal(len(str(pdf)), 24)
#         assert_almost_equal(pdf.Density(3), 0.11156508007421491)
#         pmf = pdf.MakePmf()
#         assert_almost_equal(pmf[1.0], 0.02977166586593202)
#         xs, ps = pdf.Render()
#         assert_equal(xs[0], 0.0)
#         assert_almost_equal(ps[0], 0.5)
        
#     def testEstimatedPdf(self):
#         pdf = thinkstats2.EstimatedPdf([1, 2, 2, 3, 5])
#         assert_equal(len(str(pdf)), 30)
#         assert_almost_equal(pdf.Density(3)[0], 0.19629968)
#         pmf = pdf.MakePmf()
#         assert_almost_equal(pmf[1.0], 0.010172282816895044)        
#         pmf = pdf.MakePmf(low=0, high=6)
#         assert_almost_equal(pmf[0.0], 0.0050742294053582942)
        
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

