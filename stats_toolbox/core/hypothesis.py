""" From Think Stats Chapter 9  

HypothesisTest is an abstract parent class. Child classes based on HypothesisTest 
inherit __init__ and pvalue and provide test_statistic, run_model, and optionally make_model.

__init__ takes the data in whatever form is appropriate. It calls make_model, which builds a 
representation of the null hypothesis, then passes the data to test_statistic, which computes 
the size of the effect in the sample data given.

pvalue computes the probability of the apparent effect under the null hypothesis. It takes as a 
parameter iters, which is the number of simulations to run. The first line generates simulated 
data, computes test statistics, and stores them in test_stats. The result is the fraction of 
elements in test_stats that exceed or equal the observed test statistic, self.actual.

"""

import matplotlib.pyplot as plt

from .cdf import Cdf
from .shared import UnimplementedMethodException


class HypothesisTest(object):

    """Represents a hypothesis test."""

    def __init__(self, data):
        """Initializes.

        data: data in whatever form is relevant
        """
        self.data = data
        self.make_model()
        self.actual = self.test_statistic(data)
        self.test_stats = None
        self.test_cdf = None

    def pvalue(self, iters=1000):
        """Computes the distribution of the test statistic and p-value.

        iters: number of iterations

        returns: float p-value
        """
        self.test_stats = [self.test_statistic(self.run_model()) 
                           for _ in range(iters)]
        self.test_cdf = Cdf(self.test_stats)

        count = sum(1 for x in self.test_stats if x >= self.actual)
        return count / iters

    def max_test_stat(self):
        """Returns the largest test statistic seen during simulations."""
        return max(self.test_stats)

    def plot_cdf(self, label='_no_legend_'):
        """Draws a Cdf with vertical lines at the observed test stat."""
        def vert_line(x):
            """Draws a vertical line at x."""
            plt.plot([x, x], [0, 1], color='0.8')

        vert_line(self.actual)
        self.test_cdf.plot(label=label)

    def test_statistic(self, data):
        """Computes the test statistic.

        data: data in whatever form is relevant        
        """
        raise UnimplementedMethodException()

    def make_model(self):
        """Build a model of the null hypothesis. """
        pass

    def run_model(self):
        """Run the model of the null hypothesis.

        returns: simulated data
        """
        raise UnimplementedMethodException()
