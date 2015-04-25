"""This package contains classes and utilites for empirical PMFs/PDFs and CDFs.

The code contained is a slightly modified version of the "thinkstats2.py" 
module written as part of the "Think Stats" book by Allen B. Downey, 
available from greenteapress.com

The modifications I made are the following:
1. Incorperate plotting methods for the classes which use seaborn/pandas. 
2. Refactoring to snake case for function name consistency.
4. Incorperate to/from conversion functins into the classes.
5. Reorganise code into a python package. 

Classes
-------

Hist: represents a histogram (map from values to integer frequencies).

Pmf: represents a probability mass function (map from values to probs).

_DictWrapper: private parent class for Hist and Pmf.

Cdf: represents a discrete cumulative distribution function

Pdf: represents a continuous probability density function

License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

# Import the main Classes
from .core.hist import Hist
from .core.pmf import Pmf
from .core.pdf import Pdf
from .core.cdf import Cdf

# Import the most usefult functions and utilsinto the main namespace
from .core.tools import cohen_effect_size
from .core.plot import mulitplot, normal_probability_plot