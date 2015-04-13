(Think) Stats Toolbox  
=====================

This package contains classes and utilites for working with statistical objects such as histograms, PMFs/PDFs and CDFs in Python.

The code contained is based on the ["thinkstats2.py" module](https://github.com/AllenDowney/ThinkStats2), written as part of the ["Think Stats" book by Allen B. Downey](http://www.greenteapress.com/thinkstats2/index.html), available from greenteapress.com

Overview
--------

I am currently going through the Think* series of books starting with ThinkStats. As a way to help learn the material and become familiar with the underlying code used, I am makeing some alterations to it along the way and packaging it up into a nice helpful library for future use.

I use numpy pandas and seaborn to do the majority of my data analysis tasks, and the code herein is designed to draw upon, and be compatible with, these existing tools. e.g. returning figure/axes objects so that plots can be further customised if required, but still use helpful defaults to minimise boier plate code and so speed up interactive analysis. 

Other modifications I have made to the original code base include:

1. Writing plotting methods for the Hist, Cdf and Pmf classes which use seaborn/mpl. 
2. Incorperate to/from conversion methods into the classes to mimic pandas api.
3. Reorganise code into a python package structure.
4. Refactoring function names to snake case instead of upper cammel case (personal preference).
6. Adding a small number of custom ploting utilites to handle ploting multiple objects.  

Classes Ported so Far
----------------------
`Hist`: represents a histogram (map from values to integer frequencies).

`Pmf`: represents a probability mass function (map from values to probs).

`Cdf`: represents a discrete cumulative distribution function.

Functions Ported so Far
-----------------------
`cohen_effect_size`: Return Cohen's d statisitc for effect size between two samples.

`normal_probability_plot`: Create a normal probability plot with a fitted line

`multiplot`: Plot multiple stats objects on a single axes.

Licence
-------
GNU GPLv3 http://www.gnu.org/licenses/gpl.html
