# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Analysis of Howell's data with pymc3

# +
import numpy as np              
import matplotlib.pyplot as plt # type: ignore

import pandas as pd             # type: ignore
# -

# Partial census data for !Kung San people (Africa), collected by Nancy Howell (~ 1960), csv from R. McElreath, "Statistical Rethinking", 2020.

# +
howell: pd.DataFrame

try:
    howell = pd.read_csv('https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv', sep=';', dtype={'male': bool})
except:
    howell = pd.read_csv('Howell1.csv', sep=';', dtype={'male': bool})
# -

# ## A normal model for the height
#
# We want to analyse the hypothesis that the height of adult people is normally distributed, therefore we design this statistical model ($h$ is the height), with an *a priori* normal distribution of the mean, and an *a priori* uniform distribution of the standard deviation.
#
#
# $ h \sim N(\mu, \sigma)$
#
# $ \mu \sim N(170, 20) $
#
# $ \sigma \sim U(0, 50) $
#
#

import pymc3 as pm   # type: ignore

# +
norm_height = pm.Model()

with norm_height:
    mu = pm.Normal('mu_h', 170, 20)
    sigma = pm.Uniform('sigma_h', 0, 50)
    h = pm.Normal('height', mu, sigma)
# -

# ### Exercise 1
#
# Plot the *a priori* densities of the three random variables of the model. You can sample random values with the method `random` of each. For example `mu.random(size=1000)` samples 1000 values from the *a priori* distribution of `sigma`. 
#

pass

# ### Exercise 2
#
# Consider only adult ($\geq 18$) males. Redefine the model above, making the height `h` an **observed** variable, using Howell's data about adult males as observations.

pass

# ### Exercise 3
#
# Sample values from the posterior, by using `pm.sample()`. Remember to execute this within the context of the model, by using a `with` statement. 

pass

# ### Exercise 4
#
# Plot together the density of the posterior `mu_h` and the density of the prior `mu_h`.
#

pass

# ### Exercise 5
#
# Plot the posterior densities by using `az.plot_posterior`.

# +
import arviz as az # type: ignore


pass
# -

# ### Exercise 6
#
# Since `h` is now an observed variable, it is not possible to sample prior values directly from it. You can instead use `pm.sample_prior_predictive`. Compute the sample prior predictive mean of the height.

pass


# ### Exercise 7
#
# Plot together all the posterior height densities, by using all the sampled values for `mu` and `sigma` (Use the `gaussian` function below. You will get many lines! Use a gray color and a linewidth of 0.1). Add to the plot (in red) the posterior height density computed by using the mean for the posterior `mu` and `sigma`. Add to the plot (in dashed blue) the prior height density computed by using the mean for the prior `mu` and `sigma` (used the values computed by solving the previous exercise).     
#

def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (1/(2*np.pi*sigma**2)**.5)*np.exp(-(x - mu)**2/(2*sigma**2))


pass


# ## A linear regression model
#
# We want to analyze the relationship between height and weight in adult males. We consider the following model, where $h$ is the height, $w$ is the weight, $\bar w$ is the mean weight.
#
# $ h \sim N(\mu, \sigma)$
#
# $ \mu = \alpha + \beta*(w - \bar w) $
#
# $ \alpha = N(178, 20) $
#
# $ \beta = N(0, 10) $
#
# $ \sigma \sim U(0, 50) $
#

# ### Exercise 8
#
# Define the model `linear_regression` as a `pm.Model()`. Use Howell's data as observations.

pass

# ### Exercise 9
#
# Sample the model and plot the posterior densities.
#

pass

# ### Exercise 10
#
# Plot a scatter plot of heights and the deviations of the weights from the mean. Add to the plot the regression line  using as the parameters the mean of the sampled posterior values.
#

pass
