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

fig, ax = plt.subplots(ncols=3, figsize=(15,5))
ax[0].hist(sigma.random(size=10000), bins='auto', density=True)
ax[0].set_title(str(sigma))
ax[1].hist(mu.random(size=10000), bins='auto', density=True)
ax[1].set_title(str(mu))
ax[2].hist(h.random(size=10000), bins='auto', density=True)
_ = ax[2].set_title(str(h))

# ### Exercise 2
#
# Consider only adult ($\geq 18$) males. Redefine the model above, making the height `h` an **observed** variable, using Howell's data about adult males as observations.

adult_males = howell.query('male & age >= 18')

# +
norm_height_am = pm.Model()

with norm_height_am:
    mu = pm.Normal('mu_h', 170, 20)
    sigma = pm.Uniform('sigma_h', 0, 50)
    h = pm.Normal('height', mu, sigma, observed=adult_males['height'])
# -

# ### Exercise 3
#
# Sample values from the posterior, by using `pm.sample()`. Remember to execute this within the context of the model, by using a `with` statement. 

with norm_height_am:
    posterior = pm.sample(return_inferencedata=False)

# ### Exercise 4
#
# Plot together the density of the posterior `mu_h` and the density of the prior `mu_h`.
#

fig, ax = plt.subplots()
ax.hist(posterior['mu_h'], bins='auto', density=True, label='Posterior mu_h')
ax.hist(mu.random(size=1000), bins='auto', density=True, label='Prior mu_h', 
        range=(posterior['mu_h'].min(),posterior['mu_h'].max()))
_ = fig.legend()

# ### Exercise 5
#
# Plot the posterior densities by using `az.plot_posterior`.

# +
import arviz as az # type: ignore


with norm_height_am:
    pm.plot_posterior(posterior)
# -

# ### Exercise 6
#
# Since `h` is now an observed variable, it is not possible to sample prior values directly from it. You can instead use `pm.sample_prior_predictive`. Compute the sample prior predictive mean of the height.

# +
with norm_height_am:
    prior = pm.sample_prior_predictive(1000, var_names=['height'])

prior['height'].mean()

# +
fig, ax = plt.subplots()
ax.hist(prior['height'][:,0], bins='auto', density=True, label='Prior height (sample 0)')
ax.hist(prior['height'][:,1], bins='auto', density=True, label='Prior height (sample 1)')
ax.hist(prior['height'].flatten(), bins='auto', density=True, label='Prior height (all samples)')

_ = fig.legend()


# -

# ### Exercise 7
#
# Plot together all the posterior height densities, by using all the sampled values for `mu` and `sigma` (Use the `gaussian` function below. You will get many lines! Use a gray color and a linewidth of 0.1). Add to the plot (in red) the posterior height density computed by using the mean for the posterior `mu` and `sigma`. Add to the plot (in dashed blue) the prior height density computed by using the mean for the prior `mu` and `sigma` (used the values computed by solving the previous exercise).     
#

def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (1/(2*np.pi*sigma**2)**.5)*np.exp(-(x - mu)**2/(2*sigma**2))


fig, ax = plt.subplots()
x = np.linspace(100, 200, 1000)
for i in range(len(posterior)):
    ax.plot(x, gaussian(x, posterior['mu_h'][i], posterior['sigma_h'][i]), color='gray', linewidth=.1)
ax.plot(x, gaussian(x, posterior['mu_h'].mean(), posterior['sigma_h'].mean()), color='red')
ax.plot(x, gaussian(x, mu.random(size=10000).mean(), 
                       sigma.random(size=10000).mean().mean()), color='blue', linestyle='dashed')
_ = ax.set_title('Posterior height')


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

# +
linear_regression = pm.Model()

with linear_regression:
    sigma = pm.Uniform('sigma_h', 0, 50)
    alpha = pm.Normal('alpha', 178, 20)
    beta = pm.Normal('beta', 0, 10)
    mu = alpha + beta*(adult_males['weight'] - adult_males['weight'].mean())
    h = pm.Normal('height', mu, sigma, observed=adult_males['height'])
# -

# ### Exercise 9
#
# Sample the model and plot the posterior densities.
#

with linear_regression:
    post = pm.sample(return_inferencedata=False)

with linear_regression:
    pm.plot_posterior(post)

# ### Exercise 10
#
# Plot a scatter plot of heights and the deviations of the weights from the mean. Add to the plot the regression line  using as the parameters the mean of the sampled posterior values.
#

# +
d_weight = adult_males['weight'] - adult_males['weight'].mean()

x = np.linspace(d_weight.min(), d_weight.max(), 100)


fig, ax = plt.subplots()
ax.scatter(d_weight, adult_males['height'])
_ = ax.plot(x, post['alpha'].mean() + post['beta'].mean()*x, color='red')
