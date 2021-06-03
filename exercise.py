# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Analysis of Howell's data with pymc3

import numpy as np              # type: ignore
import matplotlib.pyplot as plt # type: ignore
# %matplotlib inline

import pandas as pd             # type: ignore

# Partial census data for !Kung San people (Africa), collected by Nancy Howell (~ 1960), csv from R. McElreath, "Statistical Rethinking", 2020.

# +
howell: pd.DataFrame

try:
    howell = pd.read_csv('https://raw.githubusercontent.com/rmcelreath/rethinking/master/data/Howell1.csv', sep=';', dtype={'male': bool})
except:
    howell = pd.read_csv('Howell1.csv', sep=';', dtype={'male': bool})
