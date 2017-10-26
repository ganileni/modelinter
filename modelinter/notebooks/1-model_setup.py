
# coding: utf-8

# # Model setup
# 
# here we set the model up and explain how it works

# imports and load data

# In[1]:


# all of this is explained in notebook 0
get_ipython().run_line_magic('run', 'imports.py')
from modelinter.preprocessing.imports_load_data import read_csvs, extract_arrays
raw_data = read_csvs()
arrays = extract_arrays(raw_data)


# We'll simulate a portfolio of N stocks and N options. We'll model the price of stocks, and feed the output of this model to the model for options.
# 
# ## Stocks model
# 
# The model for **stocks** is a linear regression of their returns against S&P500:
# 
# $r_{it} = \alpha_i + \beta_i I_t + \epsilon_{it} $.
# 
# with:
# 
# * $r_{it}$ - returns of stock $i$ at time $t$
# * $I_{t}$ - returns of the index $I$ (S&P500) at time $t$
# * $\alpha_i$ - zero order coefficient for stock $i$
# * $\beta_i$ - first-order coefficient for stock $i$
# * $\epsilon_{it}$ - residuals of the regression for stock $i$ at time $t$
# 
# We expect $\alpha_{i}$ to be negligible for stocks, therefore we should be able to approximate:
# 
# $r_{it} \approx \beta_i I_t $
# 
# We'll call this the **A** model. A more accurate prediction would entail sampling from the distribution of these random variables:
# 
# $r_{it} \approx \beta_i I_t + \epsilon_{it} $
# 
# where we assume that the $\epsilon$ terms are normally distributed, centered on zero $E[\epsilon_{it}] = 0$, and uncorrelated ($Cov[\epsilon_{it}, \epsilon_{jt}] = 0$ $\forall i \neq j $). We will call this the **B** model, and we'll evaluate it by estimating the standard deviation of the residuals of the regression, and sampling from a normal distribution with adequate parametrisation.
# 
# Finally, an even better model, which we'll call the **C** model, would involve removing the assumption of uncorrelated residuals: $\Sigma_{ij} \equiv Cov[\epsilon_{it}, \epsilon_{jt}] \neq 0$. We will estimate this model by evaluating the covariance matrix $\Sigma_{ij}$ with the GLASSO algorithm.
# 
# 
# ___
# 
# #### extrapolating forward in time
# 
# this model, being evaluated on daily returns, will predict daily returns. We want to understand what happens on timescales that are characteristic of the CCAR scenario (~4 years). In order to do so, we will have to chain daily predictions. Let's suppose that we want to predict the returns of stock at time $T = D \Delta t$. We can write the returns on index $I$ at time $T$ as the sum of the returns on all days:
# 
# $I_T = \prod_t^D (1 + I_t) \approx \sum_t^D I_t$
# 
# Mind that this formula works approximately for linear returns, while it's exact for log returns. Similarly for the stock:
# 
# $r_T = \prod_t^D (1 + r_t) \approx \sum_t^D r_t \\
# = \sum_t^D (\beta_i I_{t} + \epsilon_{it}) = \beta_i I_T + \sum_t^D \epsilon_{it}$
# 
# This is important, because the rules for calculating uncertainty follow from it:
# 
# $\sigma(r_T) = \sigma(\sum_t^D \epsilon_{it}) = \sqrt{D} \sigma(\epsilon_i)$
# 
# where we made the hypothesis that $\epsilon_{it}$  is a stationary process, i.e. $\epsilon_{it} = \epsilon_{i}$, and that it follows a normal distribution.
# 

# ## Options model
# 
# For the **options**, we'll consider one put option per stock, with moneyness ($S/K$) of 1.1, that ends in five years. We'll assume a risk free rate of return of 0.25%. The model will be the original black-scholes.
# 
# $C(S_t, t) = N(d_1)S_t - N(d_2) Ke^{-r(T - t)} \\
#      d_1 = \frac{1}{\sigma\sqrt{T - t}}\left[\ln\left(\frac{S_t}{K}\right) + \left(r + \frac{\sigma^2}{2}\right)(T - t)\right] \\
#      d_2 = d_1 - \sigma\sqrt{T - t}$
# 
# with:
# * $N(\cdot)$ is the cumulative distribution function of the standard normal distribution
# * $\tau \equiv T - t$ is the time to maturity (expressed in years), with $T$ the maturity date, and $t$ current time
# * $S$ is the spot price of the underlying asset
# * $K$ is the strike price
# * $r$ is the risk free rate (annual rate, expressed in terms of continuous compounding)
# * $\sigma$ is the volatility of returns of the underlying asset
# 
# and all values are intended to be calculated at time $t$.
# 
# ___
# 
# We will model the volatility $\sigma$ with a linear regression on the VIX index similar to the one for the stocks:
# 
# $\sigma_{it} = \eta_i + \theta_i J_t + \delta_{it} $
# 
# therefore we should have an **A** model like this:
# 
# $\sigma_{it} \approx \eta_i + \theta_i J_t $,
# 
# where $J_t$ is the VIX index. We will also define, in analogy with the stocks, an **B** model:
# 
# $\sigma_{it} \approx \eta_i + \theta_i J_t + \delta_{it}$,
# 
# where we consider the $\delta$ to be normally distributed with mean 0, and standard deviation measured from the residuals of the regression. Finally, we'll define a **C** model where $\Sigma_{ij} \equiv Cov[\delta_{it}, \delta_{jt}] \neq 0$.
# 
# ---
# 
# We won't use implied volatility $VI_t$ data directly to estimate the $\theta_i$ and parametrize the model. We will instead assume that realized future volatility $\sigma_{t+p}$ as a proxy for it, i.e. we assume:
# 
# $VI_t \propto \sigma_{t+p}$.
# 
# For this reason, we equivalently delayed  VIX by $p=-10$ days, following the result of preliminary analysis.
# 
# ___
# 
# #### extrapolating forward in time
# 
# The model for implied volatilities doesn't predict returns, but volatilities directly. The value for implied volatility at time $T$ cannot be expressed as a composition of changes along time, therefore the uncertainty for the predictions doesn't increase over time:
# 
# $std(\sigma_T) = std(\delta_i)$
# 
# and of course we made the implicit hypothesis the hypothesis that $\delta_{it}$  is a stationary process, i.e. $\delta_{it} = \delta_{i}$, and that it follows a normal distribution.

# let's define the functions that evaluate the model.

# ## Code review
# All the code is in the modelinter.pgm module. let's go through it.

# In[2]:


from modelinter.models.pgm import     BasePgmParams, StocksPgmParams,     StocksPgm, StocksPgmA,     StocksPgmB, StocksPgmC, OptionsPgm


# the parameters of the PGM will be stored in an object of class `BasePgmParams`, that only has the logic for saving/loading the parameters and estimating GLASSO on the residuals

# In[3]:


view_code(BasePgmParams)


# The realization of a PGM parameters object for the stocks model is a `StocksPgmParams` object.  On initialization it estimates the parameters for the model from the regression, and then calculates GLASSO on the residuals.

# In[4]:


view_code(StocksPgmParams)


# When predicting, we'll use a realization of the `StocksPgm` base class. This base class just contains the logic for initialization (it gets passed a `StocksPgmParams` when doing `__init__()`), and the part of the prediction logic that will be called by all subclasses.

# In[5]:


view_code(StocksPgm)


# The actual realizations of `StocksPgm` are the three kinds of model A, B and C discussed both above and in the paper: `StocksPgmA`, `StocksPgmB` and `StocksPgmC`.
# 
# `StocksPgmA` just takes the prediction method from the base class, and changes the signature a little:

# In[6]:


view_code(StocksPgmA)


# `StocksPgmB` and `StocksPgmC`'s `predict()` methods instead take the base prediction and generate the residuals.

# In[7]:


view_code(StocksPgmB); view_code(StocksPgmC)


# The options model is structured the same way, with
# 
# * `BasePgmParams`
#     * `OptionsPgmParams`
# * `OptionsPgm`
#     * `optionsPgmA`
#     * `optionsPgmB`
#     * `optionsPgmC`
#     
# the logic in `OptionsPgm` is a bit more convoluted, though, because its `predict()` method contains the code to calculate options prices through the Black-Scholes formula, which is structured as two ugly nested loops. The logic for generating volatilities is instead found in the subclasses.

# In[9]:


view_code(OptionsPgm)


# ## Test calculation
# Now let's test that they actually work:

# In[15]:


#DefaultSettings is just a namespace containing some settings for eval_PGM
from modelinter.models.calculations import eval_PGM, DefaultSettings
#we'll also need a couple of constants
from modelinter.models.constants import Const
import numpy as np


# In[11]:


#val_pgm is instead a function that estimates all the model
#parameters from the data:
view_code(eval_PGM)


# In[12]:


#let's evaluate them
models = eval_PGM(arrays, DefaultSettings)


# In[13]:


#does it work?
#prediction for stocks
(models.PGM_stock_A.predict(.01)[:20],
 models.PGM_stock_B.predict(.01)[:20],
 models.PGM_stock_C.predict(.01)[:20])


# In[19]:


#let's set an arbitrary level for vix
vix_set = np.mean(arrays.vix)
#prediction for options
(
models.PGM_options_A.predict(
                vix=vix_set,
                r=Const.RISK_FREE_RATE.value,
                strike_price = None,
                stocks_prices = arrays.stocks_p[-1],
                tau_i = [1]*arrays.stocks_p.shape[-1],
                flag_i = ['p']*arrays.stocks_p.shape[-1],
                moneyness_i=1.1,
                )[:20],
models.PGM_options_B.predict(
                vix=vix_set,
                r=Const.RISK_FREE_RATE.value,
                strike_price = None,
                stocks_prices = arrays.stocks_p[-1],
                tau_i = [1]*arrays.stocks_p.shape[-1],
                flag_i = ['p']*arrays.stocks_p.shape[-1],
                moneyness_i=1.1,
                )[:20],
models.PGM_options_C.predict(
                vix=vix_set,
                r=Const.RISK_FREE_RATE.value,
                strike_price = None,
                stocks_prices = arrays.stocks_p[-1],
                tau_i = [1]*arrays.stocks_p.shape[-1],
                flag_i = ['p']*arrays.stocks_p.shape[-1],
                moneyness_i=1.1,
                )[:20]
)

