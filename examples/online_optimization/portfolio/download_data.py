import pandas as pd
import numpy as np
import datetime as dt
from os import path, remove
import pickle
import quandl
np.random.seed(0)

# Asset data download as in
# https://github.com/cvxgrp/cvxportfolio/blob/master/examples/DataEstimatesRiskModel.ipynb
DATA_DIR = './online_optimization/portfolio/data/'

# Assert names and data
SP_names = path.join(DATA_DIR, 'SP100.csv')
SP_data = path.join(DATA_DIR, 'SP100.pickle')

# Asset names
assets = pd.read_csv(SP_names, comment='#').set_index('Symbol')

# Last 10 years
start_date = dt.date(2005, 1, 1)
end_date = dt.date(2017, 12, 31)   # WIKIP unmaintained after end of march 2018

QUANDL = {
    'authtoken': "Whjs1wx72N2A7BxEbRDV",  # Quandl API key
    'start_date': start_date,
    'end_date': end_date}

RISK_FREE_SYMBOL = "USDOLLAR"
asset_names = assets.index.tolist()
data = {}

'''
Download data from QUANDL
'''
try:
    # Data already exists
    with open(SP_data, 'rb') as handle:
        data = pickle.load(handle)
except FileNotFoundError:
    for ticker in asset_names:
        # Download assets for ticker in assets.index:
        if ticker not in data:
            print('downloading %5s from %10s to %10s' % (ticker,
                                                         QUANDL['start_date'],
                                                         QUANDL['end_date']))
            quandl_name = "WIKI/" + ticker.replace(".", "_")
            data[ticker] = quandl.get(quandl_name, **QUANDL)

    # Download USDOLLAR value
    print('downloading %s from %s to %s' % (RISK_FREE_SYMBOL,
                                            QUANDL['start_date'],
                                            QUANDL['end_date']))
    data[RISK_FREE_SYMBOL] = quandl.get("FRED/DTB3", **QUANDL)

    with open(SP_data, 'wb') as handle:
        pickle.dump(data, handle)

print("Cleaning up data...", end="")
# Compute prices from adjusted closing value
prices = pd.DataFrame({x: data[x]['Adj. Close'] for x in asset_names})
n_days = len(prices)

# Compute log-volatility (sigmas) as in Black-Scholes model
open_prices = pd.DataFrame({x: data[x]['Open'] for x in asset_names})
close_prices = pd.DataFrame({x: data[x]['Close'] for x in asset_names})
sigmas = np.abs(np.log(open_prices.astype(float)) -
                np.log(close_prices.astype(float)))

# Extract volumes
volumes = pd.DataFrame({x: data[x]['Adj. Volume'] for x in asset_names})

# Ignore bad assets with more than 2% missing values
bad_assets = prices.columns[prices.isnull().sum() > n_days * 0.02]
prices = prices.loc[:, ~prices.columns.isin(bad_assets)]
sigmas = sigmas.loc[:, ~sigmas.columns.isin(bad_assets)]
volumes = volumes.loc[:, ~volumes.columns.isin(bad_assets)]
n_assets = len(prices.columns)


# Ignore days with more than 90% missing values
# (prices, sigmas or volumes) across the assets
bad_days_sigmas = sigmas.index[sigmas.isnull().sum(1) > n_assets * .9]
bad_days_prices = prices.index[prices.isnull().sum(1) > n_assets * .9]
bad_days_volumes = volumes.index[volumes.isnull().sum(1) > n_assets * .9]
bad_days = pd.Index(
    set(bad_days_sigmas).union(set(bad_days_prices), set(bad_days_volumes))
).sort_values()
prices = prices.loc[~prices.index.isin(bad_days)]
sigmas = sigmas.loc[~sigmas.index.isin(bad_days)]
volumes = volumes.loc[~volumes.index.isin(bad_days)]
n_days = len(prices)

# Fill NaN values
prices = prices.fillna(method='ffill')
sigmas = sigmas.fillna(method='ffill')
volumes = volumes.fillna(method='ffill')

# Get final values
volumes = volumes * prices  # Volumes in dollars
returns = prices.pct_change().fillna(method='ffill').iloc[1:]  # Shift by 1

# Remove assets with dubious returns
bad_assets = returns.columns[((returns < - .5).sum() > 0) |
                             ((returns > 2.).sum() > 0)]
prices = prices.loc[:, ~prices.columns.isin(bad_assets)]
sigmas = sigmas.loc[:, ~sigmas.columns.isin(bad_assets)]
volumes = volumes.loc[:, ~volumes.columns.isin(bad_assets)]
returns = returns.loc[:, ~returns.columns.isin(bad_assets)]

# Add USDOLLAR returns
# Normalize by 250 (trading days) and 100 (they are in percentage)
returns_usdollar = data[RISK_FREE_SYMBOL]
returns_usdollar = returns_usdollar.loc[~returns_usdollar.index.isin(bad_days)]
returns[RISK_FREE_SYMBOL] = returns_usdollar['Value']/(250*100)
returns = returns.fillna(method='ffill')
n_assets = len(prices.columns)

# Compute estimates
# https://github.com/cvxgrp/cvxportfolio/blob/master/examples/DataEstimatesRiskModel.ipynb

print("[OK]")

'''
Generate return estimates
'''
std_r = np.round(returns.std().mean(), decimals=2)  # 0.02
std_n = 0.14142  # std_n^2 ~ 0.02! Also, std_n ~ 10 * std_r
noise = pd.DataFrame(index=returns.index,
                     columns=returns.columns,
                     data=std_n * np.random.randn(*returns.values.shape))
alpha = std_r**2 / (std_r**2 + std_n**2)
returns_estimates = alpha * (returns + noise)
# No uncertainty in cash account
returns_estimates[RISK_FREE_SYMBOL] = returns[RISK_FREE_SYMBOL]

'''
Get factor risk model

:math: \\hat{r}
:math:\\Sigma = F \\Sigma^F \\Sigma^T + D

'''
print("Computing factor risk model...", end='')

# Average returns for each day
# NB. Not used because estimated online
#  r_hat = {}
#  for day in returns.index:
#
#      sampled_returns = returns.loc[
#          (returns.index >= day - pd.Timedelta("10 days")) &
#          (returns.index < day)
#      ]
#
#      if not sampled_returns.empty:
#          r_hat[day] = sampled_returns.mean()

#  r_hat_df = returns.rolling(window=250).mean().dropna()  # Mean
# r_hat.xs('2018-03-27')  # Example access values

# N.B. We use a factor risk model for Sigma (not this full model)
#  Sigma_hat = returns.rolling(window=250,  # Covar
#                              closed='neither').cov().dropna()

# Get first days of the month
# https://github.com/cvxgrp/cvxportfolio/blob/05c8df138a493967668b7c966fb49e722db7c6f2/examples/DataEstimatesRiskModel.ipynb
time_delta = pd.Timedelta("730 days")
start_date_estimation = start_date + time_delta
first_day_month = pd.date_range(start=start_date_estimation,
                                end=end_date, freq='MS')

k = 15   # Use only 15 factors
exposures, sigma_factors, idyos = {}, {}, {}

for day in first_day_month:

    sampled_returns = returns.loc[
        (returns.index >= day - time_delta) &  # 2 years before
        (returns.index < day)
    ]

    if not sampled_returns.empty:

        second_moment =  \
            sampled_returns.values.T @ sampled_returns.values / \
            len(sampled_returns)

        e_val, e_vec = np.linalg.eigh(second_moment)

        # Largest k factors
        sigma_factors[day] = pd.Series(
                data=e_val[-k:],  # \\Sigma^F as a diagonal matrix
                index=pd.RangeIndex(k)
        )

        exposures[day] = pd.DataFrame(data=e_vec[:, -k:],
                                      index=returns.columns)

        # All other factors (approximate with the diagonal part of the matrix)
        idyos[day] = pd.Series(
            data=np.diag(e_vec[:, :-k]@np.diag(e_val[:-k])@e_vec[:, :-k].T),
            index=returns.columns
            )

print("[OK]")

# Create collective data structures
exposures_df = pd.concat(exposures.values(), keys=first_day_month)
idyos_df = pd.concat(idyos.values(), keys=first_day_month)
sigma_factors_df = pd.concat(sigma_factors.values(), keys=first_day_month)

# Store simulation and risk data
SIMULATION_DATA = path.join(DATA_DIR, 'simulation_data.h5')
if path.isfile(SIMULATION_DATA):
    remove(SIMULATION_DATA)

with pd.HDFStore(SIMULATION_DATA) as simulation:
    simulation['prices'] = prices
    simulation['volumes'] = volumes
    simulation['returns'] = returns
    simulation['returns_estimates'] = returns_estimates
    simulation['sigmas'] = sigmas

RISK_DATA = path.join(DATA_DIR, 'risk_data.h5')
if path.isfile(RISK_DATA):
    remove(RISK_DATA)
with pd.HDFStore(RISK_DATA) as risk:
    risk['exposures'] = exposures_df
    risk['idyos'] = idyos_df
    risk['sigma_factors'] = sigma_factors_df
