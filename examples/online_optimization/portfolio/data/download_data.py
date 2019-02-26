import pandas as pd
import numpy as np
import datetime as dt
import pickle
import quandl

# Asset data download as in
# https://github.com/cvxgrp/cvxportfolio/blob/master/examples/DataEstimatesRiskModel.ipynb

# Assert names
SP500_names = 'SP500.csv'
assets = pd.read_csv('SP500.csv', comment='#').set_index('Symbol')

# Last 10 years
start_date = dt.date(2008, 1, 1)
end_date = dt.date(2018, 12, 31)

QUANDL = {
    'authtoken': "Whjs1wx72N2A7BxEbRDV",  # Quandl API key
    'start_date': start_date,
    'end_date': end_date}

RISK_FREE_SYMBOL = "USDOLLAR"
asset_names = assets.index.tolist()  # + [RISK_FREE_SYMBOL]
data = {}


SP500_data = 'SP500.pickle'
try:
    # Data already exists
    with open(SP500_data, 'rb') as handle:
        data = pickle.load(handle)
except FileNotFoundError:
    # Download assets
    for ticker in assets.index:
        if ticker in data:
            continue
        print('downloading %s from %s to %s' % (ticker,
                                                QUANDL['start_date'],
                                                QUANDL['end_date']))
        quandl_name = "WIKI/" + ticker.replace(".", "_")
        data[ticker] = quandl.get(quandl_name, **QUANDL)

    # Download USDOLLAR value
    print('downloading %s from %s to %s' % (RISK_FREE_SYMBOL,
                                            QUANDL['start_date'],
                                            QUANDL['end_date']))
    data[RISK_FREE_SYMBOL] = quandl.get("FRED/DTB3", **QUANDL)

    with open(SP500_data, 'wb') as handle:
        pickle.dump(data, handle)


# Compute prices from adjusted closing value
prices = pd.DataFrame({x: data[x]['Adj. Close'] for x in asset_names})
#  prices[RISK_FREE_SYMBOL] = data[RISK_FREE_SYMBOL]['Value']
n_days = len(prices)

# Compute log-volatility (sigmas) as in Black-Scholes model
open_prices = pd.DataFrame({x: data[x]['Open'] for x in asset_names})
close_prices = pd.DataFrame({x: data[x]['Close'] for x in asset_names})
sigmas = np.abs(np.log(open_prices.astype(float)) -
                np.log(close_prices.astype(float)))
#  sigmas[RISK_FREE_SYMBOL] = pd.Series(np.nan * np.ones(n_days))

# Extract volumes
volumes = pd.DataFrame({x: data[x]['Adj. Volume'] for x in asset_names})
#  volumes[RISK_FREE_SYMBOL] = pd.Series(np.nan * np.ones(n_days))

# Fix risk-free (???) => WHY??
#  prices[RISK_FREE_SYMBOL] = \
#      10000*(1 + prices[RISK_FREE_SYMBOL]/(100*250)).cumprod()
# 10.000 is the amount of money invested first!




# Ignore bad assets with more than 2% missing values
bad_assets = prices.columns[prices.isnull().sum() > n_days * 0.02]
if bad_assets.any():
    print('Assets %s have too many NaNs, removing them' % bad_assets.values)
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

#  print("Removing these days from dataset because they have too many NaNs:")
#  print(pd.DataFrame({'NaN prices': prices.isnull().sum(1)[bad_days],
#                      'NaN volumes': volumes.isnull().sum(1)[bad_days],
#                      'NaN sigmas': sigmas.isnull().sum(1)[bad_days]}))

prices = prices.loc[~prices.index.isin(bad_days)]
sigmas = sigmas.loc[~sigmas.index.isin(bad_days)]
volumes = volumes.loc[~volumes.index.isin(bad_days)]
n_days = len(prices)
#  print(pd.DataFrame({'Remaining NaN price': prices.isnull().sum(),
                    #  'Remaining NaN volumes': volumes.isnull().sum(),
                    #  'Remaining NaN sigmas': sigmas.isnull().sum()}))

# Fill values
#  print("Using forward fill for remaining part of the dataset")
prices = prices.fillna(method='ffill')
sigmas = sigmas.fillna(method='ffill')
volumes = volumes.fillna(method='ffill')

#  print(pd.DataFrame({'Remaining NaN price': prices.isnull().sum(),
#                      'Remaining NaN volumes': volumes.isnull().sum(),
#                      'Remaining NaN sigmas': sigmas.isnull().sum()}))


# Get final values
volumes = volumes * prices  # Volumes in dollars
#  returns = (prices.diff()/prices.shift(1)).fillna(method='ffill').iloc[1:]
returns = prices.pct_change().fillna(method='ffill').iloc[1:]

# Remove assets with dubious returns
bad_assets = returns.columns[((- .5 > returns).sum() > 0) |
                             ((returns > 2.).sum() > 0)]
if len(bad_assets):
    print('Assets %s have dubious returns, removed' % bad_assets.values)
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
r_hat = returns.rolling(window=250).mean().dropna()  # Mean
Sigma_hat = returns.rolling(window=250, closed='neither').cov().dropna()  # Covariance

# Access to values (example)
# r_hat.xs('2018-03-27')
# Sigma_hat.xs('2018-03-27')

# Get factor risk model
# \Sigma = F \Sigma^F \Sgima^T + D

# Get first days of the month
# https://github.com/cvxgrp/cvxportfolio/blob/05c8df138a493967668b7c966fb49e722db7c6f2/examples/DataEstimatesRiskModel.ipynb
#  first_day_month = \
#      pd.date_range(
#          start=returns.index[
#              next(i for (i, el) in
#                   #  enumerate(returns.index >= pd.Timestamp(start_date))
#                   enumerate(returns.index >= '2012-01-01')
#                   if el
#                   ) - 1],
#          end=returns.index[-1], freq='MS')
#
first_day_month = pd.date_range(start=start_date, end=end_date, freq='MS')

k = 15   # Use only 15 factors

exposures, sigma_factors, idyos = {}, {}, {}

for day in first_day_month:

    sampled_returns = returns.loc[
        (returns.index >= day - pd.Timedelta("700 days")) &
        (returns.index < day)
    ]

    second_moment =  \
        sampled_returns.values.T @ sampled_returns.values/len(sampled_returns)

    e_val, e_vec = np.linalg.eigh(second_moment)

    # Largest k factors
    sigma_factors[day] = np.diag(e_val[-k:])  # \Sigma^F
    exposures[day] = pd.DataFrame(data=e_vec[:, -k:],
                                  index=returns.columns)

    # All other factors (approximate with the diagonal part of the matrix)
    idyos[day] = pd.Series(
        data=np.diag(e_vec[:, :-k] @ np.diag(e_val[:-k]) @ e_vec[:, :-k].T),
        index=returns.columns
        )

exposures_df = pd.DataFrame(exposures)




# TODO: Store values
# TODO: Cleanup

