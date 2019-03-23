import pandas as pd
from simulation.simulation import MarketSimulator
from simulation.policy import Hold, Optimal
import datetime as dt
import logging
import matplotlib.pylab as plt
import numpy as np
from os import path, remove

# Load data
with pd.HDFStore("./data/simulation_data.h5") as sim_data:
    prices = sim_data['prices']
    volumes = sim_data['volumes']
    returns = sim_data['returns']
    returns_estimates = sim_data['returns_estimates']

with pd.HDFStore("./data/risk_data.h5") as risk_data:
    exposures = risk_data['exposures']
    sigma_factors = risk_data['sigma_factors']
    idyos = risk_data['idyos']
    risk = {'exposures': exposures,
            'sigma_factors': sigma_factors,
            'idyos': idyos}

data = {'prices': prices,
        'returns': returns,
        'risk': risk}


# Initial portfolio allocation (equally divided over the assets
w_init = pd.Series(index=returns.columns, data=1)
w_init.USDOLLAR = 0.
w_init /= sum(w_init)

# Invest funds
funds = 1e4
h_init = w_init * funds

# Times
t_start = dt.date(2008, 1, 1)
t_end = dt.date(2013, 1, 1)
T_periods = 2

# For loop propagating the market as in simulator.run_backtest
simulator = MarketSimulator(returns=returns,
                            volumes=volumes,
                            cash_key='USDOLLAR')

# Optimal polict
op_policy = Optimal(returns_estimates, risk, periods=T_periods)
op_results = simulator.backtest(h_init, t_start=t_start,
                                t_end=t_end,
                                policy=op_policy, log_level=logging.WARNING)

print(op_results['stats'])


# Add w_init to data
data['w_init'] = op_results['h'].div(op_results['h'].sum(axis=1), axis=0)


def get_learning_data(data, t_start, t_end, T=1):

    returns = data['returns']
    times = returns.index
    n_times = len(times[(times >= str(t_start)) &
                        (times <= str(t_end))])
    sample_times = times[times >= str(t_start)]
    df = pd.DataFrame()

    for t in range(n_times):
        t_series = {}

        # Initial value
        t_series['w_init'] = data['w_init'].loc[sample_times[t]].values

        # hat_r
        for period in range(1, T+1):
            t_series['hat_r_%s' % str(period)] = \
                returns.loc[sample_times[t + period]].values

        # Risk estimate
        month = dt.date(sample_times[t].year, sample_times[t].month, 1)
        t_series['F'] = data['risk']['exposures'].loc[month].values
        t_series['Sigma_F'] = \
            np.diag(data['risk']['sigma_factors'].loc[month].values)
        t_series['sqrt_D'] = np.sqrt(data['risk']['idyos'].loc[month].values)

        df = df.append(pd.Series(t_series), ignore_index=True)

    return df


learning_data = get_learning_data(data, t_start, t_end, T_periods)

DATA_DIR = "./data/"
LEARN_DATA = path.join(DATA_DIR, 'learn_data.h5')
if path.isfile(LEARN_DATA):
    remove(LEARN_DATA)
with pd.HDFStore(LEARN_DATA) as learn:
    learn['data'] = learning_data
