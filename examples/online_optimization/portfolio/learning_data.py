import pandas as pd
from simulation.simulation import MarketSimulator
from simulation.policy import Hold, Optimal
from mlopt.sampling import uniform_sphere_sample
from mlopt.settings import DIVISION_TOL
import datetime as dt
import logging
import matplotlib.pylab as plt
import numpy as np
from os import path, remove


DATA_FOLDER = './online_optimization/portfolio/data'


def get_dimensions(data_folder=DATA_FOLDER):
    
    with pd.HDFStore(path.join(data_folder, "risk_data.h5")) as risk_data:
        exposures = risk_data['exposures']

    return exposures.loc[exposures.index[0][0]].shape


def simulate_system(data_folder, t_start, t_end, T_periods, lambda_cost=None):
    """Load stocks and simulate data"""
    # Load data
    with pd.HDFStore(path.join(data_folder, "simulation_data.h5")) as sim_data:
        prices = sim_data['prices']
        volumes = sim_data['volumes']
        returns = sim_data['returns']
        returns_estimates = sim_data['returns_estimates']

    with pd.HDFStore(path.join(data_folder, "risk_data.h5")) as risk_data:
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

    # For loop propagating the market as in simulator.run_backtest
    simulator = MarketSimulator(returns=returns,
                                volumes=volumes,
                                cash_key='USDOLLAR')

    # Optimal polict
    op_policy = Optimal(returns_estimates, risk,
                        periods=T_periods, k=None,
                        lambda_cost=lambda_cost)
    op_results = simulator.backtest(h_init, t_start=t_start,
                                    t_end=t_end,
                                    policy=op_policy,
                                    log_level=logging.INFO)

    # Print results
    print(op_results['stats'])

    # Add w_init to data (normalized value of h)
    data['w_init'] = op_results['h'].div(op_results['h'].sum(axis=1), axis=0)

    return data


def extract_learning_data(data, t_start, t_end, T=1):
    """Get learning data from simulation data."""

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
        t_series['sqrt_Sigma_F'] = \
            np.sqrt(data['risk']['sigma_factors'].loc[month].values)
        t_series['sqrt_D'] = np.sqrt(data['risk']['idyos'].loc[month].values)

        df = df.append(pd.Series(t_series), ignore_index=True)

    return df


def store_data(learning_data, data_folder):
    LEARN_DATA = path.join(data_folder, 'learn_data.h5')
    if path.isfile(LEARN_DATA):
        remove(LEARN_DATA)
    with pd.HDFStore(LEARN_DATA) as learn:
        learn['data'] = learning_data


def learning_data(data_folder=DATA_FOLDER,
                  lambda_cost=None,
                  t_start=dt.date(2008, 1, 1),
                  t_end=dt.date(2013, 1, 1),
                  T_periods=1):

    print("Cost penalties")
    print(lambda_cost)

    # Simulate
    simulation_data = simulate_system(data_folder,
                                      t_start, t_end,
                                      T_periods,
                                      lambda_cost=lambda_cost)
    # Get learning data from simulation
    learning_data = extract_learning_data(simulation_data,
                                          t_start,
                                          t_end,
                                          T_periods)

    return learning_data


def sample_around_points(df, n_total, radius={}, shuffle=True):
    """
    Sample around points provided in the dataframe for a total of
    n_total points. We sample each parameter using a uniform
    distribution over a ball centered at the point in df row.
    """

    np.random.seed(0)
    n_samples_per_point = np.round(n_total / len(df), decimals=0).astype(int)

    df_samples = pd.DataFrame()

    for idx, row in df.iterrows():
        df_row = pd.DataFrame()

        # For each column sample points and create series
        for col in df.columns:

            norm_val = np.linalg.norm(row[col])

            if norm_val < DIVISION_TOL:
                norm_val = 1.

            if col in radius:
                rad = radius[col] * norm_val
            else:
                rad = 1e-04 * norm_val

            # Sample from uniform shpere (vectorize first)
            samples = uniform_sphere_sample(row[col].flatten(), rad,
                                            n=n_samples_per_point)

            if len(samples[0]) == 1:
                # Flatten list
                samples = [item for sublist in samples for item in sublist]
            elif row[col].ndim > 1:    # Matrix
                # Reshape vectorized form to matrix
                samples = [np.reshape(s, row[col].shape) for s in samples]

            # Round stuff
            if col in ['sqrt_Sigma_F', 'sqrt_D']:
                samples = [np.maximum(s, 1e-05) for s in samples]

            #  if col in ['s_init', 'past_d']:
            #      samples = np.maximum(np.around(samples, decimals=0),
            #                           0).astype(int)
            #  elif col == 'z_init':
            #      samples = np.minimum(np.maximum(
            #          np.around(samples, decimals=0), 0), 1).astype(int)
            #
            #  elif col in ['P_load']:
            #      samples = np.maximum(samples, 0)

            #  elif col in ['E_init']:
            #      samples = np.minimum(np.maximum(samples, 5.3), 10.1)

            df_row[col] = list(samples)

        df_samples = df_samples.append(df_row)

        # Shuffle data
        df_samples = df_samples.sample(frac=1).reset_index(drop=True)

    return df_samples







