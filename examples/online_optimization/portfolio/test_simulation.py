import pandas as pd
from simulation.simulation import MarketSimulator
from simulation.policy import Hold, SinglePeriod
import datetime as dt
import logging
import matplotlib.pylab as plt
# Write basic simulation code with basic strategies


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


# Initial portfolio allocation (equally divided over the assets
#  w_init = pd.Series(index=returns.columns, data=1)
#  w_init.USDOLLAR = 0.
#  w_init /= sum(w_init)


# Point all on apple
w_init = pd.Series(index=returns.columns, data=0.)
w_init['AAPL'] = 1.


# Invest funds
funds = 1e4
h_init = w_init * funds

# Times
start = dt.date(2007, 1, 1)
#  end = dt.date(2017, 1, 1)
end = dt.date(2009, 1, 1)


# For loop propagating the market as in simulator.run_backtest
simulator = MarketSimulator(returns=returns,
                            volumes=volumes,
                            cash_key='USDOLLAR')

# Hold policy
#  results_hold = simulator.backtest(h_init, t_start=start, t_end=end,
#                                    policy=Hold(), log_level=logging.WARNING)
#  v_hold = [h.sum() for _, h in results_hold['h'].iterrows()]

# SinglePeriod policyy
sp_policy = SinglePeriod(returns_estimates, risk)
results_sp = simulator.backtest(h_init, t_start=start, t_end=end,
                                policy=sp_policy, log_level=logging.WARNING)

# Plot results
#  day = results_sp['h'].index
#  plt.figure()
#  #  plt.plot(day, v_hold, label='hold')
#  plt.plot(day, results_sp['v'], label='single period')
#  plt.title("Portfolio value")
#  plt.legend()
#  plt.show(block=False)

#  plt.figure()
#  plt.plot(day, prices.AAPL[results['h'].index])
#  plt.title("Apple price")
#  plt.show(block=False)

print(results_sp['stats'])

#  # TODO
#  - [x] Write check for randomness! Finish it in tests
#  - [x] Write stats risk/excess returns
#  - [ ] Write complete loop
#  - [ ] Write MLOPT full constraints selection
#  - [ ]