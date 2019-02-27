import pandas as pd
# Write basic simulation code with basic strategies


# Load data
with pd.HDFStore("./data/simulation_data.h5") as sim_data:
    returns = sim_data['returns']
    returns_estimates = sim_data['returns_estimates']


# Initial portfolio allocation (equally divided over the assets
w_init = pd.Series(index=returns.columns, data=1)
w_init.USDOLLAR = 0.
w_init /= sum(w_init)

# Define basic function to perform optimization

# For loop propagating the market as in simulator.run_backtest
