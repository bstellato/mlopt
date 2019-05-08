import logging
import numpy as np
import pandas as pd
from tqdm import tqdm


class MarketSimulator(object):
    def __init__(self, returns=None,
                 volumes=None,
                 borrow_cost=0.0001,
                 bid_ask_spread=0.001,
                 cash_key='cash'):
        self.returns = returns
        self.volumes = volumes
        self.borrow_cost = borrow_cost
        self.bid_ask_spread = bid_ask_spread
        self.cash_key = cash_key

    def holding_cost(self, h_plus):
        h_plus_stock = h_plus[h_plus.index != self.cash_key].values
        h_neg = np.minimum(h_plus_stock, 0)
        return np.sum(self.borrow_cost * h_neg)

    def transaction_cost(self, u):
        """Linear transaction cost model
        cost = bid_ask_spread * |u|
        """
        u_stock = u[u.index != self.cash_key].values
        cost = .5 * self.bid_ask_spread * np.abs(u_stock)
        return np.sum(cost)

    def propagate(self, h, u, t):
        """
        Propagate portfolio h over time period t given trades u.
        """
        h_plus = h + u  # Value + trades

        # Costs
        holding_cost = self.holding_cost(h_plus)
        transaction_cost = self.transaction_cost(u)
        costs = holding_cost + transaction_cost

        # Remove money from cash
        money_out = - sum(u[u.index != self.cash_key])

        u[self.cash_key] = money_out - costs
        h_plus[self.cash_key] = h[self.cash_key] + u[self.cash_key]

        h_next = self.returns.loc[t] * h_plus + h_plus

        return h_next

    def backtest(self, h_init, t_start, t_end, policy,
                 log_level=logging.INFO, verbose=False):
        """Run portfolio backtest"""

        logging.basicConfig(level=log_level)

        h_log, u_log = pd.DataFrame(), pd.DataFrame()

        h = h_init  # Initial portfolio
        h_log = h_log.append(h, ignore_index=True)

        times = self.returns.index[
            (self.returns.index >= str(t_start)) &
            (self.returns.index <= str(t_end))]

        logging.info("Backtest from %s to %s" % (times[0], times[-1]))

        for t in tqdm(times, desc="Backtest simulation"):
            logging.debug("Getting trades at time %s" % t)

            u = policy.trades(h, t, verbose=verbose)
            h = self.propagate(h, u, t)

            # Log results
            h_log = h_log.append(h, ignore_index=True)
            u_log = u_log.append(u, ignore_index=True)

        logging.info("Backtest ended.")

        # Time data
        t_final = self.returns[self.returns.index > str(t_end)].index[0]
        h_log['t'] = times.append(pd.Index([t_final]))
        h_log = h_log.set_index('t')
        u_log['t'] = times
        u_log = u_log.set_index('t')
        v = h_log.sum(axis=1)  # Value of the portfolio over time
        returns = v.values[1:] / v.values[:-1] - 1  # (v_{t+1} - v_{t})/v_{t}
        risk_free_returns = self.returns.loc[times, self.cash_key]
        excess_returns = returns - risk_free_returns

        # Statistics
        mean_return = returns.mean() * 250 * 100
        mean_excess_return = excess_returns.mean() * 250 * 100
        sigma_excess_return = \
            excess_returns.std() * np.sqrt(250) * 100  # 250 days, 100 perc

        stats = pd.Series({'Mean return (%)': mean_return,
                           'Mean excess return (%)': mean_excess_return,
                           'Excess risk (%)': sigma_excess_return})

        return {'h': h_log,
                'u': u_log,
                'v': v,
                'returns': returns,
                'excess_returns': excess_returns,
                'stats': stats}
