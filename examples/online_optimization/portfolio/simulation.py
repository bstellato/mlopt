class MarketSimulator(object):
    def __init__(self, returns=None, volumes=None, cash_key='cash'):
        self.returns = returns
        self.volumes = volumes
        self.cash_key = cash_key

    def propagate(self, h, u, t):
        """
        Propagate portfolio h over time period t given trades u.
        """
        h_plus = h + u  # Value + trades
        costs = 0  # Add actual costs
        u[self.cash_key] = - sum([u.index != self.cash_key]) - sum(costs)
        h_plus[self.cash_key] = h[self.cash_key] + u[self.cash_key]

        h_next = self.returns.loc[t] * h_plus + h_plus

        return h_next




