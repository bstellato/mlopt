import cvxpy as cp
import numpy as np


def P_load_profile(horizon):
    """
    Generate desired power P_des trend
    """

    # The desired power is supposed to be piecewise linear. Each element has
    # length l[i] and angle a[i]

    np.random.seed(5)
    a = np.random.choice([-1, 1], 80) * .05 * np.random.rand(80)/10
    l = 20 * np.ones(80)

    #  a = np.array([0.5, -0.5,  0.2, -0.7,  0.6, -
    #                0.2,  0.7, -0.5,  0.8, -0.4,
    #                0.2, 0.1, -0.1, -0.3, 0.2, 0.1, -0.1]) / 10.
    #  l = np.array([40., 20., 40., 40., 20., 40., 30., 40.,
    #                30., 60., 40., 20., 40., 30., 40., 20., 60.])

    # Get required power
    P_des = np.arange(a[0], (a[0]*l[0]) + a[0], a[0])

    for i in range(1, len(l)):
        P_des = np.append(P_des, np.arange(
            P_des[-1]+a[i], P_des[-1] + a[i]*l[i] + a[i], a[i]))

    # Slice P_des to match the step length (tau = 4, 5 steps each 20.)
    P_des = P_des[0:len(P_des) + 1:5]

    # Slice up to get the desired horizon
    P_des = P_des[:horizon]

    # Get only positive values
    P_des = np.maximum(P_des, 0)

    return P_des


def control_problem(T=10, P_load=P_load_profile(10)):
    # Parameters
    alpha = 6.7 * 1e-04
    beta = 0.2
    gamma = 80
    E_min = 5.2
    E_max = 10.2
    P_max = 1.2
    n_switch = 10
    tau = 1.0

    # Initial values
    E_init = (E_min + E_max) / 2.
    z_init = 0
    s_init = 0

    # Past d values
    # t = T - 2 => t - T = -2 => past_d[0]
    # t = 0 => t - T = -T => past_d[T-2]
    past_d = np.zeros(T - 1)

    # Variables
    E = cp.Variable(T)  # Capacitor
    P = cp.Variable(T)      # Cell power
    s = cp.Variable(T)
    z = cp.Variable(T, boolean=True)
    w = cp.Variable(T)
    d = cp.Variable(T, boolean=True)

    variables = {'E': E,
                 'P': P,
                 'z': z}

    # Constraints
    bounds = []
    bounds += [E_min <= E, E <= E_max]
    bounds += [0 <= P, P <= z * P_max]
    bounds += [-1 <= w, w <= 1]

    # Capacitor dynamics
    capacitor_dynamics = []
    for t in range(T-1):
        capacitor_dynamics += [E[t+1] == E[t] + tau * (P[t] - P_load[t])]

    # Switch positions
    switch_positions = [z[t+1] == z[t] + w[t] for t in range(T-1)]

    switch_accumulation = []
    for t in range(T - 1):
        if t - T >= 0:
            switch_accumulation += [s[t+1] == s[t] + d[t]
                                    - d[t - T]]
        else:
            switch_accumulation += [s[t+1] == s[t] + d[t]
                                    - past_d[abs(t - T + 2)]]

    # Number of switchings
    number_switchings = [s <= n_switch]

    # Logical relationships
    G = np.array([[1., 0., -1.],
                  [-1., 0., -1.],
                  [1., 2., 2.],
                  [-1., -2., 2.]])
    g = np.array([0., 0., 3., 1.])

    logics = []
    for t in range(T):
        logics += [G[:, 0] * w[t] + G[:, 1] * z[t] + G[:, 2] * d[t] <= g]

    # Initial values (parameters)
    initial_values = [E[0] == E_init, z[0] == z_init, s[0] == s_init]

    constraints = bounds + \
        capacitor_dynamics + initial_values + \
        switch_positions + \
        switch_accumulation + \
        number_switchings + \
        logics

    # Objective
    cost = 0
    for t in range(T-1):
        cost += alpha * (P[t]) ** 2 + beta * P[t] + gamma * z[t]
    objective = cp.Minimize(cost)

    prob = cp.Problem(objective, constraints)

    return prob, variables


# Main code
P_load = P_load_profile(180)
#  P_load = P_load[100:]  # Choose only last one

import matplotlib.pylab as plt
t = range(len(P_load))
plt.figure()
plt.step(t, P_load, where='post')
plt.show(block=False)

T = 20
prob, variables = control_problem(T=T, P_load=P_load)
prob.solve(solver=cp.GUROBI, verbose=True)


t = range(T)
plt.figure()
plt.step(t, variables['P'].value, where='post', label="P")
plt.legend()
plt.show(block=False)

plt.figure()
plt.step(t, variables['z'].value, label="z", where='post')
plt.legend()
plt.show(block=False)

plt.figure()
plt.step(t, variables['E'].value, label="E", where='post')
plt.legend()
plt.show(block=False)


# TODO: Write loop with parameters!
