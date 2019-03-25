import cvxpy as cp
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm


def P_load_profile(horizon):
    """
    Generate desired power P_des trend
    """

    # The desired power is supposed to be piecewise linear. Each element has
    # length l[i] and angle a[i]

    np.random.seed(5)
    a = np.random.choice([-1, 1], 80) * .05 * np.random.rand(80)/10
    l = 20 * np.ones(80)

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


def control_problem(T=10, tau=1.0):
    # Parameters
    alpha = 6.7 * 1e-04
    beta = 0.2
    gamma = 80
    E_min = 5.2
    E_max = 10.2
    P_max = 1.2
    n_switch = 20

    # Initial values
    E_init = cp.Parameter(name='E_init')
    z_init = cp.Parameter(name='z_init')
    s_init = cp.Parameter(name='s_init')

    # Load profile
    P_load = cp.Parameter(T, name='P_load')
    #  E_init = (E_min + E_max) / 2.
    #  z_init = 0
    #  s_init = 0

    # Past d values
    # t = T - 2 => t - T = -2 => past_d[0]
    # t = 0 => t - T = -T => past_d[T-2]
    #  past_d = np.zeros(T - 1)
    past_d = cp.Parameter(T - 1, 'past_d')

    # Variables
    E = cp.Variable(T, name='E')  # Capacitor
    P = cp.Variable(T, name='P')      # Cell power
    s = cp.Variable(T, name='s')
    z = cp.Variable(T, name='z', boolean=True)
    w = cp.Variable(T, name='w')
    d = cp.Variable(T, name='d', boolean=True)
    d.value = np.zeros(T)

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

    return prob


def populate_parameters(problem, **params):
    for p in problem.parameters():
        p.value = params[p.name()]


def get_solution(problem):
    solution = {}
    for v in problem.variables():
        solution[v.name()] = v.value
    return solution


def update_past_d(problem, past_d):
    past_d_new = past_d[1:]
    for v in problem.variables():
        if v.name() == 'd':
            d_0 = v.value[0]
    past_d_new = np.append(past_d_new, d_0)
    return past_d_new


'''
Main code
'''
T_total = 100
T_horizon = 10
tau = 1.0
P_load = P_load_profile(T_total) + 0.05
#  P_load = 0.25 * np.ones(T_total)

# Define problem
problem = control_problem(T_horizon, tau=tau)

# Define loop
E_vec = [7.7]
z_vec = [0.]
s_vec = [0.]
P_vec = [0.]
past_d_vec = [np.zeros(T_horizon - 1)]
P_load_vec = [P_load[:T_horizon]]

sol_vec = []
n_sim = T_total - 2 * T_horizon
for t in tqdm(range(n_sim)):

    # Compute control
    params = {'E_init': E_vec[-1],
              'z_init': z_vec[-1],
              's_init': s_vec[-1],
              'past_d': past_d_vec[-1],
              'P_load': P_load_vec[-1]}
    populate_parameters(problem, **params)
    problem.solve(solver=cp.GUROBI)
    sol = get_solution(problem)
    sol_vec.append(sol)

    # Apply and propagate state
    E_vec.append(E_vec[-1] + tau * (sol['P'][0] - P_load[t]))
    z_vec.append(z_vec[-1] + sol['w'][0])
    s_vec.append(s_vec[-1] + sol['d'][0]
                 - past_d_vec[-1][T_horizon - 2])
    past_d_vec.append(update_past_d(problem, past_d_vec[-1]))
    #  past_d_vec.append(np.zeros(T_horizon - 1))
    P_load_vec.append(P_load[t+1:t+1 + T_horizon])
    P_vec.append(sol['P'][0])


#  t = range(len(P_load))
#  plt.figure()
#  plt.step(t, P_load, where='post')
#  plt.show(block=False)
#
# P_vec
#  P_vec = [x['P'][0] for x in sol_vec]

#  t_plot = t[:n_sim]
t_plot = range(n_sim)
plt.figure()
f, axarr = plt.subplots(4, sharex=True)
axarr[0].step(t_plot, E_vec[:n_sim], where='post', label="E")
axarr[0].legend()
axarr[1].step(t_plot, P_load[:n_sim], where='post', label='P_load')
axarr[1].legend()
axarr[2].step(t_plot, P_vec[:n_sim], where='post', label='P_vec')
axarr[2].legend()
axarr[3].step(t_plot, z_vec[:n_sim], where='post', label='z')
axarr[3].legend()
plt.show(block=False)


# Store parameter values


# Get number of strategies just from parameters


# Sample with balls around all the parameters


# Try to learn optimizer

#  P_load = P_load[100:]  # Choose only last one

#

# TODO: Write loop with parameters!
