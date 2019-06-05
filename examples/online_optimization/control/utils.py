import numpy as np
import cvxpy as cp
import pandas as pd
from tqdm import tqdm
from mlopt.sampling import uniform_sphere_sample
from mlopt.settings import DIVISION_TOL
import matplotlib.pylab as plt
import copy
import random
from bisect import bisect


#  def P_load_profile(horizon,
#                     seed=5):
#      """
#      Generate desired power P_des trend
#      """
#
#      # The desired power is supposed to be piecewise linear. Each element has
#      # length l[i] and angle a[i]
#
#      np.random.seed(seed)
#      a = np.random.choice([-1, 1], 80) * .05 * np.random.rand(80)/10
#      l = 20 * np.ones(80)
#
#      # Get required power
#      P_des = np.arange(a[0], (a[0]*l[0]) + a[0], a[0])
#
#      for i in range(1, len(l)):
#          P_des = np.append(P_des, np.arange(
#              P_des[-1]+a[i], P_des[-1] + a[i]*l[i] + a[i], a[i]))
#
#      # Slice P_des to match the step length (tau = 4, 5 steps each 20.)
#      P_des = P_des[0:len(P_des) + 1:5]
#
#      # Slice up to get the desired horizon
#      P_des = P_des[:horizon]
#
#      # Get only positive values
#      P_des = np.maximum(P_des, 0)
#
#
#      return P_des


def plot_sim_data(sim_data, T_horizon,
                  P_load, title='Subplots', name='name'):
    T_total = len(P_load)
    n_sim = T_total - 2 * T_horizon
    t_plot = range(n_sim)
    f, axarr = plt.subplots(5, sharex=True)
    f.suptitle(title)  # or plt.suptitle('Main title')
    axarr[0].plot(t_plot, sim_data['E'][:n_sim], label="E")
    axarr[0].legend()
    axarr[1].plot(t_plot, P_load[:n_sim], label='P_load')
    axarr[1].legend()
    axarr[2].step(t_plot, sim_data['P'][:n_sim], where='post',
                  label='P_vec')
    axarr[2].legend()
    axarr[3].step(t_plot, sim_data['z'][:n_sim], where='post',
                  label='z')
    axarr[3].legend()
    axarr[4].step(t_plot, sim_data['s'][:n_sim], where='post',
                  label='s')
    axarr[4].legend()
    plt.tight_layout()
    plt.savefig(name + title + ".pdf")


def P_load_profile(T, seed=0):
    """
    Generate desired power P_des trend
    """
    random.seed(seed)

    t_bp = 2. * np.array([0., 2., 4., 6., 8., 10.,
                          15., 18., 27., 42., 56.,
                          53., 76., 92., 104., 107,
                          113., 125., 129., 134., 148.,
                          192., 176., 192., 200.])
    t_period = t_bp[-1]

    # Get number of periods
    n_periods = T // int(t_period)

    P_p = np.zeros(int(t_period))

    # Assign period values
    P_bp = .25 * np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5,
                          0.5, 0.2, 0.4, 0.25, 0.3,
                          0.0, 0.1, 0.0, 0.3, 0.4,
                          0.5, 0.35, 0.45, 0.25, 0.15,
                          0.3, 0.35, 0.25, 0.0, 0.0])
    random.shuffle(P_bp)   # Shuffle values

    t = np.arange(t_period)

    # Assign breakpoints to actual values
    for t_idx in range(int(t_period)):
        P_p[t_idx] = P_bp[bisect(t_bp, t[t_idx]) - 1]

    P = np.tile(P_p, n_periods + 1)[:T]

    return P


def control_problem(T=10,
                    tau=1.0,
                    alpha=6.7 * 1e-04,
                    beta=0.2,
                    gamma=0.08,
                    delta=0.1,
                    E_min=5.2,
                    E_max=10.2,
                    P_max=1.2,
                    n_switch=5):

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

    # past_d[0] => t - T = 0 - T
    # past_d[T-1] => t - T = T -1 - T = -1
    past_d = cp.Parameter(T, 'past_d')

    # Variables
    E = cp.Variable(T, name='E')  # Capacitor
    P = cp.Variable(T, name='P')      # Cell power
    s = cp.Variable(T+1, name='s')
    z = cp.Variable(T, name='z', boolean=True)
    w = cp.Variable(T, name='w', integer=True)
    d = cp.Variable(T, name='d', boolean=True)

    # Constraints
    bounds = []
    #  bounds += [E_min <= E, E <= E_max]
    bounds += [0 <= P, P <= z * P_max]
    bounds += [-1 <= w, w <= 1]
    #  bounds += [0. <= z, z <= 1]

    # Capacitor dynamics
    capacitor_dynamics = []
    for t in range(T-1):
        capacitor_dynamics += [E[t+1] == E[t] + tau * (P[t] - P_load[t])]

    # Switch positions
    switch_positions = [z[t+1] == z[t] + w[t] for t in range(T-1)]

    switch_accumulation = []
    for t in range(T):
        switch_accumulation += [s[t+1] == s[t] + d[t] - past_d[t]]

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
        capacitor_dynamics + \
        initial_values + \
        switch_positions + \
        switch_accumulation + \
        number_switchings + \
        logics

    # Objective
    cost = 0
    for t in range(T-1):
        cost += alpha * (P[t]) ** 2 + beta * P[t] + gamma * z[t]
        cost += delta * (cp.pos(E_min - E[t+1]) + cp.pos(E[t+1] - E_max))

    # DEBUG: Regularize for consistency
    #  cost += 0.2 * cp.sum_squares(z)

    objective = cp.Minimize(cost)

    prob = cp.Problem(objective, constraints)

    return prob


def populate_parameters(problem, params):
    for p in problem.parameters():
        p.value = params[p.name()]


def get_solution(problem):
    solution = {}
    for v in problem.variables():
        solution[v.name()] = v.value
    return solution


def update_past_d(problem, past_d):
    past_d_new = np.copy(past_d[1:])  # DEBUG
    for v in problem.variables():
        if v.name() == 'd':
            d_0 = v.value[0]
    past_d_new = np.append(past_d_new, d_0)
    return past_d_new


def sim_data_to_params(sim_data):
    """Convert simulation data to training parameters"""
    df = pd.DataFrame({k: v for k, v in sim_data.items()
                       if k not in ['sol', 'P']})
    df.columns = ['%s_init' % x
                  if x in ['E', 's', 'z'] else x for x in df.columns]
    return df


def basic_loop_solve(problem, params):
    populate_parameters(problem, params)
    #  problem.get_problem_data(cp.GUROBI)
    problem.solve(solver=cp.GUROBI)
    if problem.status != 'optimal':
        raise ValueError('Error in Gurobi solution')
    return get_solution(problem)


def predict_loop_solve(optimizer, params):
    #  optimizer._problem.populate(pd.Series(params))
    #  x_test = optimizer._problem.solve()
    x_pred = optimizer.solve(pd.Series(params))
    #  if np.linalg.norm(x_pred['x'] - x_test['x']) > 1e-04:
    #      print("\nmismatch:\n")
    #      print("distance x = ", np.linalg.norm(x_pred['x'] - x_test['x']))
    #      print("infeasibility test: ", x_test['infeasibility'])
    #      print("infeasibility pred: ", x_pred['infeasibility'])
    #      print("cost test: ", x_test['cost'])
    #      print("cost pred: ", x_pred['cost'])
    #      print("\n\n")
    # Round parameters

    return get_solution(optimizer._problem)


def is_sol_equal(sol1, sol2):
    comparison = True
    for k in sol1.keys():
        if k != 'sol':
            if not np.allclose(sol1[k], sol2[k]):
                print("Different %s" % k)
                comparison = False
    return comparison


def simulate_loop(problem,
                  init_data,
                  solve_fn,
                  P_load,
                  T_total,
                  tau=1.0):

    T_horizon = [x for x in problem.parameters()
                 if x.name() == 'P_load'][0].shape[0]

    sim_data = copy.deepcopy(init_data)

    n_sim = T_total - 2 * T_horizon
    for t in tqdm(range(n_sim)):

        # Compute control
        params = {'E_init': sim_data['E'][-1],
                  'z_init': sim_data['z'][-1],
                  's_init': sim_data['s'][-1],
                  'past_d': sim_data['past_d'][-1],
                  'P_load': sim_data['P_load'][-1]}
        sol = solve_fn(problem, params)

        sim_data['sol'].append(sol)

        # Apply and propagate state
        sim_data['E'].append(sim_data['E'][-1] +
                             tau * (sol['P'][0] - P_load[t]))
        sim_data['z'].append(sim_data['z'][-1] + sol['w'][0])
        #  sim_data['s'].append(sim_data['s'][-1] + sol['d'][0]
        #                       - sim_data['past_d'][-1][T_horizon - 2])
        sim_data['s'].append(sim_data['s'][-1] + sol['d'][0]
                             - sim_data['past_d'][-1][0])
        sim_data['past_d'].append(update_past_d(problem,
                                                sim_data['past_d'][-1]))
        sim_data['P_load'].append(P_load[t+1:t+1 + T_horizon])
        sim_data['P'].append(sol['P'][0])

    return sim_data


def performance(problem, sim_data):
    P = sim_data['P']
    z = sim_data['z']
    n_sim = len(P)
    alpha = problem.objective.args[0].args[-3].args[0].value
    beta = problem.objective.args[0].args[-2].args[0].value
    gamma = problem.objective.args[0].args[-1].args[0].value
    cl_cost = 0
    for i in range(n_sim):
        cl_cost += alpha * (P[i] ** 2) + beta * P[i] + gamma * z[i]

    return cl_cost


def sample_around_points(df,
                         n_total=8000,
                         radius={}):
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

            samples = uniform_sphere_sample(row[col], rad,
                                            n=n_samples_per_point)

            # Round stuff
            if col in ['past_d']:
                samples = np.maximum(np.around(samples, decimals=0),
                                     0).astype(int)

            elif col == 's_init':
                samples = np.minimum(np.maximum(np.around(samples, decimals=0),
                                     0), 5).astype(int)

            elif col == 'z_init':
                samples = np.minimum(np.maximum(
                    np.around(samples, decimals=0), 0), 1).astype(int)

            elif col in ['P_load']:
                samples = np.maximum(samples, 0)

            elif col in ['E_init']:
                samples = np.minimum(np.maximum(samples, 5.2), 10.2)

            if len(samples[0]) == 1:
                # Flatten list
                samples = [item for sublist in samples for item in sublist]

            df_row[col] = list(samples)

        df_samples = df_samples.append(df_row)

    return df_samples
