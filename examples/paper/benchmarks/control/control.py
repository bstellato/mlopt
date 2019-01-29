# Needed for slurm
import os
import sys
sys.path.append(os.getcwd())

# Standard imports
from mlopt.sampling import uniform_sphere_sample
from mlopt.utils import benchmark
import mlopt
import numpy as np
import scipy.sparse as spa
import cvxpy as cp
import pandas as pd


def generate_P_des(T):
    """
    Generate desired power P_des trend
    """

    # The desired power is supposed to be piecewise linear. Each element has
    # length l[i] and angle a[i]
    a = np.array([0.5, -0.5,  0.2, -0.7,  0.6, -
                  0.2,  0.7, -0.5,  0.8, -0.4]) / 10.
    l = np.array([40., 20., 40., 40., 20., 40., 30., 40., 30., 60.])

    # Get required power
    P_des = np.arange(a[0], (a[0]*l[0]) + a[0], a[0])

    for i in range(1, len(l)):
        P_des = np.append(P_des, np.arange(
            P_des[-1]+a[i], P_des[-1] + a[i]*l[i] + a[i], a[i]))

    # Slice P_des to match the step length (tau = 4, 5 steps each 20.)
    P_des = P_des[0:len(P_des):5]

    # Slice up to get the desired horizon
    P_des = P_des[:T]

    return P_des


np.random.seed(1)
name = "control"

# Output folder
output_folder = "output/" + name
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Define data
# The only dimension changing in these examples is the
# horizon length
#  T_vec = np.array([5, 10, 20, 30, 40], dtype=int)
T_vec = np.array([10, 15, 20, 25, 30, 35, 40], dtype=int)


# Function to sample points
def sample(theta_bar, n=100):

    # Get bar values for certain parameters
    E_0_bar = [theta_bar[0]]
    P_des_bar = theta_bar[1:]

    # Sample points from multivariate ball
    X_E_0 = uniform_sphere_sample(E_0_bar, 0.5, n=n)
    X_P_des = uniform_sphere_sample(P_des_bar, 0.5, n=n)

    df = pd.DataFrame({'E_0': X_E_0.tolist(),
                       'P_des': X_P_des.tolist()})

    return df


# Main script
results_general = pd.DataFrame()
results_detail = pd.DataFrame()


for T in T_vec:
    '''
    Define control problem
    '''

    # Generate example
    tau = 4.    # length of the discretization time interval

    # Parameters
    P_des_bar = generate_P_des(T)
    E_0_bar = 40.    # Initial charge

    # Constraints on electric charge
    E_max = 50.  # Maximum charge
    E_0 = cp.Parameter(nonneg=True, name='E_0')

    # Constraints on power
    P_max = 1.
    P_des = cp.Parameter(T, name='P_des')

    # Previous binary input
    z_prev = 0.  # z_{-1}

    # Define quadratic cost
    alpha = 1.   # quadratic term
    beta = 1.   # linear term
    gamma = 1.  # constant term
    delta = 1.  # cost of turning on the switches
    eta = 0.1    # Penalty on last stage

    # Variables
    E = cp.Variable(T+1)
    P_batt = cp.Variable(T)
    P_eng = cp.Variable(T)
    z = cp.Variable(T, integer=True)
    h = cp.Variable(T)  # For maximum

    # Constraints
    constraints = []

    # Dynamics E_{t+1} = E_t - tau P_batt
    for t in range(T):
        constraints += [E[t+1] == E[t] - tau * P_batt[t]]
    constraints += [E[0] == E_0]

    # 0 <= P_eng <= P_max * z
    for t in range(T):
        constraints += [0 <= P_eng[t], P_eng[t] <= P_max * z[t]]

    # 0 <= E <= E_max
    #  constraints += [E[t] <= E_max for t in range(T)]
    constraints += [E >= 0, E <= E_max]

    # P_des <= P_batt + P_eng
    constraints += [P_des[t] <= P_batt[t] + P_eng[t] for t in range(T)]

    # Constrain z
    constraints += [0 <= z, z <= 1]

    # Cost
    cost = eta * (E[T] - E_max) ** 2
    for t in range(T):
        cost += alpha * (P_eng[t]) ** 2 + beta * P_eng[t] + gamma * z[t]
        # Add (z - z_prev) constraints
        cost += delta * h[t]
        constraints += [h[t] >= 0]
        if t == 0:
            constraints += [h[t] >= (z[t] - z_prev)]
        else:
            constraints += [h[t] >= (z[t] - z[t-1])]
        #  if t == 0:
        #      cost += delta * cp.pos(z[t] - z_prev)
        #  else:
        #      cost += delta * cp.pos(z[t] - z[t-1])

    # Define optimizer
    m = mlopt.Optimizer(cp.Minimize(cost), constraints,
                        name=name)

    '''
    Define parameters average values
    '''
    theta_bar = np.concatenate(([E_0_bar], P_des_bar))

    '''
    Train and solve
    '''

    # Train and test using pytorch
    data_file = os.path.join(output_folder, "%s_T%d" %
                             (name, T))

    # Benchmark and append results
    temp_general, temp_detail = benchmark(m, data_file,
                                          theta_bar,
                                          lambda n: sample(theta_bar, n),
                                          {'T': T}
                                          )
    results_general = results_general.append(temp_general)
    results_detail = results_detail.append(temp_detail)

# Store cumulative results
results_general.to_csv(os.path.join(output_folder,
                                    "%s_general.csv" % name))
results_detail.to_csv(os.path.join(output_folder,
                                   "%s_detail.csv" % name))
