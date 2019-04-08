import examples.online_optimization.control.utils as u
import logging
import numpy as np
import mlopt
import pickle
import matplotlib.pylab as plt


np.random.seed(0)

DATA_FILE = 'online_control_small.pkl'

'''
Get trajectory
'''
T_total = 180
T_horizon = 20
tau = 1.0
P_load = u.P_load_profile(T_total)

# Define problem
problem = u.control_problem(T_horizon, tau=tau)

# Create simulation data
init_data = {'E': [7.7],
             'z': [0.],
             's': [0.],
             'P': [],
             'past_d': [np.zeros(T_horizon)],
             'P_load': [P_load[:T_horizon]],
             'sol': []}

sim_data = u.simulate_loop(problem, init_data,
                           u.basic_loop_solve,
                           P_load,
                           T_total)


# Store simulation data as parameter values (avoid sol parameter)
#  df = u.sim_data_to_params(sim_data)


# Sample over balls around all the parameters
#  df_train = u.sample_around_points(df,
#                                    radius={'z_init': .5,  # .2,
#                                            's_init': .5,  # .2,
#                                            'P_load': 0.5,  # 0.01
#                                            },
#                                    n_total=10000)

# Get number of strategies just from parameters
m_mlopt = mlopt.Optimizer(problem.objective, problem.constraints,
                          log_level=logging.INFO)
#  m_mlopt._get_samples(df_train, parallel=True)
#  m_mlopt.save_training_data(DATA_FILE, delete_existing=True)


#  # Learn optimizer
#  params = {
#      'learning_rate': [0.001, 0.01, 0.1],
#      'batch_size': [32, 64, 128],
#      'n_epochs': [1000, 1500]
#  }
m_mlopt.load_training_data(DATA_FILE)
m_mlopt.condense_strategies(k_max_strategies=100, parallel=False)

#  m_mlopt.train(parallel=False,
#                learner=mlopt.PYTORCH,
#                n_best=10,
#                params=params)

#  # Generate test trajectory and collect points
#  n_sim_test = 100
#  P_load_test = u.P_load_profile(n_sim_test, seed=1)
#  sim_data_test = u.simulate_loop(problem, init_data,
#                                  u.basic_loop_solve,
#                                  P_load_test,
#                                  n_sim_test)
#  df_test = u.sim_data_to_params(sim_data_test)
#
#
#  # Evaluate performance on those parameters
#  res_general, res_detail = m_mlopt.performance(df_test,
#                                                parallel=False,
#                                                use_cache=False)
#  res_general.to_csv("./output/online_control_general.csv")
#  res_detail.to_csv("./output/online_control_detail.csv")
