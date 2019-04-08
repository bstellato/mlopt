import examples.online_optimization.control.utils as u
import logging
import numpy as np
import mlopt
import pickle
import matplotlib.pylab as plt


np.random.seed(0)

DATA_FILE = 'online_control.pkl'

'''
Get trajectory
'''
T_total = 180
T_horizon = 20
tau = 1.0
P_load = u.P_load_profile(T_total)

# Define problem
problem = u.control_problem(T_horizon, tau=tau)

# Solve single point with optimizer
#  m.solve(theta)

#  with open('sim_data.pkl', 'rb') as h:
#      sim_data_load = pickle.load(h)

# Create simulation data
#  init_data = {'E': [7.7],
#               'z': [0.],
#               's': [0.],
#               'P': [],
#               'past_d': [np.zeros(T_horizon)],
#               'P_load': [P_load[:T_horizon]],
#               'sol': []}
#
#  sim_data = u.simulate_loop(problem, init_data,
#                             u.basic_loop_solve,
#                             P_load,
#                             T_total)

#  with open('sim_data.pkl', 'wb') as h:
    #  pickle.dump(sim_data, h)


# Compare dictionaries
#  def is_sim_data_equal(data1, data2):
#      comparison = True
#      for k in data1.keys():
#          if k is not 'sol':
#              if not np.allclose(data1[k], data2[k]):
#                  print("Different %s" % k)
#                  comparison = False
#      return comparison
#
#
#  comparison = is_sim_data_equal(sim_data, sim_data_load)

# Store simulation data as parameter values (avoid sol parameter)
#  df = u.sim_data_to_params(sim_data)


# Sample over balls around all the parameters
#  df_train = u.sample_around_points(df,
#                                    radius={'z_init': .5,  # .2,
#                                            's_init': .5,  # .2,
#                                            'P_load': 0.1,  # 0.01
#                                            },
#                                    n_total=1000)


# DEBUG Check number of strategies in simulation
#  print("Get samples from simulation")
#  m_mlopt = mlopt.Optimizer(problem.objective, problem.constraints,
#                            log_level=logging.DEBUG)
#  m_mlopt._get_samples(df, parallel=False)


# Check if sim_data changes

# Get number of strategies just from parameters
#  print("Get samples normally")
m_mlopt = mlopt.Optimizer(problem.objective, problem.constraints,
                          log_level=logging.INFO)
#  m_mlopt._get_samples(df_train, parallel=False)
#  m_mlopt.save_training_data(DATA_FILE, delete_existing=True)
#

#  # Learn optimizer
#  params = {
#      'learning_rate': [0.001, 0.01, 0.1],
#      'batch_size': [32, 64, 128],
#      'n_epochs': [1000, 1500]
#  }
m_mlopt.load_training_data(DATA_FILE)
m_mlopt.condense_strategies()

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
#
#  # Plot
#  # Plot only load
#  #  n_sim_test = 100
#  #  P_load_test = u.P_load_profile(n_sim_test, seed=0)
#  #  t_plot = range(n_sim_test)
#  #  plt.figure()
#  #  plt.plot(t_plot, P_load_test, label='P_load_test')
#  #  plt.legend()
#  #  #  plt.savefig('P_load_test.pdf')
#  #  plt.show(block=False)
#
#
#
#  #  def plot_sim_data(sim_data, P_load, title='Subplots'):
#  #      n_sim = T_total - 2 * T_horizon
#  #      t_plot = range(n_sim)
#  #      f, axarr = plt.subplots(5, sharex=True)
#  #      f.suptitle(title)  # or plt.suptitle('Main title')
#  #      axarr[0].plot(t_plot, sim_data['E'][:n_sim], label="E")
#  #      axarr[0].legend()
#  #      axarr[1].plot(t_plot, P_load[:n_sim], label='P_load')
#  #      axarr[1].legend()
#  #      axarr[2].step(t_plot, sim_data['P'][:n_sim], where='post', label='P_vec')
#  #      axarr[2].legend()
#  #      axarr[3].step(t_plot, sim_data['z'][:n_sim], where='post', label='z')
#  #      axarr[3].legend()
#  #      axarr[4].step(t_plot, sim_data['s'][:n_sim], where='post', label='s')
#  #      axarr[4].legend()
#  #      plt.show(block=False)
#  #
#  #
#  #  plot_sim_data(sim_data, P_load, title='sim_data')
#  #  plot_sim_data(sim_data_load, P_load, title='sim_data_load')
