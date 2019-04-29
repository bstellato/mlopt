# Needed for slurm
import os
import sys
sys.path.append(os.getcwd())

import online_optimization.control.utils as u
import logging
import numpy as np
import mlopt
import pickle
import argparse
import os
import pandas as pd

STORAGE_DIR = "/pool001/stellato/online/control"


def main():
    # Problem data
    T_total = 500
    tau = 1.0
    n_train = 5000
    n_sim_test = 500
    nn_params = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [32, 64],
        'n_epochs': [200, 300],
        'n_layers': [7, 10]
        #  {'learning_rate': 0.0001, 'batch_size': 64, 'n_epochs': 300, 'n_layers': 10}
        #  'learning_rate': [0.0001],
        #  'batch_size': [64],
        #  'n_epochs': [300],
        #  'n_layers': [10]
    }

    desc = 'Online Control Example'

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--horizon', type=int, default=10, metavar='N',
                        help='horizon length (default: 10)')
    arguments = parser.parse_args()
    T_horizon = arguments.horizon

    print(desc, " N = %d\n" % T_horizon)

    np.random.seed(0)

    # Output folder
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)

    EXAMPLE_NAME = STORAGE_DIR + '/online_control_%d_' % T_horizon

    # Get trajectory
    P_load = u.P_load_profile(T_total)

    # Create simulation data
    init_data = {'E': [7.7],
                 'z': [0.],
                 's': [0.],
                 'P': [],
                 'past_d': [np.zeros(T_horizon)],
                 'P_load': [P_load[:T_horizon]],
                 'sol': []}

    # Define problem
    problem = u.control_problem(T_horizon, tau=tau)

    sim_data = u.simulate_loop(problem, init_data,
                               u.basic_loop_solve,
                               P_load,
                               T_total)

    # Store simulation data as parameter values (avoid sol parameter)
    df = u.sim_data_to_params(sim_data)

    # Sample over balls around all the parameters
    df_train = u.sample_around_points(df,
                                      radius={'z_init': .5,  # .2,
                                              's_init': .5,  # .2,
                                              'P_load': 0.1,  # 0.01
                                              },
                                      n_total=n_train)

    m_mlopt = mlopt.Optimizer(problem.objective, problem.constraints,
                              log_level=logging.INFO)

    # Get samples
    m_mlopt._get_samples(df_train, parallel=True, condense_strategies=False)
    #  m_mlopt._compute_sample_strategy_pairs(parallel=True)
    m_mlopt.save_training_data(EXAMPLE_NAME + 'condensed.pkl',
                               delete_existing=True)

    #  m_mlopt.load_training_data(EXAMPLE_NAME + 'condensed.pkl')
    #  m_mlopt.condense_strategies()

    # Learn optimizer
    m_mlopt.train(learner=mlopt.PYTORCH,
                  n_best=10,
                  params=nn_params)

    # Generate test trajectory and collect points
    print("Simulate loop again to get trajectory points")
    # TODO: Change seed!
    P_load_test = u.P_load_profile(n_sim_test, seed=1)

    # Loop with basic function
    sim_data_test = u.simulate_loop(problem, init_data,
                                    u.basic_loop_solve,
                                    P_load_test,
                                    n_sim_test)
    # Loop with predictor
    sim_data_mlopt = u.simulate_loop(m_mlopt, init_data,
                                     u.predict_loop_solve,
                                     P_load_test,
                                     n_sim_test)

    # Evaluate open-loop performance on those parameters
    df_test = u.sim_data_to_params(sim_data_test)
    res_general, res_detail = m_mlopt.performance(df_test,
                                                  parallel=True,
                                                  use_cache=True)

    # Evaluate loop performance
    perf_solver = u.performance(problem, sim_data_test)
    perf_mlopt = u.performance(problem, sim_data_mlopt)
    res_general['perf_solver'] = perf_solver
    res_general['perf_mlopt'] = perf_mlopt
    res_general['perf_degradation_perc'] = 100 * (1. - perf_mlopt/perf_solver)

    # Export files
    with open(EXAMPLE_NAME + 'sim_data_mlopt.pkl', 'wb') as handle:
        pickle.dump(sim_data_mlopt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(EXAMPLE_NAME + 'sim_data_test.pkl', 'wb') as handle:
        pickle.dump(sim_data_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    res_general.to_csv(EXAMPLE_NAME + "test_general.csv",
                       header=True)
    res_detail.to_csv(EXAMPLE_NAME + "test_detail.csv")

    u.plot_sim_data(sim_data_mlopt, T_horizon, P_load_test,
                    title='sim_data_mlopt',
                    name=EXAMPLE_NAME)
    u.plot_sim_data(sim_data_test, T_horizon, P_load_test,
                    title='sim_data_test',
                    name=EXAMPLE_NAME)


if __name__ == '__main__':
    main()
