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


STORAGE_DIR = "/home/gridsan/stellato/results/online/control"


np.random.seed(1)


if __name__ == '__main__':

    desc = 'Online Control Example'

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--horizon', type=int, default=10, metavar='N',
                        help='horizon length (default: 10)')
    arguments = parser.parse_args()
    T_horizon = arguments.horizon

    EXAMPLE_NAME = STORAGE_DIR + '/control_%d_' % T_horizon

    # Problem data
    T_total = 500
    tau = 1.0
    n_train = 100000
    n_sim_test = 1000
    nn_params = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [32],
        'n_epochs': [20],
        # OLD STUFF
        # 'n_layers': [7, 10]
        #  {'learning_rate': 0.0001, 'batch_size': 64, 'n_epochs': 300, 'n_layers': 10}
        #  'learning_rate': [0.0001],
        #  'batch_size': [64],
        #  'n_epochs': [300],
        #  'n_layers': [10]
    }

    logging.info(desc, " N = %d\n" % T_horizon)

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

    print("Get learning data by simulating closed loop")
    sim_data = u.simulate_loop(problem, init_data,
                               u.basic_loop_solve,
                               P_load,
                               T_total)

    # Store simulation data as parameter values (avoid sol parameter)
    df = u.sim_data_to_params(sim_data)

    # Create mlopt problem
    m_mlopt = mlopt.Optimizer(problem.objective, problem.constraints,
                              log_level=logging.INFO,
                              parallel=True)


    # Check if learning data already there
    if not os.path.isfile(EXAMPLE_NAME + 'data.pkl'):


        # Sample over balls around all the parameters
        df_train = u.sample_around_points(df,
                                          radius={'z_init': .4,  # .2,
                                                  #  's_init': .6,  # .2,
                                                  'P_load': 0.001,  # 0.01
                                                  },
                                          n_total=n_train)


        # Get samples
        m_mlopt.get_samples(df_train,
                            parallel=True,
                            filter_strategies=False)
        #  m_mlopt._compute_sample_strategy_pairs(parallel=True)
        m_mlopt.save_training_data(EXAMPLE_NAME + 'data.pkl',
                                   delete_existing=True)
    else:
        print("Loading data from file")
        m_mlopt.load_training_data(EXAMPLE_NAME + 'data.pkl')

        # DO NOT FILTER STRATEGIES
        # m_mlopt.filter_strategies()
        # m_mlopt.save_training_data(EXAMPLE_NAME + 'data_filtered.pkl',
        #                            delete_existing=True)

    # # Learn optimizer
    m_mlopt.train(learner=mlopt.PYTORCH,
                  n_best=10,
                  filter_strategies=False,
                  parallel=False,
                  params=nn_params)

    # # Generate test trajectory and collect points
    logging.info("Simulate loop again to get trajectory points")
    P_load_test = u.P_load_profile(n_sim_test, seed=1)

    sim_data_test = u.simulate_loop(problem, init_data,
                                    u.basic_loop_solve,
                                    P_load_test,
                                    T_total)

    # Evaluate open-loop performance on those parameters
    df_test = u.sim_data_to_params(sim_data_test)
    res_general, res_detail = m_mlopt.performance(df_test,
                                                  parallel=False,
                                                  use_cache=True)

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
