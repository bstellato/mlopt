import examples.online_optimization.control.utils as u
import logging
import numpy as np
import mlopt
import pickle
import matplotlib.pylab as plt
import argparse

STORAGE_DIR = "/pool001/stellato"


def main():
    parser = argparse.ArgumentParser(description='Online Control Example')
    parser.add_argument('--horizon', type=int, default=10, metavar='N',
                        help='horizon length (default: 10)')
    arguments = parser.parse_args()
    T_horizon = arguments.horizon

    np.random.seed(0)

    EXAMPLE_NAME = STORAGE_DIR + '/online_control_%d' % T_horizon
    DATA_FILE = EXAMPLE_NAME + '.pkl'

    '''
    Get trajectory
    '''
    T_total = 180
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

    #  sim_data = u.simulate_loop(problem, init_data,
    #                             u.basic_loop_solve,
    #                             P_load,
    #                             T_total)

    # Store simulation data as parameter values (avoid sol parameter)
    #  df = u.sim_data_to_params(sim_data)

    # Sample over balls around all the parameters
    #  df_train = u.sample_around_points(df,
    #                                    radius={'z_init': .5,  # .2,
    #                                            's_init': .5,  # .2,
    #                                            'P_load': 0.25,  # 0.01
    #                                            },
    #                                    n_total=5000)

    # Get number of strategies just from parameters
    print("Get samples normally")
    m_mlopt = mlopt.Optimizer(problem.objective, problem.constraints,
                              log_level=logging.INFO)

    # Get samples
    #  m_mlopt._get_samples(df_train, parallel=True, condense_strategies=False)
    #  m_mlopt._compute_sample_strategy_pairs()
    #  m_mlopt.save_training_data(DATA_FILE, delete_existing=True)

    # Load training data
    #  m_mlopt.load_training_data(DATA_FILE)
    #  m_mlopt.condense_strategies(parallel=True)
    #  m_mlopt.save_training_data(EXAMPLE_NAME + '_condensed.pkl',
    #                             delete_existing=True)

    m_mlopt = mlopt.Optimizer(problem.objective, problem.constraints,
                              log_level=logging.INFO)
    m_mlopt.load_training_data(EXAMPLE_NAME + '_condensed.pkl')

    # Learn optimizer
    params = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [32, 64],
        'n_epochs': [200, 300],
        'n_layers': [7, 10]
    }
    m_mlopt.load_training_data(DATA_FILE)
    m_mlopt.train(parallel=False,
                  learner=mlopt.PYTORCH,
                  n_best=10,
                  params=params)

    # Generate test trajectory and collect points
    print("Simulate loop again to get trajectory points")
    n_sim_test = 100
    P_load_test = u.P_load_profile(n_sim_test, seed=1)
    # Redefine initial data
    init_data = {'E': [7.7],
                 'z': [0.],
                 's': [0.],
                 'P': [],
                 'past_d': [np.zeros(T_horizon)],
                 'P_load': [P_load[:T_horizon]],
                 'sol': []}
    sim_data_test = u.simulate_loop(problem, init_data,
                                    u.basic_loop_solve,
                                    P_load_test,
                                    n_sim_test)
    df_test = u.sim_data_to_params(sim_data_test)

    # Evaluate performance on those parameters
    res_general, res_detail = m_mlopt.performance(df_test,
                                                  parallel=True,
                                                  use_cache=True)
    res_general.to_csv(EXAMPLE_NAME + "_general.csv")
    res_detail.to_csv(EXAMPLE_NAME + "_detail.csv")


if __name__ == '__main__':
    main()
