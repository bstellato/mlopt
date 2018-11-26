import numpy as np
from scipy.special import gammainc
import pandas as pd
from mlopt.strategy import encode_strategies
from mlopt.settings import SAMPLING_TOL as EPS


class Sampler(object):
    """
    Optimization problem sampler.

    Parameters
    ----------

    """

    def __init__(self,
                 problem,
                 sampling_fn=None,
                 n_samples_iter=5000,
                 max_iter=int(1e2),
                 alpha=0.97,
                 n_samples=0):
        self.problem = problem  # Optimization problem
        self.sampling_fn = sampling_fn
        self.n_samples_iter = n_samples_iter
        self.max_iter = max_iter
        self.alpha = alpha
        self.n_samples = n_samples   # Initialize numer of samples
        self.good_turing_smooth = 1.  # Initialize Good Turing estimator

    def frequencies(self, labels):
        """
        Get frequency for each unique strategy
        """
        return np.array([len(np.where(labels == i)[0])
                         for i in np.unique(labels)])

    def compute_good_turing(self, labels):
        """Compute good turing estimator"""
        # Get frequencies
        freq = self.frequencies(labels)

        # Check if there are labels appearing only once
        if not any(np.where(freq == 1)[0]):
            print("No labels appearing only once")
            n1 = 0
            #  n1 = np.inf
        else:
            # Get frequency of frequencies
            freq_freq = self.frequencies(freq)
            n1 = freq_freq[0]

        # Get Good Turing estimator
        self.good_turing = n1/self.n_samples

        # Get Good Turing estimator
        self.good_turing_smooth = self.alpha * n1/self.n_samples + \
            (1 - self.alpha) * self.good_turing_smooth

    def sample(self, epsilon=EPS, beta=1e-05):
        """
        Iterative sampling.
        """

        print("Iterative sampling")

        # Initialize dataframes
        theta = pd.DataFrame()
        s_theta = []

        # Start with 100 samples
        for self.niter in range(self.max_iter):
            # Sample new points
            theta_new = self.sampling_fn(self.n_samples_iter)
            s_theta_new = [r['strategy']
                           for r in self.problem.solve_parametric(theta_new)]
            theta = theta.append(theta_new, ignore_index=True)
            s_theta += s_theta_new
            self.n_samples += self.n_samples_iter

            # Get unique strategies
            labels, encoding = encode_strategies(s_theta)

            # Get Good Turing Estimator
            self.compute_good_turing(labels)

            print("i: %d, gt: %.2e, gt smooth: %.2e, n: %d " %
                  (self.niter+1, self.good_turing, self.good_turing_smooth,
                   self.n_samples))

            if (self.good_turing_smooth < epsilon):
                break

            #  # Get bound from theory
            #  c = 2 * np.sqrt(2) + np.sqrt(3)
            #  bound = good_turing_est
            #  bound += c * np.sqrt((1 / n_samples) * np.log(3 / beta))
            #  print("Bound ", bound)
            #  if bound < epsilon:
            #      break

        return theta, labels, encoding


def uniform_sphere_sample(center, radius, n=1):
    """
    Generate a single vector sample to x

    The function initially samples the points using a Normal Distribution with
    `randn`. Then the incomplete gamma function is used to map the points
    radially to fit in the hypersphere of finite radius r with a uniform
    spatial distribution.

    In order to have a uniform distributions over the sphere we multiply the
    vectors by f(radius): f(radius)*radius is distributed with density
    proportional to radius^n on
    [0,1].

    Parameters
    ----------
    center : numpy array
        Center of the sphere.
    radius : float
        Radius of the sphere.
    n : int, optional
        Number of samples. Default 1.

    Returns
    -------
    numpy array :
        Array of points with dimension m x n. n is the number of points and
        m the dimension.
    """
    n_dim = len(center)
    x = np.random.normal(size=(n, n_dim))
    ssq = np.sum(x ** 2, axis=1)
    fr = radius * gammainc(n_dim / 2, ssq / 2) ** (1 / n_dim) / np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n, 1), (1, n_dim))
    p = center + np.multiply(x, frtiled)
    return p

# Debug: plot points
#  from matplotlib import pyplot as plt
#  fig1 = plt.figure(1)
#  ax1 = fig1.gca()
#  center = np.array([0, 0])
#  radius = 1
#  p = uniform_sphere_sample(center, radius, 10000)
#  ax1.scatter(p[:, 0], p[:, 1], s=0.5)
#  ax1.add_artist(plt.Circle(center, radius, fill=False, color="0.5"))
#  ax1.set_xlim(-1.5, 1.5)
#  ax1.set_ylim(-1.5, 1.5)
#  ax1.set_aspect("equal")
#  plt.show()
