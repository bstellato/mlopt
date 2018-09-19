import numpy as np
from scipy.special import gammainc


def uniform_sphere_sample(center, radius, N=100):
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
    N : int, optional
        Number of samples. Default 100.

    Returns
    -------
    numpy array :
        Array of points with dimension m x n. n is the number of points and
        m the dimension.
    """
    n_dim = len(center)
    x = np.random.normal(size=(N, n_dim))
    ssq = np.sum(x ** 2, axis=1)
    fr = radius * gammainc(n_dim / 2, ssq / 2) ** (1 / n_dim) / np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(N, 1), (1, n_dim))
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
