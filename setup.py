#!/usr/bin/env python
from distutils.core import setup


# Read README.rst file
def readme():
    with open('README.rst') as f:
        return f.read()


setup(name='mlopt',
      version='0.0.1',
      description='The Machine Learning Optimizer',
      long_description=readme(),
      author='Bartolomeo Stellato, Dimitris Bertsimas',
      author_email='bartolomeo.stellato@gmail.com',
      url='https://mlopt.org/',
      packages=['mlopt',
                'mlopt.learners',
                'mlopt.tests'],
      install_requires=["cvxpy",
                        "numpy",
                        "scipy"],
      license='Apache License, Version 2.0',
      )
