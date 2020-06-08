from setuptools import setup


# Read README.rst file
def readme():
    with open('README.md') as f:
        return f.read()


setup(name='mlopt',
      version='0.0.2',
      description='The Machine Learning Optimizer',
      long_description=readme(),
      long_description_content_type='text/markdown',
      author='Bartolomeo Stellato, Dimitris Bertsimas',
      author_email='bartolomeo.stellato@gmail.com',
      url='https://mlopt.org/',
      packages=['mlopt',
                'mlopt.learners',
                'mlopt.tests'],
      install_requires=["cvxpy",
                        "optuna",
                        "numpy",
                        "scipy",
                        "pandas",
                        "joblib",
                        "tqdm",
                        "scikit-learn",
                        "gurobipy",
                        # TODO: Choose a default one to keep
                        "xgboost",
                        "torch",
                        "torchvision",
                        "pytorch-lightning",
                        "scikit-umfpack"
                        ],
      license='Apache License, Version 2.0',
      )
