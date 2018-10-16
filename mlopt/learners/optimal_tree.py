from mlopt.learners.learner import Learner
from mlopt.settings import N_BEST
from mlopt.utils import pandas2array
import numpy as np
import datetime
import os
from subprocess import call


class OptimalTree(Learner):

    def __init__(self,
                 **options):
        """
        Initialize OptimalTrees class.

        Parameters
        ----------
        options : dict
            Learner options as a dictionary.
        """

        # Load Julia
        import julia
        from julia.Base import Array
        import julia.OptimalTrees as OT
        self.Array = Array
        self.OT = OT
        jl = julia.Julia()
        # Create sparsity symbol in python
        self.SPARSITY = jl.eval('PyCall.pyjlwrap_new(:sparsity)')

        # Assign settings
        self.sparse = options.pop('sparse', True)
        self.n_best = options.pop('n_best', N_BEST)
        self.options = {
            'max_depth': 10,
        }
        if self.sparse:
            self.options['hyperplane_config'] = [{self.SPARSITY: 2}]
            self.options['fast_num_support_restarts'] = 10

    def train(self, X, y):

        # Convert X to array
        self.n_train = len(X)
        X = self.pandas2array(X)

        # Create classifier
        self.lnr = self.OT.OptimalTreeClassifier(**self.options)

        # Train classifier
        self.OT.fit_b(self.lnr, X, y)

        # TODO: Move export tree to export function
        #  # Export tree
        #  if self.export_tree:
        #      output_name = datetime.datetime.now().strftime("%y-%m-%d_%H:%M")
        #      export_tree_name = os.path.join(self.output_folder, output_name)
        #      if not os.path.exists(self.output_folder):
        #          os.makedirs(self.output_folder)
        #      print("Export tree to ", export_tree_name)
        #      # NB Julia call with subfolder does not work
        #      self.OT.writedot("%s.dot" % export_tree_name, self.lnr)
        #      #  os.rename("%s.dot" % output_na export_tree_name % export_tree_name)
        #
        #      call(["dot", "-Tpdf", "-o",
        #            "%s.pdf" % export_tree_name,
        #            "%s.dot" % export_tree_name])
        #

    def predict(self, X):

        # Unroll pandas dataframes
        X = self.pandas2array(X)

        # Evaluate probabilities
        y = self.Array(self.OT.predict_proba(self.lnr, X))

        return self.pick_best_probabilities(y)
