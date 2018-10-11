from .learner import Learner
import numpy as np
import julia
from julia.Base import Array
import julia.OptimalTrees as OT
import datetime
import os
from subprocess import call


jl = julia.Julia()
SPARSITY = jl.eval('PyCall.pyjlwrap_new(:sparsity)')


class OptimalTree(Learner):

    def __enter__(self):
        """Enter for context manager"""
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit for context manager"""

    def __init__(self,
                 sparse=False,
                 export_tree=False,
                 output_folder="output"
                 ):

        # Assign settings
        self.sparse = sparse
        self.options = {
                        'max_depth': 10,
                        #  'minbucket': 1,
                        #  'cp': 0.001
                       }
        if self.sparse:
            # Create symbol (maybe not ideal way)
            self.options['hyperplane_config'] = [{SPARSITY: 2}]
            self.options['fast_num_support_restarts'] = 10

        self.export_tree = export_tree,
        self.output_folder = output_folder

    def train(self, X, y):

        # Convert X to array
        self.n_train = len(X)
        X = self.pandas2array(X)

        # Create classifier
        self.lnr = OT.OptimalTreeClassifier(**self.options)

        # Train classifier
        OT.fit_b(self.lnr, X, y)

        # Export tree
        if self.export_tree:
            output_name = datetime.datetime.now().strftime("%y-%m-%d_%H:%M")
            export_tree_name = os.path.join(self.output_folder, output_name)
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
            print("Export tree to ", export_tree_name)
            # NB Julia call with subfolder does not work
            OT.writedot("%s.dot" % export_tree_name, self.lnr)
            #  os.rename("%s.dot" % output_na export_tree_name % export_tree_name)

            call(["dot", "-Tpdf", "-o",
                  "%s.pdf" % export_tree_name,
                  "%s.dot" % export_tree_name])

    def predict(self, X):

        return self.predict_best(X, k=1)

    def predict_best(self, X, k=1):

        #  # Get right shape
        n_points = len(X)
        X = self.pandas2array(X)

        # Evaluate probabilities
        proba = Array(OT.predict_proba(self.lnr, X))

        # Sort probabilities
        idx_probs = np.empty((n_points, k), dtype='int')
        for i in range(n_points):
            # Get best k indices
            # NB. Argsort sorts in reverse mode
            idx_probs[i, :] = np.argsort(proba[i, :])[-k:]

        return idx_probs

        #  # Predict using internal model with data X
        #  # Test model
        #  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        #  # Calculate accuracy
        #  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #  print("Accuracy:", accuracy.eval({x: mnist.test.images,
        #                                    y: mnist.test.labels}))
        #
