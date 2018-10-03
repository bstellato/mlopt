from .learner import Learner
from julia.Base import Array
import julia.OptimalTrees as OT
import datetime
import os
from subprocess import call

class OptimalTree(Learner):

    def __init__(self,
                 sparse=False,
                 export_tree=False,
                 output_folder="output"
                 ):

        # Assign settings
        self.sparse = sparse,
        self.options = {
                        'max_depth': 10,
                        'minbucket': 1,
                        'cp': 0.001
                       }
        if self.sparse:
            # Create symbol (maybe not ideal way)
            sparsity = j.eval('PyCall.pyjlwrap_new(:sparsity)'
            self.options['hyperplane_config'] = [{sparsity: 2}]
            self.options['fast_num_support_restarts'] = 10

        self.export_tree = export_tree,
        self.output_folder = output_solver

    def train(self, X_train, y_train):

        # Convert X to array
        X_train = np.array(X_train)

        # Create classifier
        self.lnr = OT.OptimalTreeClassifier(*self.options)

        # Train classifier
        OT.fit_b(self.lnr, X_train, y_train)

        # Export tree
        if self.export_tree:
            output_name = datetime.datetime.now().strftime("%y-%m-%d_%H:%M")
            export_tree_name = os.path.join(self.output_folder, output_name)
            print("Export tree to ", export_tree_name)
            OT.writedot("%s.dot" % export_tree_name, self.lnr)
            call(["dot", "-Tpdf", "-o",
                  "%s.pdf" % export_tree_name,
                  "%s.dot" % export_tree_name])

    def predict(self, X_pred):

        return self.predict_best(X_pred, k=1)

    def predict_best(self, X_pred, k=1):

        #  # Get right shape
        X_pred = np.array(X_pred)
        n_points = len(X_pred)

        # Evaluate probabilities
        proba_pred = Array(OT.predict_proba(self.lnr, X_pred))

        # Sort probabilities
        idx_probs = np.empty((n_points, k), dtype='int')
        for i in range(n_points):
            # Get best k indices
            # NB. Argsort sorts in reverse mode
            idx_probs[i, :] = np.argsort(proba_pred[i, :])[-k:]

        return idx_probs

        #  # Predict using internal model with data X
        #  # Test model
        #  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        #  # Calculate accuracy
        #  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #  print("Accuracy:", accuracy.eval({x: mnist.test.images,
        #                                    y: mnist.test.labels}))
        #
