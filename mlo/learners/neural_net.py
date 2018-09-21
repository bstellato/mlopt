from .learner import Learner
import numpy as np

#  import pandas as pd
import tensorflow as tf

# Utils
from tqdm import tqdm, trange


class NeuralNet(Learner):

    def __init__(self,
                 n_input,
                 n_layers,
                 n_classes,
                 learning_rate=0.01,
                 training_epochs=100,
                 batch_size=100):

        # Assign settings
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.n_input = n_input
        self.n_classes = n_classes
        self.n_layers = n_layers

        # Define weights and biases
        self.weights = {'in': tf.Variable(tf.random_normal([n_input,
                                                            n_layers[0]])),
                        'out': tf.Variable(tf.random_normal([n_layers[-1],
                                                             n_classes]))}
        self.biases = {'in': tf.Variable(tf.random_normal([n_layers[0]])),
                       'out': tf.Variable(tf.random_normal([n_classes]))}
        for i in range(len(n_layers)-1):
            self.weights['h%d' % (i+1)] = \
                tf.Variable(tf.random_normal([n_layers[i], n_layers[i+1]]))
            self.biases['h%d' % (i+1)] = \
                tf.Variable(tf.random_normal([n_layers[i+1]]))

    def neural_network(self, x, weights, biases):
        # Input
        layer = tf.add(tf.matmul(x, weights['in']), biases['in'])
        # Hidden
        for i in range(1, len(self.n_layers)):
            layer = tf.add(tf.matmul(layer, weights['h%d' % i]),
                           biases['h%d' % i])
            layer = tf.nn.relu(layer)
        # Output
        #  out = tf.nn.softmax(tf.matmul(layer, weights['out']) + biases['out'])
        out = tf.matmul(layer, weights['out']) + biases['out']
        return out

    def train(self, X_train, y_train):
        self.n_train = len(X_train)

        # Create dataset from input data
        ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))

        # Divide dataset in matches and repeat
        ds = ds.batch(self.batch_size).repeat(self.training_epochs)
        ds = ds.shuffle(buffer_size=100000000)  # Shuffle all the possible elements
        iterator = ds.make_one_shot_iterator()

        # Create dataset batch iterator
        next_batch = iterator.get_next()

        #  if n_train < 100:
        #      # For small dataset use same batch size
        #      # as complete set of points.
        #      self.batch_size = n_train

        # Define neural network
        self.x = tf.placeholder("float", shape=[None, self.n_input])
        self.y = tf.placeholder("int32", shape=[None, 1])

        # Construct model
        self.logits = self.neural_network(self.x, self.weights, self.biases)

        # Define loss and optimizer
        self.cost = tf.losses.sparse_softmax_cross_entropy(self.y,
                                                           self.logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        minimize_step = optimizer.minimize(self.cost)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            # Training cycle
            #  for epoch in tqdm(range(self.training_epochs)):
            with trange(self.training_epochs, desc="Training neural net") as t:
                for epoch in t:

                    avg_cost = 0.
                    total_batch = int(self.n_train/self.batch_size)

                    # Loop over all batches
                    for i in range(total_batch):
                        try:
                            batch_x, batch_y = sess.run(next_batch)
                        except tf.errors.OutOfRangeError:
                            break

                        # Reshape batch size
                        batch_y = np.reshape(batch_y, (-1, 1))

                        # Run optimization (backprop) and cost (loss value)
                        _, cost_value = sess.run([minimize_step, self.cost],
                                                 feed_dict={self.x: batch_x,
                                                            self.y: batch_y})
                        # Compute average loss
                        avg_cost += cost_value / total_batch

                    # Display logs per epoch step
                    t.set_description("Training neural net (epoch %i, cost %.2e)" % (epoch, avg_cost))

            # TODO: Save model!
            #  saver.save(sess, )

    def predict(self, X_pred):

        return self.predict_best(X_pred, k=1)

    def predict_best(self, x_pred, k=1):

        # Get right shape
        x = np.array([x_pred])

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            # Evaluate probabilities
            proba = tf.nn.softmax(self.logits)
            proba_pred = proba.eval({self.x: x})[0]

        # Sort probabilities
        idx_probs = np.argsort(proba_pred)[::-1]

        # Get best k indices
        return idx_probs[:k]

        #  # Predict using internal model with data X
        #  # Test model
        #  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        #  # Calculate accuracy
        #  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #  print("Accuracy:", accuracy.eval({x: mnist.test.images,
        #                                    y: mnist.test.labels}))
        #
