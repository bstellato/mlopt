from .learner import Learner
import numpy as np

#  import pandas as pd
import tensorflow as tf

# Utils
import tqdm


class NeuralNet(Learner):

    def __init__(self, n_layers, n_classes,
                 learning_rate=0.01,
                 training_epochs=50,
                 batch_size=100,
                 display_step=1):

        # Assign settings
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.display_step = display_step
        self.n_classes = n_classes

        # Define weights and biases
        self.weights = {'out': tf.Variable(tf.random_normal([n_layers[-1],
                                                             n_classes]))}
        self.biases = {'out': tf.Variable(tf.random_normal([n_classes]))}
        for i in range(len(n_layers)):
            self.weights['h%d' % i] = \
                tf.Variable(tf.random_normal([n_layers[i], n_layers[i+1]]))
            self.biases['h%d' % i] = \
                tf.Variable(tf.random_normal([n_layers[i]]))

    def neural_network(self, x, weights, biases):
        layer = x  # First layer
        for i in range(len(self.n_layers)):
            layer = tf.add(tf.matmul(layer, weights['h%d' % i]),
                           biases['b%d' % i])
            layer = tf.nn.relu(layer)
        out = tf.nn.softmax(tf.matmul(layer, weights['out']) + biases['out'])
        return out

    def train(self, X_train, y_train):
        n_train = len(X_train)

        # Create dataset from input data
        ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))

        # Repeat dataset for each epoch
        ds.repeat(self.training_epochs)

        # Create dataset batch iterator
        iterator = ds.batch(self.batch_size).make_one_shot_iterator()
        next_batch = iterator.get_next()

        #  if n_train < 100:
        #      # For small dataset use same batch size
        #      # as complete set of points.
        #      self.batch_size = n_train

        # Define neural network
        self.x = tf.placeholder("float", [None, self.n_layers[0]])
        self.y = tf.placeholder("float", [None, self.n_classes])

        # Construct model
        self.y_pred = self.neural_network(self.x, self.weights, self.biases)

        # Define loss and optimizer
        self.cost = tf.reduce_mean(-tf.reduce_sum(y_train*tf.log(self.y_pred),
                                                  reduction_indices=1))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        minimize_step = optimizer.minimize(self.cost)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            # Training cycle
            for epoch in tqdm(range(self.training_epochs)):

                avg_cost = 0.
                total_batch = int(n_train/self.batch_size)

                # Loop over all batches
                for i in range(total_batch):

                    batch_x, batch_y = sess.run(next_batch)

                    # Run optimization (backprop) and cost (loss value)
                    _, cost_value = sess.run([minimize_step, self.cost],
                                             feed_dict={self.x: batch_x,
                                                        self.y: batch_y})
                    # Compute average loss
                    avg_cost += cost_value / total_batch

                # Display logs per epoch step
                if (epoch+1) % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1),
                          "cost=", "{:.9f}".format(avg_cost))

            print("Optimization Finished!")

            # TODO: Save model!
            #  saver.save(sess, )

    def predict(self, X_pred):

        return self.predict_best(X_pred, k=1)

    def predict_best(self, X_pred, k=1):

        with tf.Session():
            # Evaluate probabilities
            proba = self.y_pred.eval({self.x: X_pred})

        # Sort probabilities
        idx_probs = np.argsort(proba)[::-1]

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
