from .learner import Learner

# TensorFlow and tf.keras
import tensorflow as tf


class NN(Learner):

    def __init__(self, n_layers, n_classes,
                 learning_rate=0.01,
                 training_epochs=25,
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

    def train(self, X, y):
        n_train = len(X)

        # Define neural network
        # tf Graph Input
        x = tf.placeholder("float", [None, self.n_layers[0]])
        y = tf.placeholder("float", [None, self.n_classes])

        # Construct model
        y_pred = self.neural_network(x, self.weights, self.biases)

        # Define loss and optimizer
        cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pred),
                                             reduction_indices=1))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        minimize_step = optimizer.minimize(cost)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Start training
        with tf.Session() as sess:

            # Run the initializer
            sess.run(init)

            # TODO: Fix batch/epochs business!
            # TODO: Continue from here
            #  # Training cycle
            #  for epoch in range(self.training_epochs):
            #      avg_cost = 0.
            #      total_batch = int(mnist.train.num_examples/batch_size)
            #      # Loop over all batches
            #      for i in range(total_batch):
            #          batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            #          # Run optimization op (backprop) and cost op (to get loss value)
            #          _, c = sess.run([minimize_step, cost], feed_dict={x: batch_xs,
            #              y: batch_ys})
            #          # Compute average loss
            #          avg_cost += c / total_batch
            #      # Display logs per epoch step
            #      if (epoch+1) % display_step == 0:
            #          print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
            #
            #  print("Optimization Finished!")
            #
            #  # Test model
            #  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
            #  # Calculate accuracy
            #  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            #  print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
            #
            #





