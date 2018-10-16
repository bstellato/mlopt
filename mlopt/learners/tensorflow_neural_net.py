from mlopt.learners.learner import Learner
from mlopt.settings import N_BEST
from mlopt.utils import pandas2array
import tensorflow as tf
from tqdm import trange


class TensorFlowNeuralNet(Learner):

    #  def __enter__(self):
    #      """Enter for context manager"""
    #      return self
    #
    #  def __exit__(self, exc_type, exc_value, traceback):
    #      """Exit for context manager"""
    #      self.sess.close()  # Close tensorflow session

    def __init__(self, **options):
        """
        Initialize Tensorflow neural network class.

        Parameters
        ----------
        options : dict
            Learner options as a dictionary.
        """

        # TODO: This needs to go into some __enter__/__exit__
        # method.
        tf.set_random_seed(1)  # Random seed for reproducibility
        self.sess = tf.Session()  # Initialize tf session

        # Unpack settings
        self.learning_rate = options.pop('learning_rate', 0.01)
        self.n_epochs = options.pop('n_epochs', 1000)
        self.batch_size = options.pop('batch_size', 100)
        self.n_input = options.pop('n_input')
        self.n_classes = options.pop('n_classes')
        self.n_best = options.pop('n_best', N_BEST)

    def neural_network(self, x):

        n_layers = (self.n_classes + self.n_input) / 2

        # Fully connected layer (in tf contrib folder for now)
        layer1 = tf.layers.dense(x, n_layers, activation=tf.nn.relu)

        # Second layer
        layer2 = tf.layers.dense(layer1, n_layers, activation=tf.nn.relu)

        # Second layer
        layer3 = tf.layers.dense(layer2, self.n_classes)

        #  # Apply Dropout (if is_training is False, dropout is not applied)
        #  fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = layer3

        return out

    def train(self, X_train, y_train):
        self.n_train = len(X_train)

        # Unroll pandas df to array
        X_train = pandas2array(X_train)

        # Create dataset from input data
        ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))

        # Divide dataset in matches and repeat
        ds = ds.batch(self.batch_size).repeat(self.n_epochs)
        ds = ds.shuffle(buffer_size=100000000)  # Shuffle elements
        iterator = ds.make_one_shot_iterator()

        # Create dataset batch iterator
        next_batch = iterator.get_next()

        # Define neural network
        self.x = tf.placeholder("float", shape=[None, self.n_input])
        self.y = tf.placeholder("int64", shape=[None])

        # Construct model
        self.logits = self.neural_network(self.x)

        # Define loss and optimizer
        self.cost = \
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits, labels=self.y))
        #  self.cost = tf.losses.sparse_softmax_cross_entropy(self.y,
        #                                                     self.logits)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        minimize_step = optimizer.minimize(self.cost)

        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        # Run the initializer
        self.sess.run(init)

        # Training cycle
        #  for epoch in tqdm(range(self.n_epochs)):
        with trange(self.n_epochs, desc="Training neural net") as t:
            for epoch in t:

                avg_cost = 0.
                total_batch = int(self.n_train/self.batch_size)

                # Loop over all batches
                for i in range(total_batch):
                    try:
                        batch_x, batch_y = self.sess.run(next_batch)
                    except tf.errors.OutOfRangeError:
                        break

                    #  t.write("Epoch %i/%i, batch %i/%i" % (epoch,
                    #                                        self.n_epochs,
                    #                                        i, total_batch))

                    # Run optimization (backprop) and cost (loss value)
                    _, cost_value = self.sess.run([minimize_step, self.cost],
                                                  feed_dict={self.x: batch_x,
                                                             self.y: batch_y})
                    # Compute average loss
                    avg_cost += cost_value / total_batch

                # Display logs per epoch step
                t.set_description(
                    "Training neural net (epoch %4i, cost %.2e)" %
                    (epoch + 1, avg_cost))

        #  # Test model
        #  correct_prediction = tf.equal(tf.argmax(self.logits, 1), self.y)
        #  # Calculate accuracy
        #  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #  accuracy_val = self.sess.run([accuracy],
        #                               feed_dict={self.x: X_train,
        #                                          self.y: y_train})[0]
        #  print("Accuracy:", accuracy_val)

        #  print("Logits", tf.argmax(self.logits, 1).eval({self.x: X_train,
        #                                                  self.y: y_train}))

        # TODO: Save model!
        #  saver.save(sess, )

    def predict(self, X):

        # Unroll pandas df to array
        X = pandas2array(X)

        # Evaluate probabilities
        proba = tf.nn.softmax(self.logits)
        y = self.sess.run([proba], feed_dict={self.x: X})[0]

        return self.pick_best_probabilities(y)

        #  # Predict using internal model with data X
        #  # Test model
        #  correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        #  # Calculate accuracy
        #  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #  print("Accuracy:", accuracy.eval({x: mnist.test.images,
        #                                    y: mnist.test.labels}))
        #
