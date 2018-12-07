# OS functions
import os
# NumPy
import numpy as np
# TensorFlow
import tensorflow as tf
# Matplotlib
import matplotlib.pyplot as plt
# Sklearn
from sklearn import preprocessing

# ---------------------
# 2. Useful functions
# ---------------------
def init_weights_bias(shape, _name):
    # Think of seeding the tf random function
    weights = tf.Variable(tf.random_normal(shape, 0, 0.1), name = "weight_" + _name)
    bias    = tf.Variable(tf.random_normal((1,shape[1]), 0, 0.1), name = "bias_" + _name)
    return weights, bias

# Define a hidden layer
def define_hidden_layer(name, input, n_input, n_hidden):
    w, b = init_weights_bias((n_input, n_hidden), name)
    tf.summary.histogram("weight " + name, w)

    for index in np.arange(0, n_input):
        tf.summary.scalar("a" + name + "_weight_" + str(index), w[index,0])

    with tf.name_scope(name):
        layer = tf.add(tf.matmul(input, w), b)
        update_moving_averages = None
        out_layer = tf.nn.leaky_relu(layer)
        # out_layer = tf.nn.softmax(layer)
        # out_layer = tf.nn.tanh(layer)

    return out_layer

# Define an output layer
def define_output_layer(name, input, n_input, n_hidden):
    w, b = init_weights_bias((n_input, n_hidden), name)
    with tf.name_scope(name):
        layer = tf.add(tf.matmul(input, w), b)
    return layer

# Define the global model
def define_model(input, x_input_size, n_hidden_1, n_hidden_2, n_hidden_3, y_output_size):

    tf.summary.histogram("input", input)
    layer_1 = define_hidden_layer("hidden_1", input, x_input_size, n_hidden_1)
    layer_2 = define_hidden_layer("hidden_2", layer_1, n_hidden_1, n_hidden_2)
    layer_3 = define_hidden_layer("hidden_3", layer_2, n_hidden_2, n_hidden_3)
    output = define_output_layer("output",   layer_3, n_hidden_3,   y_output_size)

    tf.summary.histogram("output", output)

    # output = define_output_layer("output", layer_2, n_hidden_2, y_output_size)

    return output

# Create the model
class ega_mlp_neural_network:

    # Constructor
    def __init__(self, model_name, nb_param, nl1, nl2, nl3, lr, batch_size, scaler):

        # Layer's sizes
        x_input_size = nb_param

        y_output_size = 1
        self.batch_size = batch_size
        self.model_name = model_name

        # Symbols
        self.scaler = None
        self.tf_mean  = tf.Variable(scaler.mean_,  name="scaler.mean_")
        self.tf_scale = tf.Variable(scaler.scale_, name="scaler.scale_")

        self.tf_x = tf.placeholder("float", shape=[None, x_input_size])
        self.tf_y = tf.placeholder("float", shape=[None, y_output_size])

        # Define the model
        self.nn_output = define_model(self.tf_x, nb_param, nl1, nl2, nl3, y_output_size)

        # Backward propagation
        self.cost = tf.reduce_mean(tf.squared_difference(self.tf_y, self.nn_output))
        self.optimizer_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.cost)
        self.saver = tf.train.Saver()

        tf.logging.set_verbosity(tf.logging.INFO)
        tf.summary.scalar('cost', self.cost)

        self.merged_summary_op = tf.summary.merge_all()

    # Start the machinery
    def start(self, load_session=False):
        # Start TensorFlow
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        # Get directory for events
        f_index = 0
        while os.path.exists("./data/%s" % str(f_index).zfill(3)):
            f_index = f_index + 1
        self.train_writer = tf.summary.FileWriter("./data/" + str(f_index).zfill(3), self.sess.graph)
        self.sess.run(self.init)

        # If checkpoint exists, restore the weights
        if load_session:
            checkpoint = tf.train.get_checkpoint_state("./data/" + self.model_name)
            if checkpoint and checkpoint.model_checkpoint_path:
                print("The model weights was restored from %s" % checkpoint.model_checkpoint_path)
                self.saver.restore(self.sess, checkpoint.model_checkpoint_path)

    # Split the data
    def prepare_dataset(self, X, Y, ratio = 0.2):
        indices = np.random.permutation(len(X))
        NB_ROWS = X.shape[0]
        NB_ROWS_TRAINING = int(NB_ROWS * (1 - ratio))
        training_indexes, validation_indexes = indices[:NB_ROWS_TRAINING], indices[NB_ROWS_TRAINING:]
        training_X, validation_X = X[training_indexes, :], X[validation_indexes, :]
        training_Y, validation_Y = Y[training_indexes], Y[validation_indexes]

        return training_X, validation_X, training_Y, validation_Y

    # Run the neural network gradient descent
    def run(self, training_X, validation_X, training_Y, validation_Y, max_epoch = 100):

        training_size = len(training_X)
        total_batch = int(training_size / self.batch_size)
        previous_delta =  None
        # Epoch
        for epoch in range(max_epoch):

            # Train with each example
            for i in range(total_batch - 1):

                last_index = (i + 1) * self.batch_size
                if last_index > len(training_X):
                    last_index = len(training_X)
                batch_x = training_X[i * self.batch_size:last_index]
                batch_y = np.reshape((training_Y[i * self.batch_size:last_index]), [-1, 1])

                # _, summ = self.sess.run([self.optimizer_op, self.merged_summary_op],
                #                    feed_dict={self.tf_x: batch_x, self.tf_y: np.reshape(batch_y, [-1, 1])})
                #
                _, summ = self.sess.run([self.optimizer_op, self.merged_summary_op],
                                        feed_dict={self.tf_x: batch_x, self.tf_y: batch_y})

                if i % 100 == 0:
                    self.train_writer.add_summary(summ, i * self.batch_size + epoch * training_size)

            # Training accuracy
            prediction_training = np.ravel(self.sess.run(self.nn_output, feed_dict={self.tf_x: training_X}))
            delta_ratio = abs(np.ravel(training_Y) - prediction_training)
            # train_accuracy = np.mean(np.sqrt(np.square(training_Y - prediction)) / abs(np.mean(training_Y)))
            train_accuracy = np.mean(delta_ratio)

            # Validation accuracy
            prediction = np.ravel(self.sess.run(self.nn_output, feed_dict={self.tf_x: validation_X}))
            delta_ratio = abs(np.ravel(validation_Y) - prediction)
            validation_accuracy = np.mean(delta_ratio)

            # validation_accuracy = np.mean(
            #     np.sqrt(np.square(validation_Y - prediction)) / abs(np.mean(validation_Y)))

            save_string = ""

            # Save the weights
            if epoch % 20 == 0:
                self.saver.save(self.sess, "data/" + self.model_name + "/" + self.model_name, global_step=epoch)
                save_string = "Saved - "

            # Plot
            if epoch % 500 == 0:

                # plt.scatter(np.arange(len(training_Y)), training_Y, s=1, c='blue')
                # plt.scatter(np.arange(len(training_Y)), prediction_training, s=1, c='red')

                if previous_delta is not None:
                    plt.scatter(np.arange(len(training_Y)), previous_delta, s=1, c='gray')

                plt.scatter(np.arange(len(training_Y)), prediction_training - np.ravel(training_Y), s=2, c='red')
                previous_delta = prediction_training - np.ravel(training_Y)
                plt.show()

            print(save_string + "Epoch = %d, training accuracy = %.4f validation accuracy = %.4f" %
                  (epoch + 1, train_accuracy, validation_accuracy))

    def getScaledInput(self, X_):

        if self.scaler is None:
            # Define the scaler
            self.scaler = preprocessing.StandardScaler()
            self.scaler.mean_  = self.sess.run(self.tf_mean)
            self.scaler.scale_ = self.sess.run(self.tf_scale)

        return self.scaler.transform(X_)

    def getPredictedValues(self, _input_values):

        prediction = np.ravel(self.sess.run(self.nn_output, feed_dict={self.tf_x: _input_values}))

        return prediction

    def closeSession(self):
        self.sess.close()