import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000


# using linear model y = W * x + b
class Model(object):
    def __init__(self):
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b


def loss(pred_y, des_y):
    return tf.reduce_mean(tf.square(pred_y - des_y))


def train(model, inputs, outputs, learning_rate, loss_func):
    with tf.GradientTape() as t:
        current_loss = loss_func(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


tf.enable_eager_execution()

model = Model()

inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])
outputs = (TRUE_W * inputs) + (TRUE_b + noise)

# plt.scatter(inputs, outputs, c='b')
# plt.scatter(inputs, model(inputs), c='r')
# plt.show()

layer = tf.keras.layers.Dense(2)
print(layer((inputs, outputs)))
