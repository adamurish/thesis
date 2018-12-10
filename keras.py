from tensorflow import keras
import numpy as np
import tensorflow as tf

BATCH_SIZE = 100
TRUE_M = 7.0

tf.enable_eager_execution()
model = keras.Sequential()

inputs = tf.random_normal((BATCH_SIZE, 1))
outputs = TRUE_M * inputs
test_inputs = tf.random_normal((BATCH_SIZE * 2, 1))
test_outputs = TRUE_M * test_inputs

model.add(keras.layers.Dense(10, input_shape=(BATCH_SIZE, 1)))
model.add(keras.layers.Dense(1))
print(model.layers)
model.compile(loss=keras.losses.mean_squared_error, optimizer=tf.train.AdamOptimizer(learning_rate=0.1))
model.fit(inputs, outputs, epochs=50, batch_size=32)
print(model.predict(test_inputs))
