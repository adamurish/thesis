import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


def plot_sample(num, images, labels):
    plt.figure(figsize=(10, 10))
    if num/5 > round(num/5):
        col = int(num/5) + 1
    else:
        col = round(num/5)
    for i in range(num):
        plt.subplot(5, col, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(labels[i])
    plt.show()


dataset = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0

plot_sample(25, test_images, test_labels)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
loss, accuracy = model.evaluate(test_images, test_labels)
print(accuracy)

print(np.argmax(model.predict(np.array([test_images[0]]))))
plt.imshow(test_images[0])
plt.show()
