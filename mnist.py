import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2
import numpy as np


def plot_sample(labels, imgs, num=25):
    """
    Plots an array of images with their respective labels

    :param labels: numpy array of labels
    :param imgs: numpy array of images
    :param num: amount to plot, default 25
    """
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
        # img = cv2.merge((imgs[i][:, :, 2], imgs[i][:, :, 1], imgs[i][:, :, 0]))
        plt.imshow(imgs[i])
        plt.xlabel(labels[i])
    plt.show()


mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
plot_sample(train_labels, train_images)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)

loss, accuracy = model.evaluate(test_images, test_labels)
print(accuracy)

test = cv2.imread('paint.png')
test = np.array([test[:, :, 0] / 255.0])
predictions = model.predict(test)
plot_sample([np.argmax(predictions[0])], test, num=1)
