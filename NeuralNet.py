import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow .keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
import mnist_loader

x, y = mnist_loader.load_cropped_data()
print(x.shape[1:])
net = Sequential()

net.add(Conv2D(16, (5, 5), input_shape=x.shape[1:]))
net.add(Activation("relu"))
net.add(MaxPooling2D(pool_size=(2, 2)))

net.add(Conv2D(16, (5, 5), input_shape=x.shape[1:]))
net.add(Activation("relu"))
net.add(MaxPooling2D(pool_size=(2, 2)))

net.add(Flatten())
net.add(Dense(64))
net.add(Activation("sigmoid"))

net.add(Dense(10, activation=tf.nn.softmax))


opt = tf.keras.optimizers.SGD(learning_rate=0.1)
net.compile(optimizer=opt,
            loss="sparse_categorical_crossentropy", metrics=["accuracy"])

net.fit(x, y, epochs=20, batch_size=10, validation_split=0.1)


# net.save("networks/2x16C + 64D.model")
