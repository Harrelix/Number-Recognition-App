import numpy as np
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_cropped_data()


x_data = []
y_data = []

for train_d in training_data:
    x_data.append(train_d[0])
    y_data.append(np.argmax(train_d[1]))
del training_data
print("ft")

for test_d in test_data:
    x_data.append(test_d[0])
    y_data.append(test_d[1])
del test_data
print("ft")

for val_d in validation_data:
    x_data.append(val_d[0])
    y_data.append(val_d[1])
del validation_data
print("fv")

x_data = np.array(x_data).reshape((-1, 20, 20, 1))
y_data = np.array(y_data)

np.save('data/x_data', x_data)
np.save('data/y_data', y_data)
