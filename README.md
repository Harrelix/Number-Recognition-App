# Number-Recognition-App
An app that uses PyQt and Tensorflow to classify handwritten digits.
## Instructions
Requires: PyQt5, Tensorflow, PIL, numpy <br/>
Run NumberApp.py. Draw a digit with mouse on the left square and click *Guess*. The LCD screen shows the digit the model think you drew, and below are the probabilities of all digits. The right square shows what the model see, which is your input after some descaling and cropping. To try again, click *Clear*.
# Method
The model used is a convolutional network with 2 convolutional layer (each with 16 filters, 5x5 kernel, MaxPooling 2x2) and a 64 dense layer. The model is trained on the [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/) (each image is cropped to size 20x20), with stochastic gradient descent optimizer and cross entropy loss.
