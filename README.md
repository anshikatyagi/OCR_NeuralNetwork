# OCR_NeuralNetwork
Neural networks are supervised learning algorithms that were extensively used before SVM and linear classification came into picture. Again in the 21st century with the advent of Deep Learning, neural networks are coming back into picture.
So this project is basically trained to detect optical numeric characters and can be used for any other detection by changing the input to the neural net.
The input layer consists of images where RGB values are extracted to give each pixel a numerical value. Therefore, input neurons are equal to the number of pixels in the image(used 8*8 pixel).
The output neurons are 10, to detect numbers from 0-9.
Input has been transformed into single array by reading the image using BufferedImage and if its white assign 0, else 1.
With 64 input neurons, 15 hidden neurons and 10 output neurons, learning rate of 0.3 and momentum of 0.6 and 1000000 iteration training of the neural network has been performed. Activation weights are initially randomised and through backpropagation and sigmoid function as the activation function, the weights are updated.
Deep Learning(using more than 1 hidden layer) can improve the prediction results but activation function needs to be changed training the first hidden and last hidden layers only.

