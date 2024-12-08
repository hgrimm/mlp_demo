# mlp_demo

This demo and learning project demonstrates the fundamental steps of building, training, and using a simple Multi-Layer Perceptron (MLP) for digit recognition on the MNIST dataset in Go. Starting from basic implementations, it covers:

- Initializing an MLP model with configurable architecture (input, hidden, and output layers).
- Training the model on the MNIST dataset using gradient descent, backpropagation, and a suitable loss function (cross-entropy).
- Incorporating standard techniques such as ReLU activations in hidden layers and Softmax outputs to handle multi-class classification tasks.
- Implementing helper functions to load and preprocess MNIST data from IDX format, including image normalization and one-hot encoding of labels.
- Saving and loading trained model parameters (weights and biases) to/from a file for future inference.
- Demonstrating a simple inference client that loads a trained model and applies it to external input images (e.g., PNG format) to recognize handwritten digits.

This hands-on project provides a foundation for understanding basic neural networks, data handling, and model persistence in Go.

## screenshot of demo web page

<img src="screenshot_web_page.png" alt="screenshot of demo web page" width="600"/>

## training output

```
% sw_vers                           
ProductName:		macOS
ProductVersion:		15.1.1
BuildVersion:		24B91
% sysctl -n machdep.cpu.brand_string
Apple M2 Pro


```