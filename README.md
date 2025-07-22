# Deep Learning Builds - The parts that taught me the most...
A build of micrograd that implements the functionality built by Andrej Karpathy.


## 1 - Micrograd implementation
This file contains my implementation of Andrej Karpathy’s (https://github.com/karpathy/micrograd) from scratch, as part of his neural network lectures.  
All code and explanations are based on my learning journey, see the notebook for detailed comments and experiments.

### What is micrograd?
micrograd is a minimalistic autograd engine and neural network library written in Python, designed for simplicity and clarity.  
It demonstrates the basics of backpropagation and neural networks.

### My Learning Process
- The implementation started off with trying to understand the Value object that's implemented in this autograd system.
- It then led to manual backprop through a graph made by implementing the functions in the Value object.
- The manual backprop led to a thorough understanding of the underlying logic behind optimizing a neural network.
- Subsequently, I implemented the backprop logic after multiple trials and errors.
- The final steps were to link up a group of Value objects to create a Neuron, then a Layer and finally a Multi-Layered Perceptron

## 2 - Makemore implementation (part 1)
This file contains my implementation of the makemore language model.

### What is makemore?
makemore is a bigram model built on top of a neural network, written in Python and NumPy. It demonstrates the inner workings of a letter prediction model

### My Learning Process
- The implementation started off with understanding the initial setup and correlating integers with the alphabet in a python dictionary.
- It then led to implementing a 2D matrix, counting every occurrence of bigram combinations in all 26 letters and 1 extra start-end character.
- This 2D matrix led to a clear and thorough understanding of the probability in such models and how to leverage this for maximum accuracy.
- The next step was implementing the model using neural networks, where my work with micrograd helped understand the flow of data through the neural network
- Finally, training the model using backpropagation proved to be conceptually simple. The loss function used was negative-log-likelihood.
- The last step was sampling the model and using it to generate cases, in which it did fairly well.

