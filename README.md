# myMicrograd - The Good, the Bad, and the part that taught me the most...
A build of micrograd that implements the functionality present in Pytorch's backprop

# Micrograd Reimplementation

This repository contains my implementation of Andrej Karpathy’s (https://github.com/karpathy/micrograd) from scratch, as part of his neural network lectures.  
All code and explanations are based on my learning journey, see the notebook for detailed comments and experiments.

## What is micrograd?
micrograd is a minimalistic autograd engine and neural network library written in Python, designed for simplicity and clarity.  
It demonstrates the basics of backpropagation and neural networks.

## My Learning Process
- The implementation started off with trying to understand the Value object that's implemented in this autograd system.
- It then led to manual backprop through a graph made by implementing the functions in the Value object.
- The manual backprop led to a thorough understanding of the underlying logic behind optimizing a neural network.
- Subsequently, I implemented the backprop logic after multiple trials and errors.
- The final steps were to link up a group of Value objects to create a Neuron, then a Layer and finally a Multi-Layered Perceptron

This implementation broadened my understanding of neural networks by a lot and I'm looking forward to making more accurate and complicated models in the future
