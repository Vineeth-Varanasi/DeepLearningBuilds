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

## 3 - Neural Probabilistic Language Model Implementation (Bengio et al., 2003) - My blog of this project : https://medium.com/@vineethvaranasi06/a-summary-of-neural-probabilistic-language-models-bengio-et-al-2003-and-my-implementation-of-it-825cfe2403fd
This file documents my implementation of a neural probabilistic language model inspired by Bengio et al. (2003), building on lessons from Karpathy’s makemore MLP and extending the ideas from my previous bigram project.

### What Is the Neural Probabilistic Language Model?
The neural probabilistic language model (NPLM) is an architecture that uses distributed word (or character) representations in a neural network to predict the next item in a sequence. Unlike traditional bigram or n-gram statistical models, NPLMs use learned vector embeddings, allowing them to generalize better to unseen data and capture relationships between characters or words in a way that classic models cannot.

### My Learning Process
- Understanding the Curse of Dimensionality:
Started by exploring how the number of possible combinations grows exponentially with vocabulary and sequence length, which motivates the need for distributed representations.

#### Distributed Representations:
Learned how associating each character with a feature vector in an n-dimensional space addresses sparsity and enables the model to capture semantic or syntactic similarities.

- Architecture Implementation:
- Built a feed-forward neural network with three main layers:
- Embedding Layer: Converts input character indices to continuous-valued vectors.
- Hidden Layer: Processes concatenated embeddings with a non-linearity (tanh), enabling richer feature learning.
- Output Layer: Projects the hidden activations onto the vocabulary space, producing logits for the next character prediction.
- Used cross-entropy loss for optimization and backpropagation to train all parameters.

## 4 - Attention in transformer based neural networks - An implementation of the paper https://arxiv.org/pdf/1706.03762, or the core idea of it.
This file documents my implementation of the heart of modern day LLMs and VLMs. By far one of the most influential papers of our time, the "Transformer", dives deep into how we give a deep learning model better "context" to understand text for NLP, which was later extended to vision models.

### What is the transformer?
One of the biggest roadblocks to NLP was context. How relevant one token is to another and its meaning. A vector space did not solve this issue as intended, as a sentence can have any one of its words affecting the whole meaning. This context had to be given to a language model in an efficient way. Attention, in a way, is the "attention" that a token gives to another token within its context window. 

### My Learning Process
 - Understanding how context flows:
The main challenge in understanding a transformer was its mechanism of assigning "attention" to each token with respect to another. It was easier to grasp due to my work on classical ML and heatmaps. The architecture itself was fairly fascinating, and building it allowed me to make some minor tweaks and understand the overall flow of vectors from input, to "understood output".

#### Distributed Representations:
 - Inputs (words, images, sounds) are mapped to dense vectors instead of one-hot IDs.
 - Each dimension captures part of the meaning — info is spread across many neurons.
 - Similar items end up close in vector space (Paris ≈ London, king - man + woman ≈ queen).
 - Embeddings allow models to generalize across concepts they’ve never seen together.
 - Self-Attention and MHA project these embeddings into multiple subspaces, letting different heads focus on different relationships.

