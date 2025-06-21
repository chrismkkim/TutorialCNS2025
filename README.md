# Computational Neuroscience Meeting 2025, Florence, Italy

# Tutorial on training recurrent spiking neural networks to generate experimentally recorded neural activities


## Overview:

Recent advances in machine learning methods make it possible to train recurrent neural networks (RNNs) to perform highly complex and sophisticated tasks. One of the tasks, particularly interesting to neuroscientists, is to generate experimentally recorded neural activities in recurrent neural networks and study the dynamics of trained networks to investigate the underlying neural mechanism.

In this tutorial, you will learn how a widely-used training method, known as recursive least squares (RLS), can be adopted to train spiking RNNs to reproduce spike recordings of cortical neurons. We first give an overview of how the original FORCE learning can be adopted to train individual unit activities in a rate-based RNN, and show it can be modified to generate arbitrarily complex activity patterns in spiking RNNs. Using this method, we show only a subset of neurons embedded in a network of randomly connected excitatory and inhibitory spiking neurons can be trained to reproduce cortical neural activities.

## Three-part tutorial:

You can find three Jupyter notebooks in this repository, one notebook for each part. The source files for simulating and training the RNNs can be found in the folders named 'src_part1', 'src_part2' and 'src_part3'.

### Part 1: Train a rate-based non-spiking RNN. 
Introduces how to set up the network training and the RLS algorithm using a rate-based RNN.

### Part 2: Train a generic spiking neural network. 
Introduces a spiking neuron model and simulates a network of spiking neurons. The RLS algorithm is modified to train spiking neural networks.

### Part 3: Train a balanced excitatory-inhibitory spiking neural network. 
Introduces spiking neural networks operating in the balanced regime and trains a subset of neurons to generate spike recordings of motor neurons.


## References:

* Sussillo, D., & Abbott, L. F. (2009). Generating coherent patterns of activity from chaotic neural networks. Neuron, 63(4), 544-557.
* Kim, C. M., & Chow, C. C. (2018). Learning recurrent dynamics in spiking networks. Elife, 7, e37124.
* Finkelstein, A., Fontolan, L., Economo, M. N., Li, N., Romani, S., & Svoboda, K. (2021). Attractor dynamics gate cortical information flow during decision-making. Nature neuroscience, 24(6), 843-850.
* Kim, C. M., Finkelstein, A., Chow, C. C., Svoboda, K., & Darshan, R. (2023). Distributing task-related neural activity across a cortical network through task-independent connections. Nature Communications, 14(1), 2851.