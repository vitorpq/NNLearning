# -*- coding: utf-8 -*-
"""
Created on Tue Aug 19 22:42:33 2014

Considerações:
**19-08-2014**
//Não esquecer a formatação de entrada dos dados. O vector de entrada (inputs)
estava errado.//

Sistema simples, com Função de Erro Simples (y-a)
Otimização dos parâmetros através do método do Backpropagation somente com parada
por épocas.

Para o caso AND com layers = [2, 2, 1], precisa de eta >= 3.0. No livro foi dito
que deve ser entre 0.001 e 0.9 (Verificar com outras soluções)

Implementar outras funções de custo (Quadrática e Entropia - Logística)

Também implementar Weight Decay (Regularization)

@author: Vitor Emmanuel
"""

import numpy as np

#del j, saida, size, weights, x, y, biases, w, b

# Functions
def sigmoid(z):
    ''' Sigmoid activation function '''
    return 1.0/(1.0+np.exp(-z))
# Vectorize the function
sigmoid_vec = np.vectorize(sigmoid)

def cost_derivative(output_activations, y):
    return (output_activations - y)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

sigmoid_prime_vec = np.vectorize(sigmoid_prime)

# Backprop function
def backprop(x, y, weights, biases): # Entradas: Training sample
    # Melhorar o algoritmo pois não funciona com poucos layers ou neurons
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    # feedforward
    # feedforward
    activation = x
    activations = [x] # list to store all the activations, layer by layer
    zs = [] # list to store all the z vectors, layer by layer
    for b, w in zip(biases, weights):
        z = np.dot(w, activation)+b
        zs.append(z)
        activation = sigmoid_vec(z)
        activations.append(activation)
        # backward pass
    delta = cost_derivative(activations[-1], y) * sigmoid_prime_vec(zs[-1])
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    # Note that the variable l in the loop below is used a little
    # differently to the notation in Chapter 2 of the book.  Here,
    # l = 1 means the last layer of neurons, l = 2 is the
    # second-last layer, and so on.  It's a renumbering of the
    # scheme in the book, used here to take advantage of the fact
    # that Python can use negative indices in lists.
    for l in xrange(2, num_layers):
        z = zs[-l]
        spv = sigmoid_prime_vec(z)
        delta = np.dot(weights[-l+1].transpose(), delta) * spv
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)
    
def update_weights(mini_batch, eta, weights, biases):
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_w = [np.zeros(w.shape) for w in weights]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = backprop(x, y, weights, biases)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        weights = [w-(eta/len(mini_batch))*nw 
                            for w, nw in zip(weights, nabla_w)]
        biases = [b-(eta/len(mini_batch))*nb 
                           for b, nb in zip(biases, nabla_b)]
    return (weights, biases)
    
def evaluation(inputs, weights, biases):
    # Fazer o calculo da saida de cada camada
    output = [] # Keep the output for each training sample
    for inpu in inputs: # Cálculo para cada Training sample
        activation = inpu
        for b, w in zip(biases, weights):
            activation = sigmoid_vec(np.dot(w, activation) + b)
        output.append(activation)
    return output

# Tamanho dos Layers
sizes = [2, 2, 1]
num_layers = len(sizes)

target = [np.array([0]), np.array([0]), np.array([0]), np.array([1])]
inputs = [np.array([[0], [0]]),np.array([[0], [1]]),np.array([[1], [0]]),np.array([[1], [1]])]

# Random initialization
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
biases = [np.random.randn(y, 1) for y in sizes[1:]] # Biases weights

w = weights
b = biases
saida = []
mini_batch = zip(inputs, target)
eta = 5.0

# Run epochs for training
for j in xrange(5000): # epochs
    [w, b] = update_weights(mini_batch, eta, w, b)
    print evaluation(inputs, w, b)
