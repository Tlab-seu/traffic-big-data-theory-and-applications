"""
11.2节
"""
from numpy.random import random
import numpy as np

shape_weights = [(8, 4), (4, 3), (3, 3)]
shape_bias = [(4, 1), (3, 1), (3, 1)]
def init(shape_weights, shape_bias):  
    np.random.seed(42)  
    weights = [np.mat(random(x)) for x in shape_weights]  
    bias = [np.mat(random(x)) for x in shape_bias]  
    return weights, bias  
weights, bias = init(shape_weights, shape_bias)

[x.shape for x in weights]

[x.shape for x in bias]

def sigmoidFunc(z):
    
    return 1 / (1 + np.exp(-z))

def softmaxFunc(z):
    
    return np.exp(z) / sum(np.exp(z))

def calOneLayer(a, w, b, actFunc):
    
    z = w.T * a + b
    
    a_new = actFunc(z)
    
    return z, a_new

x_test = np.mat([[3.995], [0.1771], [2.000], [3.501], [3.000], [3.509], [37.57 ], [0.09338]])
y_test = np.mat([[1]])

aList = []
zList = []

z2, a2 = calOneLayer(x_test, weights[0], bias[0], sigmoidFunc)
aList.append(a2)
zList.append(z2)

z3, a3 = calOneLayer(a2, weights[1], bias[1], sigmoidFunc)
aList.append(a3)
zList.append(z3)

z4, a4 = calOneLayer(a3, weights[2], bias[2], softmaxFunc)
aList.append(a4)
zList.append(z4)

a4