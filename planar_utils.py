import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
from numpy import genfromtxt

def sigmoid(x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def dataset():
    NNinput= genfromtxt('input_8.csv', delimiter=',')
    NNoutput= genfromtxt('output_8.csv', delimiter=',')
    return NNinput,NNoutput
