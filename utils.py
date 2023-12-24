import numpy as np


def manhattan(x1, x2):
    return np.sum(np.abs(x1-x2))


def euclidean(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))


def minkowski(x1, x2, p):
    return (np.sum(np.abs(x1-x2)**p))**(1/p)