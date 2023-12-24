from collections import Counter
from utils import *


class KNNClassifier:
    """
    KNN algorithm from scratch for classification.

    Parameters:
        k - number of neighbors to use for model.
        distance_metric - either l1 for Manhattan, l2 for Euclidean, lp for Minkowski.
        weighting - Uniform or distance. If using distance, the inverse distance is used for weighting the classes.
        p - order of the norm parameter for Minkowski distance (must be set if using Minkowski Distance)

    Methods:
        fit - Train the model
        predict - Use model for inference on unseen data
    """
    def __init__(self, k, distance_metric, weighting, p=None):
        if distance_metric not in ['l1', 'l2', 'lp'] or weighting not in ['uniform', 'distance']:
            raise ValueError('Please check parameters and acceptable values.')

        elif distance_metric == 'lp' and not isinstance(p, int):
            raise ValueError('Must assign integer p value when using Minkowski distance.')

        self.k = k
        self.X = None
        self.Y = None
        self.weighting = weighting
        self.p = p
        if distance_metric == 'l1':
            self.distance = manhattan

        elif distance_metric == 'l2':
            self.distance = euclidean

        else:
            self.distance = minkowski

    def fit(self, x, y):
        self.X = x
        self.Y = y

    def predict(self, x):
        predictions = []
        for test_array in x:
            if self.distance == minkowski:
                distances = [(self.distance(test_array, train_array, self.p), label) for train_array, label in
                             zip(self.X, self.Y)]

            else:
                distances = [(self.distance(test_array, train_array), label) for train_array, label in
                             zip(self.X, self.Y)]

            k_nearest = sorted(distances, key=lambda h: h[0])[:self.k]
            labels_for_k_nearest = [label for (distance, label) in k_nearest]
            if self.weighting == 'uniform':
                predicted = Counter(labels_for_k_nearest).most_common(1)[0][0]
                predictions.append(predicted)

            else:
                weighted_distances = [(1 / (distance + 1e-10), label) for distance, label in k_nearest]
                summed_weights = {label: sum(distance for distance, lab in weighted_distances if lab == label)
                                  for label in labels_for_k_nearest}
                predicted = max(summed_weights, key=summed_weights.get)
                predictions.append(predicted)

        return predictions

