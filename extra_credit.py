#!/usr/bin/python2

import numpy
import math
import matplotlib.pyplot as plt
from util import readData, standardize, randomize


def likelihood(mean, std, feature):
    return (1/(std * math.sqrt(2*math.pi))) * (math.exp(-(((feature - mean) ** 2) / (2 * (std ** 2)))))

def normalize_probabilities(probabilities):
    prob_factor = 1 / sum(probabilities)
    return [prob_factor * p for p in probabilities]


def main():
    # Read in Data
    data = readData("nba_stats.csv")
    
    # Randomizes the data
    X = randomize(data)
    Y = X[:,-1] # Only the last column
    X = X[:,:-1] # All but the last column
    D = len(X[0])

    # Standardize
    standardized = standardize(X)

    # Select first 2/3 for training
    index = int(math.ceil((2.0/3.0) * len(X)))
    training = standardized[:index+1]
    testing = standardized[index+1:]
    Y_testing = Y[index+1:]

    # Divide training data into two groups
    positive = []
    negative = []
    for i in range(0, len(training)):
        if Y[i] == 1: # spam
            positive.append(training[i])
        else:
            negative.append(training[i])
    positive = numpy.array(positive).astype(float)
    negative = numpy.array(negative).astype(float)

    # Compute models for spam
    positive_model = []
    for k in range(0, D):
        positive_model.append((numpy.mean(positive[:,k]), numpy.std(positive[:,k])))

    # Compute models for non-spam
    negative_model = []
    for k in range(0,D):
        negative_model.append((numpy.mean(negative[:, k]), numpy.std(negative[:, k])))

    # Classify testing samples
    result = []
    testing_probabilities = []
    for sample in testing:
        p_positive = float(len(positive)) / len(positive) + len(negative)
        p_negative = float(len(negative)) / len(positive) + len(negative)
        for k in range(0, D):
            p_positive *= likelihood(positive_model[k][0], positive_model[k][1], sample[k])
            p_negative *= likelihood(negative_model[k][0], negative_model[k][1], sample[k])
        
        testing_probabilities.append(normalize_probabilities([p_positive, p_negative]))
        
        if p_positive > p_negative:
            result.append(1)
        else:
            result.append(0)
    
    precisions = []
    recalls = []
    for threshold in range(0, 100, 5):
        threshold = float(threshold) / 100

        TruePositives = 0.0
        TrueNegatives = 0.0
        FalsePositives = 0.0
        FalseNegatives = 0.0
        for i in range(0, len(testing_probabilities)):
            if Y_testing[i] == 1: # Positive example
                if testing_probabilities[i][0] > threshold: # Predicted positive
                    TruePositives += 1
                else: # Predicted negative
                    FalseNegatives += 1
            elif Y_testing[i] == 0: # Negative example
                if testing_probabilities[i][0] > threshold: # Predicted positive
                    FalsePositives += 1
                else: # Predicted negative
                    TrueNegatives += 1

        try:
            precision = TruePositives / (TruePositives + FalsePositives)
        except ZeroDivisionError:
            if TruePositives == 0:
                precision = 1
            else:
                precision = 0
        
        try:
            recall = TruePositives / (TruePositives + FalseNegatives)
        except ZeroDivisionError:
            if TruePositives == 0:
                recall = 1
            else:
                recall = 0
        
        precisions.append(precision)
        recalls.append(recall)

    plt.plot(recalls, precisions, 'r-o')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()



if __name__ == "__main__":
    main()