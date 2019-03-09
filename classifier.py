#!/usr/bin/python2

import numpy
import math
from util import readData, standardize, randomize


def likelihood(mean, std, feature):
    return (1/(std * math.sqrt(2*math.pi))) * (math.exp(-(((feature - mean) ** 2) / (2 * (std ** 2)))))

def main():
    # Read in Data
    data = readData("spambase.data")
    
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
    for sample in testing:
        p_positive = float(len(positive)) / len(positive) + len(negative)
        p_negative = float(len(negative)) / len(positive) + len(negative)
        for k in range(0, D):
            p_positive *= likelihood(positive_model[k][0], positive_model[k][1], sample[k])
            p_negative *= likelihood(negative_model[k][0], negative_model[k][1], sample[k])
        
        if p_positive > p_negative:
            result.append(1)
        else:
            result.append(0)
    
    # Compute statistics
    TruePositives = 0.0
    TrueNegatives = 0.0
    FalsePositives = 0.0
    FalseNegatives = 0.0
    for i in range(0, len(result)):
        if Y_testing[i] == 1: # Positive example
            if result[i] == 1: # Predicted positive
                TruePositives += 1
            elif result[i] == 0: # Predicted negative
                FalseNegatives += 1
        elif Y_testing[i] == 0: # Negative example
            if result[i] == 1: # Predicted positive
                FalsePositives += 1
            elif result[i] == 0: # Predicted negative
                TrueNegatives += 1

    try:
        precision = TruePositives / (TruePositives + FalsePositives)
        recall = TruePositives / (TruePositives + FalseNegatives)
        f_measure = (2 * precision * recall) / (precision + recall)
        accuracy = (TruePositives + TrueNegatives) / (TruePositives + TrueNegatives + FalsePositives + FalseNegatives)

        print 'Precision: ' + str(precision)
        print 'Recall: ' + str(recall)
        print 'F-measure: ' + str(f_measure)
        print 'Accuracy: ' + str(accuracy)
    except:
        pass


if __name__ == "__main__":
    main()