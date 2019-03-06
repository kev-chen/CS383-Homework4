#!/usr/bin/python2

import numpy
import math

data = []
mean = []
stdDev = []
D = 0

def readData():
    global data
    dataFile = open("spambase.data", "r")

    for line in dataFile:
        split = line.split(',')
        data.append(split)
    
    data = numpy.array(data).astype(float)
    dataFile.close()

def standardize(matrix):
    global mean
    global stdDev
    a = numpy.array(matrix).astype(float)
    mean = numpy.tile(numpy.mean(a, axis=0), (len(matrix),1))
    stdDev = numpy.tile(numpy.std(a, axis=0), (len(matrix),1))

    d = numpy.subtract(a, mean)
    d = numpy.divide(d, stdDev)

    return d


def likelihood(mean, std, feature):
    return (1/(std * math.sqrt(2*math.pi))) * (math.exp(-(((feature - mean) ** 2) / (2 * (std ** 2)))))

def main():
    global D
    # Read in Data
    readData()
    
    # Randomizes the data
    numpy.random.seed(0)
    X = data 
    numpy.random.shuffle(X)
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
    spam = []
    nonSpam = []
    for i in range(0, len(training)):
        if Y[i] == 1: # spam
            spam.append(training[i])
        else:
            nonSpam.append(training[i])
    spam = numpy.array(spam).astype(float)
    nonSpam = numpy.array(nonSpam).astype(float)

    # Compute models for spam
    spam_model = []
    for k in range(0, D):
        spam_model.append((numpy.mean(spam[:,k]), numpy.std(spam[:,k])))


    # Compute models for nonSpam
    nonSpam_model = []
    for k in range(0,D):
        nonSpam_model.append((numpy.mean(nonSpam[:, k]), numpy.std(nonSpam[:, k])))

    # Classify testing samples
    result = []
    for sample in testing:
        p_spam = float(len(spam)) / len(spam) + len(nonSpam)
        p_nonSpam = float(len(nonSpam)) / len(spam) + len(nonSpam)
        for k in range(0, D):
            p_spam *= likelihood(spam_model[k][0], spam_model[k][1], sample[k])
            p_nonSpam *= likelihood(nonSpam_model[k][0], nonSpam_model[k][1], sample[k])
        
        if p_spam > p_nonSpam:
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