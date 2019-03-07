#!/usr/bin/python2

import numpy

def readData(filename):
    try:
        dataFile = open(filename, "r")
        data = []

        for line in dataFile:
            split = line.split(',')
            data.append(split)
        
        data = numpy.array(data).astype(float)
        dataFile.close()

        return data
    except IOError as e:
        print str(e)
        quit()
    finally:
        dataFile.close()

def standardize(matrix):
    data = numpy.array(matrix).astype(float)
    mean = numpy.tile(numpy.mean(data, axis=0), (len(matrix),1))
    stdDev = numpy.tile(numpy.std(data, axis=0), (len(matrix),1))

    data = numpy.subtract(data, mean)
    data = numpy.divide(data, stdDev)

    return data