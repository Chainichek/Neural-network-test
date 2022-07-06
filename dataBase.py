from numpy import asfarray
from numpy import zeros
from neuralNetwork import nNetwork_class

class dataBase:
    def __init__(self, path):
        dataFile = open(path, 'r')
        self.dataList = dataFile.readlines()
        dataFile.close()
        pass

    def loadData(self, NN):
        for data in self.dataList:
            inputs = (asfarray(data.split(',')[1:]) / 255.0 * 0.99) + 0.01
            targets = zeros(NN.onodes) + 0.01
            targets[int(data[0])] = 0.99
            NN.train(inputs, targets)
        pass




