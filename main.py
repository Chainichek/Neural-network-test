from neuralNetwork import nNetwork_class
from dataBase import dataBase

from GUI import *
from globals import *

import numpy
import matplotlib.pyplot

NN = nNetwork_class(input_nodes, hidden_nodes, output_nodes, learning_rate)
trainData = dataBase(path)

trainData.loadData(NN, epoches)
#trainData.testData(NN, 1000)
#print(trainData.getEfficiency())

gui = window(NN)




