from neuralNetwork import *
from GUI import *
from dataBase import *

import numpy
import matplotlib.pyplot 


input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3

#gui = GUI()
NN = nNetwork_class(input_nodes, hidden_nodes, output_nodes, learning_rate)
trainData = dataBase("./local/dataset/mnist_test.csv")

trainData.loadData(NN)
all_values = trainData.dataList[0].split(',')
inputs = (asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
print(NN.query(inputs))

#scaled_input=(data_list[0].split(',') 
#image_array = numpy.asfarray(all_values[1:]).reshape((28,28)) 
#matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None') 
#matplotlib.pyplot.show()