import numpy

class nNetwork_class:
    #инициализация НС
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate): 
        #узлы входного, выходного, спрятанного слоёв
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wHidIn = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.wOutHid = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        #коэффициент обучения
        self.lr = learningrate
        pass

    #тренировка НС
    def train(): pass

    #опрос НС
    def query():
        
        pass