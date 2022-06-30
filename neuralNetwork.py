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
        #Функция активации
        self.activation = lambda x: (1/1 + numpy.exp(-x))
        pass

    #тренировка НС
    def train(): pass

    #опрос НС
    def query(self, inputsList):
        '''
        Создание матрицы вводимых значений
        ndmin=2 - минимальная размерность матрицы (то бишь двумерный массив)
        '''
        inputs = numpy.array(inputsList, ndmin=2) 
        #Скалярное произведение массива вводимых значений с матрицей весов  
        hidden_inputs = numpy.dot(self.wHidIn, inputs)
        pass