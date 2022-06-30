import numpy
import scipy.special as sc

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
        #Функция активации (сигмоида)
        self.activation = lambda x: sc.expit(x)
        pass

    #тренировка НС
    def train(): pass

    #опрос НС
    def query(self, inputsList):
        #Создание матрицы вводимых значений
        #ndmin=2 - минимальная размерность матрицы (то бишь двумерный массив)
        input_in_out = numpy.array(inputsList, ndmin=2) 
        #Скалярное произведение массива вводимых значений с матрицей весов - для входа
        #Функция активации  - для выхода
        hidden_in = numpy.dot(self.wHidIn, input_in_out)
        hidden_out = self.activation(hidden_in)

        output_in = numpy.dot(self.wOutHid, hidden_out)
        output_out = self.activation(output_in)

        return output_out