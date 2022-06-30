import numpy

class nNetwork_class:
    #������������� ��
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate): 
        #���� ��������, ���������, ����������� ����
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.wHidIn = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.wOutHid = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        #����������� ��������
        self.lr = learningrate
        #������� ���������
        self.activation = lambda x: (1/1 + numpy.exp(-x))
        pass

    #���������� ��
    def train(): pass

    #����� ��
    def query(self, inputsList):
        '''
        �������� ������� �������� ��������
        ndmin=2 - ����������� ����������� ������� (�� ���� ��������� ������)
        '''
        inputs = numpy.array(inputsList, ndmin=2) 
        #��������� ������������ ������� �������� �������� � �������� �����  
        hidden_inputs = numpy.dot(self.wHidIn, inputs)
        pass