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
        pass

    #���������� ��
    def train(): pass

    #����� ��
    def query():
        
        pass