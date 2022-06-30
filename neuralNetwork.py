import numpy
import scipy.special as sc

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
        #������� ��������� (��������)
        self.activation = lambda x: sc.expit(x)
        pass

    #���������� ��
    def train(): pass

    #����� ��
    def query(self, inputsList):
        #�������� ������� �������� ��������
        #ndmin=2 - ����������� ����������� ������� (�� ���� ��������� ������)
        input_in_out = numpy.array(inputsList, ndmin=2) 
        #��������� ������������ ������� �������� �������� � �������� ����� - ��� �����
        #������� ���������  - ��� ������
        hidden_in = numpy.dot(self.wHidIn, input_in_out)
        hidden_out = self.activation(hidden_in)

        output_in = numpy.dot(self.wOutHid, hidden_out)
        output_out = self.activation(output_in)

        return output_out