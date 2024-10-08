import numpy
import scipy.special as sc

#����� �������� ������ � ��������� �����
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
    def train(self, inputsList, targetsList):
        inputs = inputsList
        #inputs = numpy.array(inputsList, ndmin=2)
        targets = targetsList
        #targets = numpy.array(targetsList, ndmin=2)

        hidden_in = numpy.dot(self.wHidIn, inputs) 
        hidden_out = self.activation(hidden_in)

        output_in = numpy.dot(self.wOutHid, hidden_out)
        output_out = self.activation(output_in)

        output_error = targets - output_out

        hidden_error = numpy.dot(numpy.transpose(self.wOutHid), output_error)

        #�������� ����� � ������ ���������������� (��� ��� ���� �� ����������, �� ��������� �� � ������� expand_dims)
        self.wOutHid += self.lr * numpy.dot(numpy.expand_dims((output_error * output_out * (1.0 - output_out)), axis = 1), numpy.expand_dims(hidden_out, axis = 0))
        self.wHidIn += self.lr * numpy.dot(numpy.expand_dims((hidden_error * hidden_out * (1.0 - hidden_out)), axis = 1), numpy.expand_dims(inputs, axis = 0))

        pass

    #����� ��
    def query(self, inputsList):
        #�������� ������� �������� ��������
        #ndmin=2 - ����������� ����������� ������� (�� ���� ��������� ������)
        inputs = inputsList
        #inputs = numpy.array(inputsList, ndmin=2)

        #��������� ������������ ������� �������� �������� � �������� ����� - ��� �����
        #������� ���������  - ��� ������
        hidden_in = numpy.dot(self.wHidIn, inputs)
        hidden_out = self.activation(hidden_in)

        output_in = numpy.dot(self.wOutHid, hidden_out)
        output_out = self.activation(output_in)

        return output_out