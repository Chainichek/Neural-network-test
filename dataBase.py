from random import randint 

from numpy import asfarray
from numpy import zeros
from numpy import argmax

from neuralNetwork import nNetwork_class

#����� ������ � ����� ������ MNIST
#��� ������ ��������� numpy, random, ����� ��������� ����
class dataBase:
    def __init__(self, path):
        dataFile = open(path, 'r')#��� path - ���� � ���� ������ ������� .csv
        self.dataList = dataFile.readlines()
        dataFile.close()

        self.efficiency = -1.0#������������� ������ ����

        pass

    def loadData(self, NN, epoches):#��������� ������������� ������ � ���������, ��������� ������ ��������� ����
        while (epoches > 0):
            for data in self.dataList:
                inputs = (asfarray(data.split(',')[1:]) / 255.0 * 0.99) + 0.01#��� ������� ����� ���� ������� ����������� �� 0-255 � 0.01-1.0 ��� ������ � ���������� ���������
                #0.0 ����������, ��� ��� ����� ��������� ������� � ��������� �����

                targets = zeros(NN.onodes) + 0.01#"������ ������" (0 ����������, ����� ��), ������ ������� ������������� ���� 0, 1, 2... 
                targets[int(data[0])] = 0.99#������ ����� - �������

                NN.train(inputs, targets)
                pass
            epoches-=1
            pass
        pass
    
    def testData(self, NN, iterations):#��������� ����������������� ����, ��������� ������ ��������� ���� � ���-�� ������
        scorecard=[]#������ ������ �����������������, ��� 0 - ������� �����������, 1 - ��������

        if (iterations > len(self.dataList)):
            iterations = len(self.dataList)

        for i in range(iterations):#��� iterations - ���-�� ������
            data = self.dataList[randint(0, len(self.dataList))]#��������� ����� �� dataList
            inputs = (asfarray(data.split(',')[1:]) / 255.0 * 0.99) + 0.01
            target = int(data[0])#������ �����
            
            output_label = argmax(NN.query(inputs))#����� ����

            if (output_label == target):
                scorecard.append(1)
            else:
                scorecard.append(0)
            pass

        self.efficiency = sum(scorecard) / iterations#������� ������������� ����
        pass

    def getEfficiency(self):#���������� ������������� ����, ���� ��� �� ���� ���������, �� ���������� 0
        if (self.efficiency < 0.0):
            return 0
        else:
            return self.efficiency