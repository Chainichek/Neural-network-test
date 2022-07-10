from random import randint 

from numpy import asfarray
from numpy import zeros
from numpy import argmax

from neuralNetwork import nNetwork_class

#Класс работы с базой данных MNIST
#Для работы необходим numpy, random, класс нейронной сети
class dataBase:
    def __init__(self, path):
        dataFile = open(path, 'r')#где path - путь к базе данных формата .csv
        self.dataList = dataFile.readlines()
        dataFile.close()

        self.efficiency = -1.0#эффективность работы сети

        pass

    def loadData(self, NN, epoches):#загружает тренировочные данные в нейросеть, принимает объект нейронной сети
        while (epoches > 0):
            for data in self.dataList:
                inputs = (asfarray(data.split(',')[1:]) / 255.0 * 0.99) + 0.01#для каждого числа цвет пикселя переводится из 0-255 в 0.01-1.0 для работы с активацией нейросети
                #0.0 избегается, так как может испортить подсчёт в изменении весов

                targets = zeros(NN.onodes) + 0.01#"пустой массив" (0 избегается, опять же), каждый элемент соответствует цели 0, 1, 2... 
                targets[int(data[0])] = 0.99#верный ответ - целевой

                NN.train(inputs, targets)
                pass
            epoches-=1
            pass
        pass
    
    def testData(self, NN, iterations):#оценивает работоспособность сети, принимает объект нейронной сети и кол-во тестов
        scorecard=[]#массив оценки работоспособности, где 0 - неудача определения, 1 - наоборот

        if (iterations > len(self.dataList)):
            iterations = len(self.dataList)

        for i in range(iterations):#где iterations - кол-во тестов
            data = self.dataList[randint(0, len(self.dataList))]#случайное число из dataList
            inputs = (asfarray(data.split(',')[1:]) / 255.0 * 0.99) + 0.01
            target = int(data[0])#верный ответ
            
            output_label = argmax(NN.query(inputs))#ответ сети

            if (output_label == target):
                scorecard.append(1)
            else:
                scorecard.append(0)
            pass

        self.efficiency = sum(scorecard) / iterations#подсчёт эффективности сети
        pass

    def getEfficiency(self):#Возвращает эффективность сети, если она не была расчитана, то возвращает 0
        if (self.efficiency < 0.0):
            return 0
        else:
            return self.efficiency