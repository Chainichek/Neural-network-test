from neuralNetwork import nNetwork_class as NN
x = 3
y = 3
z = 3

lr = 0.3

test = NN(x, y, z, lr)

test.train([1.0, 0.5, -1.5], [0.5, 0.0, -0.5])
test.train([1.0, 0.5, -1.5], [0.5, 0.0, -0.5])

print(test.query([1.0, 0.5, -1.5]))

