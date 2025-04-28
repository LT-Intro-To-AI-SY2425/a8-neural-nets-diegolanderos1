from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")
training_data = [
    ([0.9, 0.6, 0.8, 0.3, 0.1],[1.0]),
    ([0.8, 0.8, 0.4, 0.6, 0.4],[1.0]),
    ([0.7, 0.2, 0.4, 0.6, 0.3],[1.0]),
    ([0.5, 0.5, 0.8, 0.4, 0.8],[0.0]),
    ([0.3, 0.1, 0.6, 0.8, 0.8],[0.0]),
    ([0.6, 0.3, 0.4, 0.3, 0.6],[0.0])
]

politics = NeuralNet(5,6,1)
politics.train(training_data, learning_rate = 0.7, iters = 10000, print_interval = 1000)
print(politics.test_with_expected(training_data))

print(politics.evaluate([1.0, 1.0, 1.0, 0.1, 0.1]))
print(politics.evaluate([0.5, 0.2, 0.1, 0.7, 0.7]))
print(politics.evaluate([0.8, 0.3, 0.3, 0.3, 0.8]))
print(politics.evaluate([0.8, 0.3, 0.3, 0.8, 0.3]))
print(politics.evaluate([0.9, 0.8, 0.8, 0.3, 0.6]))