import NeuralNet
import Testing
import random

xorData = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]

def xor():
    # create data
    training_list = []
    testing_list = []
    for i in range(40):
        training_list.append(random.choice(xorData))
    for i in range(60):
        testing_list.append(random.choice(xorData))
    data = (training_list, testing_list)

    # result
    result = []
    stats = []
    for perceptrons in range(0, 11):
        for i in range(5):
            nnet, testAccuracy = NeuralNet.buildNeuralNet(data, maxItr=400, hiddenLayerList=[perceptrons])
            result.append(testAccuracy)
        stats.append(("Num of perceptrons: ", str(perceptrons)))
        stats.append(("MAX:          ", str(max(result))))
        stats.append(("Average:      ", str(Testing.average(result))))
        stats.append(("Std Dev:      ", str(Testing.stDeviation(result))))
        result = []
    for i in range(len(stats)):
        print (stats[i][0] + stats[i][1])
xor()
