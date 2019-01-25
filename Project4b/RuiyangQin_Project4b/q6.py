import Testing
import NeuralNet
import NeuralNetUtil

def main():
    perceptrons = 0
    while perceptrons <= 40:
        print "===================Number of ", perceptrons, "neurons per hidden layer==================="
        i = 0
        Pen_list = []
        while i < 5:
            print "running iteration #", i+1
            nnet, testAccuracy = Testing.buildNeuralNet(Testing.penData,maxItr = 200, hiddenLayerList = [perceptrons])
            Pen_list.append(testAccuracy)
            i = i + 1
        print "| PEN TEST----------------------------------"
        print "|    Number of perceptrons is ", perceptrons
        print "|    accuracy average:", Testing.average(Pen_list)
        print "|    accuracy standard deviation:", Testing.stDeviation(Pen_list)
        print "|    max accuracy:", max(Pen_list)
        perceptrons = perceptrons + 5
        print "==========================================FINISHED========================================="
    
    print "==========================================================================================="
    print "===============================                      ======================================"
    print "===============================   Pen Test FINISHED  ======================================"
    print "===============================                      ======================================"
    print "==========================================================================================="

    perceptrons = 0
    while perceptrons <= 40:
        print "===================Number of ", perceptrons, "neurons per hidden layer==================="
        i = 0
        Car_list = []
        while i < 5:
            print "running iteration #", i+1
            nnet, testAccuracy = Testing.buildNeuralNet(Testing.carData,maxItr = 200, hiddenLayerList = [perceptrons])
            Car_list.append(testAccuracy)
            i = i + 1
        print "| CAR TEST----------------------------------"
        print "|    Number of perceptrons is ", perceptrons
        print "|    accuracy average:", Testing.average(Car_list)
        print "|    accuracy standard deviation:", Testing.stDeviation(Car_list)
        print "|    max accuracy:", max(Car_list)
        perceptrons = perceptrons + 5
        print "==========================================FINISHED========================================="
main()