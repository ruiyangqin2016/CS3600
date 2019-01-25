from NeuralNetUtil import buildExamplesFromCarData,buildExamplesFromPenData
from NeuralNet import buildNeuralNet
import cPickle 
from math import pow, sqrt
import random

def getList(num,length):
    list = [0]*length
    list[num-1] = 1
    return list

def average(argList):
    return sum(argList)/float(len(argList))

def stDeviation(argList):
    mean = average(argList)
    diffSq = [pow((val-mean),2) for val in argList]
    return sqrt(sum(diffSq)/len(argList))

def getNNXorData(fileString="datasets/xor.txt", limit=40):
    """
    returns limit # of examples from xor data file
    """
    examples=[]
    attrValues={}
    data = open(fileString)
    attrs = ['num1','num2']
    attr_values = [['0','1'],
                   ['0','1']]

    attrNNList = [('num1', {'0' : getList(1,2), '1' : getList(2,2)}),
                  ('num2',{'0' : getList(1,2), '1' : getList(2,2)})]

    classNNList = {'0' : [1,0], '1' : [0,1]}

    for index in range(len(attrs)):
        attrValues[attrs[index]]=attrNNList[index][1]

    lineNum = 0
    for line in data:
        inVec = []
        outVec = []
        count=0
        for val in line.split(','):
            if count==2:
                outVec = classNNList[val[:val.find('\n')]]
            else:
                inVec.append(attrValues[attrs[count]][val])
            count+=1
        examples.append((inVec,outVec))
        lineNum += 1
        if (lineNum >= limit):
            break
    random.shuffle(examples)
    return examples

def buildExamplesFromXorData(size=40):
    xorData = getNNXorData()
    xorDataTrainList = []
    for cdRec in xorData:
        tmpInVec = []
        for cdInRec in cdRec[0] :
            for val in cdInRec :
                tmpInVec.append(val)
        #print "in :" + str(cdRec) + " in vec : " + str(tmpInVec)
        tmpList = (tmpInVec, cdRec[1])
        xorDataTrainList.append(tmpList)
    #print "car data list : " + str(carDataList)
    #tests = len(xorDataTrainList)-size
    #xorDataTestList = [xorDataTrainList.pop(random.randint(0,tests+size-t-1)) for t in xrange(tests)]
    xorDataTestList = xorDataTrainList
    return xorDataTrainList, xorDataTestList

# actual test
xorExamples = buildExamplesFromXorData(100)

testAccuracies = {}

for numPerceptrons in range(0, 9):
    layerAccuracy = []
    for iteration in range(0, 5):
        layerAccuracy.append(buildNeuralNet(examples=xorExamples, maxItr=200, hiddenLayerList=[numPerceptrons])[1])
    maxLayerAccuracy = max(layerAccuracy)
    averageLayerAccuracy = average(layerAccuracy)
    stdLayerAccuracy = stDeviation(layerAccuracy)

    testAccuracies[numPerceptrons] = (maxLayerAccuracy, averageLayerAccuracy, stdLayerAccuracy)
    
outputFile = open('xor-training-data.txt', mode='w+')
for numOfPerceptrons in range(0, 9):
    outputFile.write("\n\nWith %d Perceptrons in the hidden layer,\n" % numOfPerceptrons)
    outputFile.write("Max Accuracy:\t%d\n" % testAccuracies[numOfPerceptrons][0])
    outputFile.write("Average Accuracy:\t%d\n" % testAccuracies[numOfPerceptrons][1])
    outputFile.write("Standard Deviation:\t%d\n" % testAccuracies[numOfPerceptrons][2])