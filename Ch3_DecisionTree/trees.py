from math import log
import operator
import numpy as np

def createDataSet():
    dataSet = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    
    labels = ['no surfacing', 'flippers']

    return dataSet, labels

def calcShannonEnt(dataSet):
    numEntries = len(dataSet) # total num. of dataset
    labelCounts = {}

    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1 # frequency of each class

    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)

    return shannonEnt

def calcGini(dataSet):
    numEntries = len(dataSet) # total num. of dataset
    labelCounts = {}

    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1 # frequency of each class

    prob2Sum = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        prob2Sum += prob * prob

    Gini = 1 - prob2Sum

    return Gini


def splitDataSet(dataSet, axis, value):
    retDataSet = []

    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)

    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList) # possible values for the feature i
        newEntropy = 0.0
        
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) # classify each data to corresp. class
            prob = len(subDataSet) / float(len(dataSet)) # weight
            newEntropy += prob * calcShannonEnt(subDataSet) # new entropy

        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain): # largest information gain
            bestInfoGain = infoGain
            bestFeature = i

    return bestFeature

def chooseBestFeatureToSplitUsingGainRatio(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestRatio = 0.0
    bestFeature = -1

    infoGain = []
    gainRatio = []

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList) # possible values for the feature i
        newEntropy = 0.0
        IV = 0.0
        
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) # classify each data to corresp. class
            prob = len(subDataSet) / float(len(dataSet)) # weight
            newEntropy += prob * calcShannonEnt(subDataSet) # new entropy
            IV -= prob * log (prob, 2)

        gain = baseEntropy - newEntropy
        infoGain.append(baseEntropy - newEntropy)
        gainRatio.append(gain / IV)

    if (len(infoGain) > 2):
        meanGain = np.mean(infoGain)
        logic = np.where(infoGain > meanGain)[0]

        for i in logic:
            if gainRatio[i] > bestRatio:
                bestFeature = i
    else:
        bestFeature = 0
        
    return bestFeature

def chooseBestFeatureToSplitUsingGini(dataSet):
    numFeatures = len(dataSet[0]) - 1
    # baseGini = calcGini(dataSet)
    bestGini = 100.0
    bestFeature = -1

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList) # possible values for the feature i
        newGini = 0.0
        
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value) # classify each data to corresp. class
            prob = len(subDataSet) / float(len(dataSet)) # weight
            newGini += prob * calcGini(subDataSet) # new Gini_index

        if (newGini < bestGini): # largest information gain
            bestGini = newGini
            bestFeature = i

    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1

    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]

    # all classes are equal
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # no more features to be used, return the majority
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # bestFeat = chooseBestFeatureToSplit(dataSet)
    # bestFeat = chooseBestFeatureToSplitUsingGainRatio(dataSet)
    bestFeat = chooseBestFeatureToSplitUsingGini(dataSet)
    bestFeatLabel = labels[bestFeat]

    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])

    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)

    return myTree


def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]

    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]

    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
