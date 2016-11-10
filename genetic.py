import math
import random
import string
import numpy as np

listofbests = []
totaldifference = []
dic = {}
bestWI = []
bestWO = []
betterdifference = 0
bettercost = 1000
savedNN = {}
bestNN = {}

sets = [ [[0,0], [0.0]],
         [[0.1, 0.1], [0.02]],
         [[0.1,0.2], [0.05]],
         [[0.2, 0.2], [0.08]],
         [[0.1,0.3], [0.10]],
         [[0.2,0.3], [0.13]]
    ]

sets2 = [(0 , 0), (0.1 , 0.1), (0.1 , 0.2), (0.2 , 0.2), (0.1 , 0.3), (0.2 , 0.3)]
#Sphere function
#Take some arbitary input
#Return it as an input for our Neural Network
def sphere(sets):
    ans = 0
    newlist = []
    for x in range(0,len(sets)):
        newlist.append([list(sets[x])])
        for y in range(0,len(sets[x])):
            ans = ans + sets[x][y]**2
            if y == 1:
                newlist[x].append([ans])
        ans = 0
    return newlist

def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

def sigmoid(x):
    return np.tanh(x)

def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, inputlayer, hiddenlayer, outputlayer):
        self.inputlayer = inputlayer
        self.hiddenlayer = hiddenlayer
        self.outputlayer = outputlayer

        self.activateInput = [1.0]*self.inputlayer
        self.activateHidden = [1.0]*self.hiddenlayer
        self.activateOutput = [1.0]*self.outputlayer
        
        self.wi = []
        self.wo = []

    def update(self, inputs):
        for i in range(self.inputlayer):
            self.activateInput[i] = inputs[i]

        for j in range(self.hiddenlayer):
            sum = 0.0
            for i in range(self.inputlayer):
                sum = sum + self.activateInput[i] * self.wi[i][j]
            self.activateHidden[j] = sigmoid(sum)

        for k in range(self.outputlayer):
            sum = 0.0
            for j in range(self.hiddenlayer):
                sum = sum + self.activateHidden[j] * self.wo[j][k]
            self.activateOutput[k] = sum

        return self.activateOutput

    def test(self, patterns):
        global allTests
        x = 0
        for p in patterns:
            x = self.update(p[0])
            print p[0], "->", x[0], ", Expected Result : ", p[1][0]

    def costfunction(self, actual, expected):
        global totaldifference
        totaldifference = []
        difference = actual[0] - expected[0]
        totaldifference.append(difference)
        averagediff = sum(totaldifference) / len(totaldifference)
        # Made a change here to stop taking average difference
        dic['AvgDiff'] = averagediff
        return (difference)**2 * 0.5

    def train(self, patterns, iterations=1, N=0.5, M=0.1):
        dic['InputWeights'] = self.wi
        dic['OutputWeights'] = self.wo
        findCost = []
        for i in range(iterations):
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                dic['Cost'] = self.costfunction(self.update(inputs),targets)
                findCost.append(dic['Cost'])
        dic['AvgCost'] = sum(findCost) / len(findCost)


def training():
    global sets
    n = NN(2, 4, 1)
    n.train(sphere(sets2))
    n.test(sphere(sets2))


def mutateI(inputlayer, hiddenlayer, generation, difference, cost):
    global bettercost
    learningRate = 0.1
    if generation == 0:
        mutatedSet = makeMatrix(inputlayer, hiddenlayer)
        for x in range(inputlayer):
            for z in range(hiddenlayer):
                mutatedSet[x][z] = random.uniform(-0.1, 0.1)
        return mutatedSet
    else:
        mutatedSet = bestWI
        if difference < 0:
            for sub_l in mutatedSet:
                rand_index = random.randrange(len(sub_l))
                sub_l[rand_index] = sub_l[rand_index] + (learningRate * bettercost)
        if difference > 0:
            for sub_l in mutatedSet:
                rand_index = random.randrange(len(sub_l))
                sub_l[rand_index] = sub_l[rand_index] - (learningRate * bettercost)
        return mutatedSet

def mutateO(hiddenlayer, outputlayer, generation, difference, cost):
    #Set weights to random if the bestweight value has not been initialized
    #Otherwise mutate weights
    global bettercost
    learningRate = 0.1
    if generation < 1:
        mutatedSet = makeMatrix(hiddenlayer, outputlayer)
        for x in range(hiddenlayer):
            for z in range(outputlayer):
                mutatedSet[x][z] = random.uniform(-1, 1)
        return mutatedSet
    else:
        mutatedSet = bestWO
        if difference < 0:
            for sub_l in mutatedSet:
                rand_index = random.randrange(len(sub_l))
                sub_l[rand_index] = sub_l[rand_index] + (learningRate * bettercost)
        if difference > 0:
            for sub_l in mutatedSet:
                rand_index = random.randrange(len(sub_l))
                sub_l[rand_index] = sub_l[rand_index] - (learningRate * bettercost)
        return mutatedSet


def Genetically(generations, population):
    global sets, bestWI, bestWO, bettercost, betterdifference, sets2

    inputlayer = 2
    hiddenlayer = 4
    outputlayer = 1
    bestNN = 0
    bestCost = 0
    smallestCost = []
    
    for t in range(0, generations):
        #Create new neural network for each generation
        nn2 = NN(inputlayer,hiddenlayer,outputlayer)
        #Clear saved NNs for new generation
        savedNN['Saved'] = []
        #Population is 500 Neural Betworks
        for x in range(0, population):
            print "Generation: ", t
            nn2.wi = mutateI(inputlayer, hiddenlayer, t, betterdifference, bettercost)
            nn2.wo = mutateO(hiddenlayer, outputlayer, t, betterdifference,bettercost)
            nn2.train(sphere(sets2))
            nn2.test(sphere(sets2))
            savedNN['Saved'].append([dic['AvgCost'],dic['InputWeights'],dic['OutputWeights'],dic['AvgDiff']])
        #Find NN with the smallest cost
        #Save it's weights and rerun
            smallestCost = min(savedNN['Saved'])
            bettercost = smallestCost[0]
            bestWI = smallestCost[1]
            bestWO = smallestCost[2]
            betterdifference = smallestCost[3]
        listofbests.append((t, smallestCost))

Genetically(200, 500)
