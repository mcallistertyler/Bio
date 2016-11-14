import math
import random
import string
import numpy as np

#Global variables used to hold various information
#taken from the population of neural networks created
#Such as the best MLP that has run so far
#The difference in input answers from expected
#The current best cost
#The current best weights
listofbests = []
totaldifference = []
dic = {}
bestWI = []
bestWO = []
betterdifference = 0
bettercost = 1000
savedNN = {}

#Dummy data used for raw testing
sets = [ [[0,0], [0.0]],
         [[0.1, 0.1], [0.02]],
         [[0.1,0.2], [0.05]],
         [[0.2, 0.2], [0.08]],
         [[0.1,0.3], [0.10]],
         [[0.2,0.3], [0.13]]
    ]
#Example data to be input into the functions to produce sample answers
#Used to test large quantities of differing inputs
#sets2 = [(0, 0), (0.01397, 0.01489151), (0.0148941, 0.0221964), (0.0201549, 0.021564), (0.015555,0.0212), (0.0201560001, 0.03029741), (0.0251560, 0.0235450), (0.03161321, 0.0296594), (0.02058564,0.0156445), (0.012454, 0.0153489), (0.013446545, 0.0248915)]
sets2 = [(0, 0),(0.1, 0.15),(0.2, 0.2), (0.25, 0.15), (0.3, 0.3), (0.35, 0.25)]

#Used for testing completely random inputs in large quantities
#sets2 = [(random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5)), (random.uniform(0, 0.5), random.uniform(0, 0.5))]

#All continuous optimizer functions
#Used to test the constraints on the neural network
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

def rastrigin(sets):
    newlist = []
    for x in range(0,len(sets)):
        ans = 20
        newlist.append([list(sets[x])])
        for y in range(0,len(sets[x])):
        	ans = ans + (sets[x][y]**2 - 10*math.cos(2 * math.pi * sets[x][y]))
        	if y == 1:
        		newlist[x].append([ans])
        		ans = 0
    return newlist

def rosenbrock(sets):
    newlist = []
    ans = 0
    for x in range(0,len(sets)):
        newlist.append([list(sets[x])])
        for y in range(0,len(sets[x])):
            if y == 0:
                ans = 100*(sets[x][y+1] - sets[x][y]**2)**2 + ((sets[x][y] - 1)**2)
            if y == 1:
                newlist[x].append([ans])
                
    return newlist

def schnefel(sets):
	ans = 0
	newlist = []
	for x in range(0,len(sets)):
		newlist.append([list(sets[x])])
		for y in range(0, len(sets[x])):
			if y == 0:
				ans = 418.9829 * 2 - sets[x][0] * math.sin(math.sqrt(abs(sets[x][0]))) + sets[x][1] * math.sin(math.sqrt(abs(sets[x][1])))
				newlist[x].append([ans])
			ans = 0
	return newlist

def diffpowers(sets):
	print 
	ans = 0
	newlist = []
	for x in range(0, len(sets)):
		newlist.append([list(sets[x])])
		for y in range(0, len(sets[x])):
			if y == 0:
				ans = math.sqrt(abs(sets[x][0])**(2+4*(0/1)) + abs(sets[x][1])**(2+4*(1/1)))
				newlist[x].append([ans])
			ans = 0
	return newlist


def makeMatrix(I, J):
	#Make a 2D list from two inputs
    m = []
    for i in range(I):
        m.append([0.0]*J)
    return m

#Various activation functions
def ReLU(x):
        return x * (x > 0)

def tanh(x):
    return np.tanh(x)

#Derivative sigmoid function
#Unused in GA but can be used in backpropagation
def dsigmoid(y):
    return 1.0 - y**2

class NN:
    def __init__(self, inputlayer, hiddenlayer, outputlayer):
    	#Initialize class variables
    	#Number of neurons
        self.inputlayer = inputlayer
        self.hiddenlayer = hiddenlayer
        self.outputlayer = outputlayer
        #Activation functions
        self.activateInput = [1.0]*self.inputlayer
        self.activateHidden = [1.0]*self.hiddenlayer
        self.activateOutput = [1.0]*self.outputlayer
        #Weights which will be set during the start of the Genetic algorithm
        self.wi = []
        self.wo = []

    def feedForward(self, inputs):
    	#Forward pass through the input layer to the hidden layer
    	#Pass from the hidden layer to the output layer using a chosen
    	#activation function. Then return our answers
        for i in range(self.inputlayer):
            self.activateInput[i] = inputs[i]

        for j in range(self.hiddenlayer):
            sum = 0.0
            for i in range(self.inputlayer):
                sum = sum + self.activateInput[i] * self.wi[i][j]
            self.activateHidden[j] = ReLU(sum)

        for k in range(self.outputlayer):
            sum = 0.0
            for j in range(self.hiddenlayer):
                sum = sum + self.activateHidden[j] * self.wo[j][k]
            self.activateOutput[k] = sum

        return self.activateOutput

    def test(self, patterns):
    	#Print out the training inputs, the values
    	#given by the neural network and the expected values to reach
        global allTests
        x = 0
        for p in patterns:
            x = self.feedForward(p[0])
            print p[0], "->", x[0], ", Expected Result : ", p[1][0]

    def finaltest(self, patterns):
    	#Run final test to return relevant outputs to COCO
        global allTests
        x = 0
        finalAnswers = []
        for p in patterns:
            x = self.feedForward(p[0])
            print p[0],  x[0], "Expected : ", p[1][0]
            finalAnswers.append(x[0])
        return finalAnswers

    def costfunction(self, actual, expected):
    	#Calculate the cost of our neural network
    	#The smaller the cost the closer we are to our desired answers
        global totaldifference
        totaldifference = []
        difference = actual[0] - expected[0]
        totaldifference.append(difference)
        averagediff = sum(totaldifference) / len(totaldifference)
        dic['AvgDiff'] = averagediff
        return (difference)**2 * 0.5
    #Dictionary created allows me to easily store important information
    #About each Neural Network
    def train(self, patterns, iterations=1):
        dic['InputWeights'] = self.wi
        dic['OutputWeights'] = self.wo
        findCost = []
        for i in range(iterations):
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.feedForward(inputs)
                dic['Cost'] = self.costfunction(self.feedForward(inputs),targets)
                findCost.append(dic['Cost'])
        dic['AvgCost'] = sum(findCost) / len(findCost)


def mutateI(inputlayer, hiddenlayer, generation, difference, cost):
	#Set the weights to random values if the current best value has not been initialized
	#Otherwise mutate the best weights we have
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
    #Otherwise mutate the weights
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


def Genetically(generations, population, sets2):
    global sets, bestWI, bestWO, bettercost, betterdifference
    #Set the number of nodes at each layer here
    inputlayer = 2
    hiddenlayer = 3
    outputlayer = 1
    bestCost = 0
    smallestCost = []
    for t in range(0, generations):
        #Create new neural network for each generation
        nn2 = NN(inputlayer,hiddenlayer,outputlayer)
        #Clear saved NNs for new generation
        savedNN['Saved'] = []
        #Population is 500 Neural Networks
        for x in range(0, population):
            print "Generation: ", t
            nn2.wi = mutateI(inputlayer, hiddenlayer, t, betterdifference, bettercost)
            nn2.wo = mutateO(hiddenlayer, outputlayer, t, betterdifference,bettercost)
            nn2.train(sphere(sets2))
            nn2.test(sphere(sets2))
            #Save relevant information into a dictionary
            savedNN['Saved'].append([dic['AvgCost'],dic['InputWeights'],dic['OutputWeights'],dic['AvgDiff']])
        	#Find NN with the smallest cost and save its weights
            smallestCost = min(savedNN['Saved'])
            bettercost = smallestCost[0]
            bestWI = smallestCost[1]
            bestWO = smallestCost[2]
            betterdifference = smallestCost[3]
        listofbests.append((t, smallestCost))
    if t == generations - 1:
        print"Finished! Final generations best member..."
        return nn2.finaltest(sphere(sets2))

#Generations and population and the training input set can be changed here
Genetically(200, 500, sets2)
