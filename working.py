import math
import random
import string
import numpy as np
from sys import argv

f = open('myfile2', 'w')

totaldifference = []
dic = {}

sets = [
         [[0.1, 0.4], [0.17]],
         [[0.3, 0.3], [0.18]],
         [[0.6, 0.2], [0.40]],
         [[0.2, 0.4], [0.20]],
         [[0.2, 0.3], [0.13]],
         [[0.3, 0.4], [0.25]],
         [[0.4, 0.5], [0.41]],
         [[0.5, 0.6], [0.61]],
         [[0.1, 0.5], [0.26]],
         [[0.2, 0.4], [0.20]]
    ]

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
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni# +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = random.uniform(-0.1, 0.1)
        # print "WAHATAF", self.wi
        #print "ITH", ithweights
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = random.uniform(-1.0, 1.0)
        #print "HTO", htoweights
        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        print "WeightO", self.wo
        print "WeightI", self.wi
        if len(inputs) != self.ni:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                print "you fuckin what", self.wo
                print "WOH", self.wo[j][k]
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao

    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k]-self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error
        

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error


    def test(self, patterns):
        global allTests
        x = 0
        #p[1] is expected results
        for p in patterns:
            x = self.update(p[0])
            #allTests.append(x[0])
            print(p[0], '->', x)
            #print "Average costs", allTests


    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])


    def costfunction(self, actual, expected):
        # print "Actually", actual[0]
        # print "Expected", expected[0]
        #difference = expected[0] - actual[0]
        difference = actual[0] - expected[0]
        totaldifference.append(difference)
        averagediff = sum(totaldifference) / len(totaldifference)
        return (difference)**2 * 0.5

    def train(self, patterns, iterations=1, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        #Weights?
        dic['InputWeights'] = self.wi
        dic['OutputWeights'] = self.wo

        #print "input weight", self.wi
        #print "output weight", self.wo
        for i in range(iterations):
            #error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                dic['Cost'] = self.costfunction(self.update(inputs),targets)
                #error = error + self.backPropagate(targets, N, M)
            # if i % 100 == 0:
            #     print('error %-.5f' % error)
        #print "Dic", dic


def training():
    global sets
    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 2, 1)
    # train it with some patterns
    n.train(sets)
    # test it
    n.test(sets)
    #n.weights()

#Take a list of weights and mutate them
def mutateOutputs(m, Difference, Cost):
    easylisting = []
    returnedWeights2 = []
    #print "Weights incoming", m
    for i in range(0, 2):
        for j in range(0, 1):
            #print "Show me ya outputs", m[i][j], "\n"
            easylisting.append(m[i][j])
    assignedWeight = random.choice(easylisting)
    easylisting.remove(assignedWeight)
    if Cost > 0:
        assignedWeight =  assignedWeight - 0.5 * Cost
        print assignedWeight
    else:
        assignedWeight = assignedWeight + 0.5 * Cost
    easylisting.append(assignedWeight)
    #Rebuild our output weights back into a 2D array of values
    for i in range(0,len(easylisting)):
        returnedWeights2.append([easylisting[i]])
    return returnedWeights2


#Compare costs at some point to decide whether new network is better

def mutateInputs(m, Difference, Cost):
    print "MANDEHMS", m
    easylisting = []
    templist = []
    templist2 = []
    returnedWeights = []
    for i in range(0, 2):
        for j in range(0, 2):
            easylisting.append(m[i][j])
    assignedWeight1 = random.choice(easylisting)
    easylisting.remove(assignedWeight1)
    assignedWeight2 = random.choice(easylisting)
    easylisting.remove(assignedWeight2)
    if Cost > 0:
        assignedWeight1 = assignedWeight1 - 0.05 * Cost
        assignedWeight2 = assignedWeight2 - 0.05 * Cost
    else:
        assignedWeight1 = assignedWeight1 + 0.05 * Cost
        assignedWeight2 = assignedWeight2 + 0.05 * Cost
    easylisting.append(assignedWeight1)
    easylisting.append(assignedWeight2)
    templist.append(easylisting[0])
    templist.append(easylisting[1])
    templist2.append(easylisting[2])
    templist2.append(easylisting[3])
    returnedWeights.append(templist)
    returnedWeights.append(templist2)
            #if Difference > 0:
    # print "How many return by deaths are you on?", returnedWeights, "Um, get out I suppose\n"
    return returnedWeights


def GA(InputWeights, OutputWeights, Cost, Difference, Iterations):
    # print "~In Genetic Algorithm~"
    # print "Input Weights", InputWeights
    # print "Output Weights", OutputWeights
    # print "Cost", Cost
    #print "Difference", Difference
    global sets
    #Initialise a new Neural Network
    currentBestInput = InputWeights
    currentBestOutput = OutputWeights
    currentCost =  Cost

    for i in range(0, Iterations):
        nn = NN(2,2,1)
        #print "Show me the input weights~", InputWeights
        #print "Show me the output weights~", OutputWeights
        #print"BEFORE TRANSFORMATION", nn.wi
        nn.wi = mutateInputs(currentBestInput, Difference, Cost)
        #print"WHO ARE YOUR", nn.wi
        nn.wo = mutateOutputs(currentBestOutput, Difference, Cost)
        nn.train(sets)
        nn.test(sets)
        if dic['Cost'] < currentCost:
            f.write(str(i) + "\n")
            #f.close()
            currentBestInput = dic['InputWeights']
            currentBestOutput = dic['OutputWeights']

training()
GA(dic['InputWeights'], dic['OutputWeights'], dic['Cost'], sum(totaldifference) / len(totaldifference), 10000)