import math
import random
import string
import numpy as np

#Random seed keeps all of our "random" values consistent
#on each run
#random.seed(1)
best_example = []
# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

#Make matrix and fill it with zeros
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y**2


class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output neurons
        self.ni = ni # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0]*self.ni
        self.ah = [1.0]*self.nh
        self.ao = [1.0]*self.no
        
        #GA will probably run here and instead of 
        #giving the weights random values it will give them
        #the best weights possible

        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = random.uniform(-0.01,0.01)#rand(-0.02,0.02)
                #self.wi[i][j] = random.uniform(-0.1, 0.1)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = random.uniform(-1.0,1.0)
                #self.wo[j][k] = random.uniform(-1, 1)
        #print "Weights\n", self.wi

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni-1):
            #self.ai[i] = sigmoid(inputs[i])
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
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)
        
        #print "Activation output", self.ao
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
                print "WEIGHTS AT THE OUTPUT POINT", self.wo[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                print "WEIGHTS AT THE OUTPUT POINT", self.wo[j][k]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5*(targets[k]-self.ao[k])**2
        return error

    #Mutate the weight given in some way
    #Change a value slightly?
    def mutate(self, m):
        print"Starting mutation..."

    #Fitness function used in Genetic Algorithm
    #Evolves a set of weights using the best example from the set
    def fitnessFunc(self, c):
        tempweight = 0.02
        #The fitness of a weight should be decided on one of these possible
        #things:
        # ! The weight would generate a value close to the target if used
        # ! There is a small amount of error or delta when using the weight
        # ! (possibly not sure - the weight is close to the target?????)
        print "Reached fitness function"
        if c < tempweight:
            bestweight = c


    #Our evolutionary algorithm
    def GA(self, targets):
        #Looks at all weights
        #If the weight is higher than the target we need
        #Remove it and replace it with another weight we have
        #but mutate it
        #Hill Climbing
        #1. Generate random solution - c
        #2. Evaluate it's fitness using the fitness function and call it
        #   the current solution
        #3. Mutate a copy of the current solution - call this copy m and evaluate it's fitness
        #4. If the fitness of m is not worse than the fitness of c then discard c
        #5. m is now the best solution
        #6. Loop for x iterations
        #randWeightC = rand(-0.02,0.02)
        randWeightC = random.uniform(-0.01, 0.01)
        c = fitnessFunc(randWeightC)
        #Makes an altered copy of the weight and runs it through the fitness
        #function again
        m = mutate(newWeight)
        #if the fitness function(m) is better than the fitness function(c)
        #make m the new weight
        #start all over again
        #return m or c depending on what the fitness function says



        #Adjust weights at the output layer

     


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        #print "PATTERNS BABY", patterns
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                #self.GA(targets)
                error = error + self.backPropagate(targets, N, M)
            if i % 100 == 0:
                print('error %-.5f' % error)


def run():

    # Sphere function inputs and outputs
    pat = [
         [[0.1, 0.4], [0.17]],
         [[0.3, 0.3], [0.18]],
         [[0.6, 0.2],[0.4]],
         [[0.2, 0.4], [0.2]],
         [[0.2, 0.3], [0.13]],
         [[0.3, 0.4], [0.25]],
         [[0.4, 0.5], [0.41]],
         [[0.5, 0.6], [0.61]],
         [[0.1, 0.5], [0.26]],
         [[0.2, 0.4], [0.2]]
    ]
    tap = [
    [[0.3, 0.5], [0.34]]
    ]
    
    # pat = [
    #     [[0.27915179031131343, 0.18573741953065392], [0.121351783665]],
    #     [[0.46506978937161747, 0.4218904703199544], [0.112424111048]],
    #     [[0.46754408657686275, 0.30344181663414976], [0.394281477933]],
    #     [[0.4249660721753197, 0.10175962787702808], [0.310674408975]],
    #     [[0.17853012151748887, 0.33226234436685087], [0.190951184366]],
    #     [[0.3885556026175936, 0.23236467488761203], [0.142271269773]],
    #     [[0.44811235214069256, 0.21060866874580952] ,[0.110752451699]],
    #     [[0.33257990284769817, 0.2979204656512586], [0.245160691492]],
    #     [[0.3399734616020408, 0.3081432778375336] ,[0.199365995632]]
    # ]


    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 2, 1)
    # train it with some patterns
    n.weights()
    n.train(pat)
    # test it
    n.test(pat)
    n.test(tap)
    n.weights() 


if __name__ == '__main__':
    run()
