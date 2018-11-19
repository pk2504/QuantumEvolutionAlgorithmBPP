#!/usr/bin/python
#
# Algorithm QEA from www.ijesi.org Volume 3 Issue 8 
#

import sys
import math
import numpy as np
import random
import copy
import time
import os
import re
import subprocess
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, Aer

class EA:
    def __init__(self):
        print ("EA __init__")
        self.t = 0
    def initialize(self):
        pass # abstract method
    def generation(self):
        pass # abstract method
    def run(self):
        self.t = 0
        self.initialize()
        while self.t < self.tmax:
            self.t += 1
            self.generation()
random.seed(1)
np.random.seed(1)

def f(x):
    #first fit to calculate number of bins to be used for packing
    bins_space=[]
    bins=0;
		
    for i in range(len(x)):
        bins_space.append(8);
		
		
    for i in range(len(x)):
        st=-1
        print(x[i])
        for j in range(bins):
            if bins_space[j] >= x[i] :
                bins_space[j] -= x[i];
                break
            st=j
        if st==bins-1 :
            bins=bins+1
            bins_space[bins-1] -= x[i];
    return bins

class rQIEA(EA):

    def __init__(self):
        self.popsize = 20
        self.dim = 10
        self.minfitness = float('inf')
        self.b = None
        self.termination = False
        self.NoFE = 0
        self.MaxNoFE = 1e4
        self.Pc = 0.05
        self.bincapacity=8
    def initialize(self):
        self.Q = np.zeros([self.popsize, 8, self.dim])# 3 qbits used so 8 possible values
        self.P = np.zeros([self.popsize, self.dim])# used for measurement of Q -> apply first fit on obtained items
        # Initialize Q(self.t) using qiskit
        for i in range(self.popsize):
            for j in range(self.dim):
                q = QuantumRegister(3)
                c = ClassicalRegister(3)
                # Create a Quantum Circuit
                qc = QuantumCircuit(q, c)
                qc.h(q)
                qc.measure(q, c)
                backend_sim = Aer.get_backend('qasm_simulator')
                job_sim = execute(qc, backend_sim)
                result_sim = job_sim.result()
                a=result_sim.get_counts(qc)
                # Show the results
                dict= ["" for x in range(9)]
                dict[1]="000"
                dict[2]="001"
                dict[3]="010"
                dict[4]="011"
                dict[5]="100"
                dict[6]="101"
                dict[7]="110"
                dict[8]="111"
                if str(result_sim)=='COMPLETED':
                    print(a)
                    for k in range(8):
                        self.Q[i][k][j]=a[dict[k+1]]

    def evaluation(self):  
        fvalues = [] # Stores number of bins used for each permutation of given item sizes measured
        for ind in self.P:
            fvalues.append(self.fitness_function(ind))
        return fvalues

    def recombination(self):
        # Recombination has to be changed
        for i in range(self.popsize):
            if random.random() < self.Pc:
                q1 = random.randint(0, self.popsize - 1)
                q2 = random.randint(0, self.popsize - 1)
                h1 = random.choice(range(self.dim + 1))
                h2 = random.choice(range(self.dim + 1))
                if h2 < h1:
                    h1, h2 = h2, h1
                temp = np.matrix(self.Q[q1], copy=True)[:,h1:h2]
                np.matrix(self.Q[q1], copy=False)[:,h1:h2] = np.matrix(self.Q[q2])[:,h1:h2] # possibly swap alphas to betas also here
                np.matrix(self.Q[q2], copy=False)[:,h1:h2] = temp

    def generation(self):
        for i in range(self.popsize):
            for j in range(self.dim):
                a=-1
                index=-1
                for k in range(8):
                    if self.Q[i][k][j]>a:
                        a=self.Q[i][k][j]
                        index=k
                self.P[i,j]=index+1
                #print(self.Q[i][k][j])
                #print(k)

        # Evaluate P(t) entire generation is in P(t)
        fvalues = self.evaluation()
        print(fvalues)
        # Select the best solution and store it into b(t)
        self.best = min(fvalues) # minmax XXX
        self.bestq = copy.deepcopy(self.Q[fvalues.index(self.best)])
        #self.recombination()
        #self.crossover()
        #self.mutation()


if __name__ == '__main__':
    rqiea = rQIEA()
    # set parameters
    rqiea.popsize = 1
    rqiea.dim = 8
    rqiea.tmax = (10000 / rqiea.popsize)
    rqiea.tmax=1
    rqiea.fitness_function = f
    rqiea.run()
    print (rqiea.best)
