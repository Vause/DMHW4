from __future__ import division
from collections import defaultdict
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
import itertools

infile = sys.argv[1]
file = open(infile, 'r') #read file

#finding largest node in graph
maxNode = 0
for line in file:
    u,v = line.split()
    if(int(u) > maxNode): maxNode = int(u)
    if(int(v) > maxNode): maxNode = int(v)
file.close()
print("Number of nodes in graph: {}".format(maxNode))

#making n by n empty adjacency matrix
adjMatrix = np.zeros((maxNode, maxNode), dtype = np.int)

#filling adjacency matrix with connections
file = open(infile, 'r')
for line in file:
    u,v = line.split()
    adjMatrix[int(u) - 1][int(v) - 1] = 1
    adjMatrix[int(v) - 1][int(u) - 1] = 1
file.close()

#degree dictionary we will use for Degree Probability
degDictionary = defaultdict(list)

#pythonic way of calculating clustering coefficient
def helper(adjMtrx,index,row):
    s=[node for node in row if adjMtrx[index][node]]
    k=len(s) #degrees
    degDictionary[k].append([1 for x in range(maxNode)]) #append degrees to our degree dictionary for later use
    if k<2: #more than one degree. We want neighbors who are not connected to us
        return 0.0
    return 2.0*sum(map(lambda x:adjMtrx[x[0]][x[1]],itertools.combinations(s,2)))/k/(k-1) 
	
#using the above method to help find clustering coefficient
def calcCC(adjMtrx):
    n=len(adjMtrx)
    r=range(n)
    return sum([helper(adjMtrx,i,r) for i in r])/n #calculate clustering coefficient
	
print("Final Clustering Coefficient: {}".format(calcCC(adjMatrix)))

degLength = dict()

for x in degDictionary.keys():
    degLength[x] = len(degDictionary[x])
probabilityDict = dict()

#for each degree in our stored degrees dictionary
for x in degDictionary.keys():
    member = len(degDictionary[x]) #calculated degrees' members
    probability = member/maxNode #individual probability
    probabilityDict[x] = probability
    print("Degree {} has {} amount of members and has a probability of {}".format(x, member, probability))

	
#plot Degree Distribution
plt.scatter(sorted(degLength.keys()), degLength.values())
plt.xlabel("Degree")
plt.ylabel("Number of Nodes with Degree")
plt.title("Degree Distribution")
plt.show()
	
#plot Degree Probability
plt.loglog(sorted(probabilityDict.keys()), probabilityDict.values(), basex=2, basey=2)
plt.xlabel("Degree")
plt.ylabel("Probability")
plt.title("Degree Probability Plot")
plt.show()