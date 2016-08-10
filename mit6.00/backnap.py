import random

class Item(object):
    def __init__(self, n, v, w):
        self.name = n
        self.value = float(v)
        self.weight = float(w)
    def getName(self):
        return self.name
    def getValue(self):
        return self.value
    def getWeight(self):
        return self.weight
    def __str__(self):
        result = '< ' + self.name + ', ' + str(self.value) + ', '\
                 + str(self.weight) + ' >'
        return result

def buildManyItems(numItems, maxVal, maxWeight):
    Items = []
    for i in range(numItems):
        #Items.append(Item(str(i), random.randrange(1, maxVal),
        #                 random.randrange(1, maxWeight) * random.random()))
        Items.append(Item(str(i), random.randrange(1, maxVal),
                          random.randrange(1, maxWeight)))
    return Items

def solve(toConsider, avail):
    global numCalls
    numCalls += 1
    if toConsider == [] or avail == 0:
        result = (0, ())
    elif toConsider[0].getWeight() > avail:
        result = solve(toConsider[1:], avail)
    else:
        nextItem = toConsider[0]
        #Explore left branch
        withVal, withToTake = solve(toConsider[1:], avail - nextItem.getWeight())
        withVal += nextItem.getValue()
        #Explor right branch
        withoutVal, withoutToTake = solve(toConsider[1:], avail)
        #Choose better branch
        if withVal > withoutVal:
            result = (withVal, withToTake + (nextItem, ))
        else:
            result = (withoutVal, withoutToTake)
    return result

def fastSolve(toConsider, avail, memo = None):
    global numCalls
    numCalls += 1
    if memo == None:
        #initialize for first invocation
        memo = {}
    if (len(toConsider), avail) in memo:
        #Use solution found earlier
        result = memo[(len(toConsider), avail)]
        return result
    if toConsider == [] or avail == 0:
        result = (0, ())
    elif toConsider[0].getWeight() > avail:
        result = fastSolve(toConsider[1:], avail, memo)
    else:
        item = toConsider[0]
        #Consider taking first item
        withVal, withToTake = fastSolve(toConsider[1:],
                                    avail - item.getWeight(), memo)
        withVal += item.getValue()
        #Consider not taking first item
        withoutVal, withoutToTake = fastSolve(toConsider[1:], avail, memo)
        #Choose better branch
        if withVal > withoutVal:
            result = (withVal, withToTake + (item, ))
        else:
            result = (withoutVal, withoutToTake)
    #Update memo
    memo[(len(toConsider), avail)] = result
    return result

def smalltest(numItems = 10, maxVal = 20, maxWeight = 15):
    global numCalls
    numCalls = 0
    Items = buildManyItems(numItems, maxVal, maxWeight)
    val, taken = fastSolve(Items, 50)
    print 'name, value, weight'
    for item in taken:
        print(item)
    print 'Total value of items taken = ' + str(val)
    print 'Number of calls = ' + str(numCalls)

import time
import sys
sys.setrecursionlimit(2000)
def test(maxVal = 10, maxWeight = 10, runSlowly = False):
    random.seed(0)
    global numCalls
    capacity = 8 * maxWeight
    print 'numItems, numTaken, Value, Solver, numCalls, Time'
    for numItems in (4, 8, 16, 32, 64, 128, 256, 512, 1024):
        Items = buildManyItems(numItems, maxVal, maxWeight)
        if runSlowly:
            tests = (fastSolve, solve)
        else:
            tests = (fastSolve, )
        for func in tests:
            numCalls = 0
            startTime = time.time()
            val, toTake = func(Items, capacity)
            elapsed = time.time() - startTime
            funcName = func.__name__
            print numItems, len(toTake), val, funcName, numCalls, elapsed
#smalltest()
test()
