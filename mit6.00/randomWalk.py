import pylab
from math import sqrt
from random import choice

class Location(object):
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
    def move(self, xc, yc):
        return Location(self.x + float(xc), self.y + float(yc))
    def getCoords(self):
        return self.x, self.y
    def getDist(self, other):
        ox, oy = other.getCoords()
        xDist = self.x - ox
        yDist = self.y - oy
        return sqrt(xDist**2 + yDist**2)
        
class CompassPt(object):
    possibles = ('n', 's', 'w', 'e')
    def __init__(self, pt):
        if pt in self.possibles: self.pt = pt
        else: raise ValueError('in CompassPt.__init__')
    def move(self, dist):
        if self.pt == 'n': return (0, dist)
        elif self.pt == 's': return (0, -dist)
        elif self.pt == 'e': return (dist, 0)
        elif self.pt == 'w': return (-dist, 0)
        else: raise ValueError('in CompassPt.move')

class Field(object):
    def __init__(self, drunk, loc):
        self.drunk = drunk
        self.loc = loc
    def move(self, cp, dist):
        oldLoc = self.loc
        xc, yc = cp.move(dist)
        self.loc = oldLoc.move(xc, yc)
    def getLoc(self):
        return self.loc
    def getDrunk(self):
        return self.drunk

class oddField(Field):
    def isChute(self):
        x, y = self.loc.getCoords()
        return abs(x) - abs(y) == 0
    def move(self, cp, dist):
        Field.move(self, cp, dist)
        if self.isChute():
            self.loc = Location(0, 0)

class Drunk(object):
    def __init__(self, name):
        self.name = name
    def move(self, field, cp, dist = 1):
        if field.getDrunk().name != self.name:
            raise ValueError('Drunk.move calling with drunk not in the field')
        for i in range(dist):
            field.move(cp, 1)

class UsualDrunk(Drunk):
    def move(self, field, dist = 1):
        cp = choice(CompassPt.possibles)
        Drunk.move(self, field, CompassPt(cp), dist)

class ColdDrunk(Drunk):
    def move(self, field, dist = 1):
        cp = choice(CompassPt.possibles)
        if cp == 's':
            Drunk.move(self, field, CompassPt(cp), 2*dist)
        else:
            Drunk.move(self, field, CompassPt(cp), dist)

class EWDrunk(Drunk):
    def move(self, field, time = 1):
        cp = choice(CompassPt.possibles)
        while cp != 'e' and cp != 'w':
            cp = choice(CompassPt.possibles)
        Drunk.move(self, field, CompassPt(cp), time)

def performTrial(time, f):
    start = f.getLoc()
    distances = [0.0]
    locs = [start]
    for t in range(1, time + 1):
        f.getDrunk().move(f)
        newLoc = f.getLoc()
        distance = newLoc.getDist(start)
        distances.append(distance)
        locs.append(newLoc)
    return distances, locs
      
#drunk = Drunk('Himer Simpson')
#for i in range(3):
#    f = Field(drunk, Location(0, 0))
#    distances = performTrial(500, f)
#    pylab.plot(distances)
#pylab.title('Himer\'s random work')
#pylab.xlabel('time')
#pylab.ylabel('Distance from original')
#
#pylab.show()
#assert False

def performSim(time, numTrials, drunkType):
    distLists = []
    locLists = []
    for trial in range(numTrials):
        d = drunkType('Drunk' + str(trial))
        f = oddField(d, Location(0, 0))
        distances, locs = performTrial(time, f)
        distLists.append(distances)
        locLists.append(locs)
    return distLists, locLists

def ansQuest(maxTime, numTrails, drunkType, title):
    means = []
    distLists, locLists = performSim(maxTime, numTrails, drunkType)
    for t in range(maxTime + 1):
        tot = 0.0
        for distL in distLists:
            tot = tot + distL[t]
        means.append(tot / len(distLists))
    pylab.figure()
    pylab.plot(means)
    pylab.ylabel('Distance')
    pylab.xlabel('steps')
    pylab.title(title)
    listX = []
    listY = []
    for locList in locLists:
        x, y = locList[-1].getCoords()
        listX.append(x)
        listY.append(y)
    pylab.figure()
    pylab.scatter(listX, listY)
    pylab.xlabel('ew distance')
    pylab.ylabel('ns diatance')
    pylab.title(title + ' Final locations')
    pylab.figure()
    pylab.hist(listX)
    pylab.xlabel('ew distance')
    pylab.ylabel('number of trials')
    pylab.title(title + ' Distribution of final locations')

if __name__ == "__main__":
    numSteps = 5000
    numTrials = 400
    ansQuest(numSteps, numTrials, UsualDrunk, 'UsualDrunk ' + str(numTrials) + ' Trials')
    #ansQuest(numSteps, numTrials, ColdDrunk, 'ColdDrunk ' + str(numTrials) + ' Trials')
    #ansQuest(numSteps, numTrials, EWDrunk, 'EWDrunk ' + str(numTrials) + ' Trials')
    pylab.show()
